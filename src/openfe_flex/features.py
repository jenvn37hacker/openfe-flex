"""Pointwise candidate generation and formula transforms, independent of models."""
from copy import deepcopy
from itertools import combinations
import random

import numpy as np
import pandas as pd

from ._upstream.FeatureGenerator import Node, FNode
from ._upstream.utils import tree_to_formula, formula_to_tree

UNARY = {"abs", "log", "sqrt", "square", "sigmoid", "round", "residual"}
BINARY = {"+", "-", "*", "/", "min", "max"}
SYMMETRIC = {"+", "*", "min", "max"}


DEFAULT_OPERATORS = ("abs", "log", "sqrt", "square", "sigmoid", "round", "residual",
                     "min", "max", "+", "-", "*", "/")


def expand_candidates(base_nodes, *, frontier=None, max_candidates=None, seed=42,
                      operators=DEFAULT_OPERATORS, excluded=()):
    """Enumerate pointwise expansions; optionally reservoir-sample a budget.

    A higher-order round requires at least one selected frontier expression.
    Formula ASTs always refer to original source columns, never internal aliases.
    """
    if max_candidates is not None and (isinstance(max_candidates, bool)
            or not isinstance(max_candidates, int) or max_candidates < 1):
        raise ValueError("max_candidates must be None or a positive integer")
    operators = tuple(operators)
    if not operators or len(set(operators)) != len(operators) or not set(operators) <= UNARY | BINARY:
        raise ValueError("Operators must be unique supported pointwise operators")
    base_nodes = list(base_nodes)
    formulas = [tree_to_formula(node) for node in base_nodes]
    if len(set(formulas)) != len(formulas):
        raise ValueError("Duplicate base expressions")
    active = set(formulas if frontier is None else (tree_to_formula(node) for node in frontier))
    excluded = set(excluded)
    def nodes():
        for op in operators:
            if op in UNARY:
                for node, formula in zip(base_nodes, formulas):
                    if formula in active:
                        yield Node(op, [deepcopy(node)])
            else:
                for (first, a), (second, b) in combinations(zip(base_nodes, formulas), 2):
                    if a not in active and b not in active:
                        continue
                    yield Node(op, [deepcopy(first), deepcopy(second)])
                    if op not in SYMMETRIC:
                        yield Node(op, [deepcopy(second), deepcopy(first)])
    rng = random.Random(seed)
    result = []
    count = 0
    for node in nodes():
        if tree_to_formula(node) in excluded:
            continue
        count += 1
        if max_candidates is None or count <= max_candidates:
            result.append(node)
        else:
            position = rng.randrange(count)
            if position < max_candidates:
                result[position] = node
    return result


def _validate_numeric_columns(columns):
    columns = list(columns)
    if len(set(columns)) != len(columns) or not all(isinstance(c, str) and c for c in columns):
        raise ValueError("Feature names must be nonempty unique strings")
    if any(any(char in c for char in "()+-*/,\n\r") for c in columns):
        raise ValueError("Use simple feature IDs, e.g. S1_rms, to serialize formulas")
    return columns


def numeric_candidates(columns, *, max_candidates=None, seed=42, operators=DEFAULT_OPERATORS):
    """All first-order formulas in the supported numeric space by default.

    A finite budget explicitly enables reproducible reservoir sampling. Neither
    this function nor the numeric engine includes frequency/categorical operators.
    """
    columns = _validate_numeric_columns(columns)
    return expand_candidates([FNode(name) for name in columns], max_candidates=max_candidates,
                             seed=seed, operators=operators)


def _validate_formula(node):
    if isinstance(node, FNode):
        return
    if node.name not in UNARY | BINARY:
        raise ValueError(f"Non-pointwise operator {node.name!r} is not supported")
    for child in node.children:
        _validate_formula(child)


def transform_manifest(data, manifest):
    """Compute generated columns only; no fitting or joint Train/Test statistics."""
    if not isinstance(data, pd.DataFrame) or not data.columns.is_unique:
        raise ValueError("Provide a DataFrame with unique named columns")
    output = {}
    for entry in manifest:
        node = formula_to_tree(entry["formula"])
        _validate_formula(node)
        required = node.get_fnode()
        if set(required) != set(entry["source_features"]):
            raise ValueError("Manifest dependencies do not match the formula")
        if not set(required) <= set(data.columns):
            raise ValueError(f"Missing source features for {entry['name']}")
        if entry["name"] in output:
            raise ValueError("Duplicate generated feature ID")
        source = data.loc[:, required].copy().astype(float).replace([np.inf, -np.inf], np.nan)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            output[entry["name"]] = node.calculate(source).replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame(output, index=data.index)


