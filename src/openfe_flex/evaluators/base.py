"""Backend contract. The search engine owns folds, losses and pruning."""
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class EvaluationContext:
    task: str
    metric: str
    n_classes: int
    seed: int


@runtime_checkable
class Evaluator(Protocol):
    """Implement for a custom backend; fit must return a fresh fitted model.

    Labels are encoded by the engine. Baselines are raw margins, not probabilities.
    predict_raw must include the supplied baseline exactly once. importance
    returns one score per input column, excluding any SHAP bias term. Built-in
    evaluators keep configuration only and do not retain fitted models. Custom
    evaluators must be pickleable when the engine uses multiprocessing.
    """

    supported_importances: tuple[str, ...]

    def check_available(self) -> None: ...

    def fit(self, data, labels, *, validation, baseline, context: EvaluationContext,
            stage: str, early_stopping_rounds: int) -> Any: ...

    def predict_raw(self, model, data, *, baseline=None) -> np.ndarray: ...

    def importance(self, model, data, *, method: str,
                   context: EvaluationContext) -> np.ndarray: ...


def numeric_data(data):
    return data.astype(float).replace([np.inf, -np.inf], np.nan)


def with_baseline(raw, baseline):
    """Add external margins without broadcasting binary/regression into n x n."""
    raw = np.asarray(raw)
    if baseline is None:
        return raw
    corrected = raw.reshape(len(raw), -1) + np.asarray(baseline).reshape(len(raw), -1)
    return corrected.ravel() if raw.ndim == 1 else corrected

