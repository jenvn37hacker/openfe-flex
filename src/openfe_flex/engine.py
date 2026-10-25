"""Shared OpenFE search engine with an injected model evaluator."""
from __future__ import annotations

import json
from copy import deepcopy
from time import perf_counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm.auto import tqdm

from ._upstream.openfe import OpenFE as _UpstreamOpenFE, _subsample
from ._upstream.FeatureGenerator import FNode
from ._upstream.utils import tree_to_formula, formula_to_tree
from .evaluators.base import EvaluationContext, Evaluator
from .features import (numeric_candidates, expand_candidates, transform_manifest,
                       _validate_formula, _validate_numeric_columns, DEFAULT_OPERATORS)

class OpenFE(_UpstreamOpenFE):
    """Numeric OpenFE search using an explicitly injected evaluator.

    Call fit on outer Train only. train_index/val_index describe an internal
    development split. baseline_cv may contain positional, group-disjoint OOF
    folds; every row must be validated once. Test/Data B must not be passed here.
    """
    def __init__(self, *, evaluator: Evaluator):
        if not isinstance(evaluator, Evaluator):
            raise TypeError("evaluator must implement the Evaluator protocol")
        self.evaluator = evaluator
        self._stage2_metrics = tuple(evaluator.supported_importances)

    def fit(self, data, label, *, task="classification", feature_boosting=True,
            stage2_metric="shap", baseline_cv=None, max_candidates=None,
            operators=DEFAULT_OPERATORS, max_order=1, n_new_features=10, **kwargs):
        """Select top features per order; higher orders extend the accepted basis.

        Returned nodes are ranked within each order, then concatenated in order.
        n_new_features=None returns all ranked survivors for a single order only.
        """
        if isinstance(max_order, bool) or not isinstance(max_order, int) or max_order < 1:
            raise ValueError("max_order must be a positive integer")
        if n_new_features is not None and (isinstance(n_new_features, bool)
                or not isinstance(n_new_features, int) or n_new_features < 1):
            raise ValueError("n_new_features must be None or a positive integer")
        if max_order > 1 and (n_new_features is None or kwargs.get("init_scores") is not None
                              or kwargs.get("candidate_features_list") is not None):
            raise ValueError("Higher orders require a finite top-k and automatic candidates/baseline")
        self.evaluator.check_available()
        baseline_cv = None if baseline_cv is None else list(baseline_cv)
        # Validate source IDs even if the caller supplies candidates.
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a DataFrame")
        _validate_numeric_columns(data.columns)
        operators = tuple(operators)
        started = perf_counter()
        raw = data.copy()
        working = raw.copy()
        selected = []
        basis = [FNode(name) for name in raw.columns]
        frontier = None
        seen = set()
        self.order_history_ = []
        self.stage2_model_ = None
        self.stage2_columns_ = []
        self.stage2_importances_ = np.empty(0)
        if n_new_features is None and max_order != 1:
            raise ValueError("Use an explicit top-k for higher orders")
        for order in range(1, max_order + 1):
            order_started = perf_counter()
            round_kwargs = dict(kwargs)
            if round_kwargs.get("candidate_features_list") is None:
                round_kwargs["candidate_features_list"] = expand_candidates(
                    basis, frontier=frontier, max_candidates=max_candidates,
                    seed=round_kwargs.get("seed", 42) + order - 1,
                    operators=operators, excluded=seen)
            candidates = round_kwargs["candidate_features_list"]
            if not candidates:
                break
            ranked = self._fit_once(
                working, label, task=task, feature_boosting=feature_boosting,
                stage2_metric=stage2_metric, baseline_cv=baseline_cv,
                max_candidates=max_candidates, **round_kwargs)
            chosen = ranked if n_new_features is None else ranked[:n_new_features]
            self.order_history_.append({
                "order": order, "candidate_count": len(candidates),
                "positive_survivors": len(ranked),
                "selected_formulas": [tree_to_formula(node) for node in chosen],
                "pruning": deepcopy(self.pruning_history_),
                "scores": [dict(formula=tree_to_formula(node), importance=float(score))
                           for node, score in self.new_features_scores_list],
                "elapsed_seconds": perf_counter() - order_started,
            })
            self.myprint(f"Order {order}: {len(chosen)} features selected in "
                         f"{perf_counter() - order_started:.2f} s")
            selected.extend(chosen)
            if not chosen:
                break
            frontier = chosen
            basis.extend(chosen)
            seen.update(tree_to_formula(node) for node in chosen)
            if order < max_order:
                generated = transform_manifest(raw, self.get_manifest(selected))
                generated.columns = [f"__openfe_generated_{i}" for i in range(len(selected))]
                if set(generated.columns) & set(raw.columns):
                    raise ValueError("Source names collide with reserved generated-feature aliases")
                working = pd.concat([raw, generated], axis=1)
        self.ranked_features_ = selected
        self.new_features_list = selected
        self.feature_names_in_ = list(raw.columns)
        self.run_metadata_ = {
            "max_candidates": max_candidates, "operators": list(operators),
            "max_order": max_order, "n_new_features": n_new_features,
            "seed": kwargs.get("seed", 42), "feature_boosting": feature_boosting,
            "stage2_metric": stage2_metric, "evaluator": type(self.evaluator).__name__,
            "evaluator_params": deepcopy(getattr(self.evaluator, "params", {})),
            "stage1_params": deepcopy(getattr(self.evaluator, "stage1_params", {})),
            "stage2_params": deepcopy(getattr(self.evaluator, "stage2_params", {})),
            "task": task, "metric": kwargs.get("metric"), "source_features": list(raw.columns),
            "n_data_blocks": kwargs.get("n_data_blocks", 4),
            "min_candidate_features": kwargs.get("min_candidate_features", 1),
            "elapsed_seconds": perf_counter() - started, "orders": self.order_history_,
        }
        return selected

    def _fit_once(self, data, label, *, task="classification", feature_boosting=True,
            stage2_metric="shap", baseline_cv=None, max_candidates=None, **kwargs):
        self.evaluator.check_available()
        if any(name in kwargs for name in ("model_params", "stage1_params", "stage2_params")):
            raise TypeError("Configure model/stage parameters on the evaluator, not OpenFE.fit")
        if not isinstance(data, pd.DataFrame) or not data.index.is_unique or not data.columns.is_unique:
            raise ValueError("data must have unique row IDs and feature names")
        if not all(isinstance(name,str) for name in data.columns):
            raise ValueError("Feature IDs must be strings")
        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in data.dtypes):
            raise ValueError("The shared engine supports numeric source features only")
        if task not in ("classification", "regression"):
            raise ValueError("Set task explicitly to classification or regression")
        if kwargs.get("stage1_metric", "predictive") != "predictive":
            raise ValueError("This backend evaluates stage-1 predictive improvement only")
        if kwargs.get("is_stage1",True) is False:
            raise ValueError("The shared engine requires stage-1 predictive pruning")
        if kwargs.get("drop_columns") is not None:
            raise ValueError("drop_columns is not supported by this backend")
        if kwargs.get("categorical_features") is not None and len(kwargs["categorical_features"]):
            raise ValueError("The shared engine supports pointwise numerical operators only")
        if stage2_metric not in self._stage2_metrics:
            raise ValueError(f"stage2_metric must be one of {self._stage2_metrics}")
        if kwargs.get("n_jobs", 1) < 1:
            raise ValueError("n_jobs must be positive")
        self.baseline_cv_ = None if baseline_cv is None else list(baseline_cv)
        self.feature_names_in_ = list(data.columns)
        if isinstance(label, (pd.DataFrame, pd.Series)) and not label.index.equals(data.index):
            raise ValueError("Label index must match data index in the same order")
        y = pd.DataFrame(np.asarray(label).reshape(-1, 1), index=data.index)
        if len(y) != len(data) or y.isna().any().any():
            raise ValueError("Labels must be complete and match row count")
        if task == "classification" and y.nunique().iloc[0] < 2:
            raise ValueError("Classification requires at least two classes")
        metric=kwargs.get("metric")
        allowed=({"multi_logloss"} if y.nunique().iloc[0]>2 else {"binary_logloss","auc"}) if task=="classification" else {"rmse"}
        if metric is not None and metric not in allowed:
            raise ValueError(f"Metric for this task must be one of {sorted(allowed)}")
        kwargs.setdefault("seed", 42)
        if kwargs.get("candidate_features_list") is None:
            kwargs["candidate_features_list"] = numeric_candidates(
                data.columns, max_candidates=max_candidates, seed=kwargs["seed"])
        for node in kwargs["candidate_features_list"]:
            _validate_formula(node)
            if not set(node.get_fnode()) <= set(data.columns):
                raise ValueError("Candidate refers to an unknown source feature")
        if not kwargs["candidate_features_list"]:
            raise ValueError("At least one candidate is required")
        kwargs.setdefault("n_data_blocks", 4)
        kwargs.setdefault("min_candidate_features", 1)
        minimum = kwargs["min_candidate_features"]
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
            raise ValueError("min_candidate_features must be a positive integer")
        blocks = kwargs["n_data_blocks"]
        if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks < 1 or blocks & (blocks - 1):
            raise ValueError("n_data_blocks must be a positive power of two")
        with TemporaryDirectory(prefix="openfe-flex-") as directory:
            kwargs["tmp_save_path"] = str(Path(directory)/"train.feather")
            result = super().fit(
                data.copy().replace([np.inf, -np.inf], np.nan), y.copy(),
                task=task, feature_boosting=feature_boosting,
                stage2_metric=stage2_metric, **kwargs)
        self.ranked_features_ = result
        return result

    def stage1_select(self, ratio=0.5):
        """Upstream halving schedule with strict positive final improvement.

        D.3 uses no halving for one data block. Score-based deduplication follows
        upstream. A positive min_candidate_features permits its documented early
        jump to full data. Unlike upstream, no top-100 fallback is applied.
        """
        train_samples = _subsample(self.train_index, self.n_data_blocks)
        val_samples = _subsample(self.val_index, self.n_data_blocks)
        if not train_samples[0] or not val_samples[0]:
            raise ValueError("n_data_blocks exceeds internal partition size")
        self.pruning_history_ = []
        candidates = list(self.candidate_features_list)
        index = 0
        while candidates:
            train, val = train_samples[index], val_samples[index]
            # sklearn model wrappers must see the global class encoding. Tiny
            # random blocks may omit classes: enlarge the block before fitting.
            if self.task == "classification" and (
                    self.label.loc[train].iloc[:, 0].nunique() != self.label.iloc[:, 0].nunique()
                    or (self.metric == "auc" and self.label.loc[val].iloc[:, 0].nunique() < 2)):
                index += 1
                continue
            scores = sorted(self._calculate_and_evaluate(candidates, train, val),
                            key=lambda item: item[1], reverse=True)
            if not all(np.isfinite(score) for _, score in scores):
                raise ValueError("Stage-1 scores must be finite")
            if not self.pruning_history_:
                scores = self.delete_same(scores)
            self.pruning_history_.append({
                "train_rows": len(train), "validation_rows": len(val),
                "evaluated": len(candidates),
                "scores": [dict(formula=tree_to_formula(node), improvement=float(score))
                           for node, score in scores],
            })
            if index == len(train_samples) - 1:
                return [node for node, score in scores if score > 0]
            count = max(int(len(scores) * ratio), min(len(scores), self.min_candidate_features))
            candidates = [node for node, _ in scores[:count]]
            index += 1
            if count <= self.min_candidate_features:
                index = len(train_samples) - 1
        return []

    def get_index(self, train_index, val_index):
        train, val = super().get_index(train_index, val_index)
        train, val = pd.Index(train), pd.Index(val)
        if (not len(train) or not len(val) or not train.is_unique or not val.is_unique
                or len(train.intersection(val)) or not train.isin(self.data.index).all()
                or not val.isin(self.data.index).all()):
            raise ValueError("Internal Train/Validation IDs must be nonempty, unique and disjoint")
        if self.task == "classification":
            classes = set(self.label.iloc[:, 0])
            if set(self.label.loc[train].iloc[:, 0]) != classes or set(self.label.loc[val].iloc[:, 0]) != classes:
                raise ValueError("Both internal partitions must contain every class")
        return train, val

    def _context(self):
        return EvaluationContext(self.task, self.metric,
                                 int(self.label.iloc[:, 0].nunique()) if self.task == "classification" else 0,
                                 self.seed)

    def get_init_score(self, init_scores, use_train=False):
        if use_train:
            raise ValueError("Only out-of-fold baseline predictions are supported")
        y = self.label.iloc[:, 0].to_numpy()
        classes = np.unique(y)
        dimensions = len(classes) if self.task == "classification" and len(classes)>2 else 1
        if init_scores is not None:
            values = np.asarray(init_scores, dtype=float).reshape(len(y), -1)
            if isinstance(init_scores, pd.DataFrame) and not init_scores.index.equals(self.data.index):
                raise ValueError("init_scores index/order must match data")
            if values.shape != (len(y), dimensions) or not np.isfinite(values).all():
                raise ValueError("init_scores must be finite raw scores with one column per class (one for binary)")
            return pd.DataFrame(values, index=self.data.index)
        if not self.feature_boosting:
            if self.task == "regression":
                baseline = [np.mean(y)]
            elif dimensions == 1:
                p = np.clip(np.mean(y), 1e-12, 1-1e-12)
                baseline = [np.log(p/(1-p))]
            else:
                baseline = np.log(np.bincount(y.astype(int))/len(y))
            return pd.DataFrame(np.tile(baseline, (len(y), 1)), index=self.data.index)
        if self.baseline_cv_ is None:
            folds = min(5, int(pd.Series(y).value_counts().min())) if self.task=="classification" else min(5,len(y))
            if folds < 2:
                raise ValueError("Insufficient data per class for an OOF baseline")
            splitter = (StratifiedKFold if self.task=="classification" else KFold)(
                n_splits=folds, shuffle=True, random_state=self.seed)
            splits = list(splitter.split(self.data, y))
        else:
            splits = list(self.baseline_cv_)
        values = np.zeros((len(y), dimensions))
        visits = np.zeros(len(y), dtype=int)
        for fold_index, (train, val) in enumerate(splits, 1):
            self.myprint(f"FeatureBoost baseline fold {fold_index}/{len(splits)}")
            train, val = np.asarray(train, dtype=int), np.asarray(val, dtype=int)
            if (not len(train) or not len(val) or len(np.unique(train))!=len(train)
                    or len(np.unique(val))!=len(val) or np.intersect1d(train,val).size
                    or min(train.min(),val.min())<0 or max(train.max(),val.max())>=len(y)):
                raise ValueError("Invalid/disjointness-violating positional OOF folds")
            if self.task=="classification" and not np.array_equal(np.unique(y[train]),classes):
                raise ValueError("Each OOF training fold must contain every class")
            model = self.evaluator.fit(
                self.data.iloc[train], y[train], baseline=None,
                validation=(self.data.iloc[val], y[val], None),
                context=self._context(), stage="baseline", early_stopping_rounds=50)
            values[val] = np.asarray(self.evaluator.predict_raw(model, self.data.iloc[val])).reshape(len(val), dimensions)
            visits[val] += 1
        if not np.all(visits==1):
            raise ValueError("OOF validation folds must cover every row exactly once")
        return pd.DataFrame(values, index=self.data.index)

    def get_init_metric(self, pred, label):
        if self.metric=="rmse":
            return float(np.sqrt(mean_squared_error(label,np.asarray(pred).ravel())))
        return super().get_init_metric(pred, label)

    def _loss(self, raw, y):
        raw = np.asarray(raw)
        if self.metric=="multi_logloss":
            return log_loss(y, softmax(raw, axis=1), labels=np.arange(raw.shape[1]))
        if self.metric in ("binary_logloss", "auc"):
            from scipy.special import expit
            prob = expit(raw.ravel())
            return roc_auc_score(y, prob) if self.metric=="auc" else log_loss(y,prob,labels=[0,1])
        return float(np.sqrt(mean_squared_error(y,raw.ravel())))

    def _evaluate(self, candidate_feature, train_y, val_y, train_init, val_init, init_metric):
        train_x = pd.DataFrame(candidate_feature.data.loc[train_y.index]).replace([np.inf, -np.inf], np.nan)
        val_x = pd.DataFrame(candidate_feature.data.loc[val_y.index]).replace([np.inf, -np.inf], np.nan)
        if train_x.iloc[:,0].nunique(dropna=False)<=1:
            return 0.0
        model = self.evaluator.fit(
            train_x, train_y, baseline=train_init, validation=(val_x, val_y, val_init),
            context=self._context(), stage="stage1", early_stopping_rounds=3)
        raw = self.evaluator.predict_raw(model, val_x, baseline=val_init)
        after = self._loss(raw,np.asarray(val_y).ravel())
        return float(after-init_metric if self.metric=="auc" else init_metric-after)

    def _calculate(self,candidate_features,train_idx,val_idx):
        if self.n_jobs!=1:
            return super()._calculate(candidate_features,train_idx,val_idx)
        data=self.data.loc[list(train_idx)+list(val_idx)]
        for node in candidate_features:
            with np.errstate(divide="ignore",invalid="ignore",over="ignore"):
                node.calculate(data,is_root=True)
            node.f_delete()
        return candidate_features

    def _calculate_and_evaluate(self,candidate_features,train_idx,val_idx):
        if self.n_jobs!=1:
            return super()._calculate_and_evaluate(candidate_features,train_idx,val_idx)
        data=self.data.loc[list(train_idx)+list(val_idx)]
        train_y,val_y=self.label.loc[train_idx],self.label.loc[val_idx]
        train_init,val_init=self.init_scores.loc[train_idx],self.init_scores.loc[val_idx]
        initial=self.get_init_metric(val_init,val_y)
        results=[]
        for node in tqdm(candidate_features,disable=not self.verbose,desc=f"{type(self.evaluator).__name__} candidates"):
            try:
                with np.errstate(divide="ignore",invalid="ignore",over="ignore"):
                    node.calculate(data,is_root=True)
                score=self._evaluate(node,train_y,val_y,train_init,val_init,initial)
                results.append((node,score))
            except Exception as exc:
                raise RuntimeError(f"Candidate failed: {tree_to_formula(node)}") from exc
            finally:
                node.delete()
        return results

    def stage2_select(self):
        if not self.candidate_features_list:
            self.stage2_model_ = None
            self.stage2_columns_ = []
            self.stage2_importances_ = np.empty(0)
            return []
        nodes = self._calculate(self.candidate_features_list,
                                self.train_index.to_list(),self.val_index.to_list())
        names = [f"autoFE_{i:04d}" for i in range(len(nodes))]
        formulas = [tree_to_formula(node) for node in nodes]
        generated = pd.DataFrame({name:node.data for name,node in zip(names,nodes)},index=nodes[0].data.index)
        # Collision-free internal names; source names are restored in the manifest.
        base = self.data.copy()
        base.columns = [f"base_{i:04d}" for i in range(base.shape[1])]
        combined = pd.concat([generated,base],axis=1)
        if self.drop_columns:
            raise ValueError("drop_columns is not supported by the numeric engine")
        model = self.evaluator.fit(
            combined.loc[self.train_index], self.label.loc[self.train_index],
            baseline=self.init_scores.loc[self.train_index],
            validation=(combined.loc[self.val_index], self.label.loc[self.val_index],
                        self.init_scores.loc[self.val_index]),
            context=self._context(), stage="stage2", early_stopping_rounds=50)
        importance = np.asarray(self.evaluator.importance(
            model, combined.loc[self.val_index], method=self.stage2_metric, context=self._context()))
        if importance.shape != (combined.shape[1],) or not np.isfinite(importance).all():
            raise ValueError("Evaluator importance must be finite with one score per input column")
        self.stage2_model_ = model
        self.stage2_columns_ = list(combined.columns)
        self.stage2_importances_ = np.asarray(importance)
        for node in nodes:
            node.delete()
        scores = [(formula_to_tree(formula),float(score))
                  for formula,score in zip(formulas,importance[:len(formulas)])]
        return sorted(scores,key=lambda pair:pair[1],reverse=True)

    def get_manifest(self, features=None):
        features = self.ranked_features_ if features is None else features
        return [dict(name=f"autoFE_{i:04d}",formula=tree_to_formula(node),
                     source_features=sorted(node.get_fnode())) for i,node in enumerate(features)]

    def save_manifest(self, path, features=None):
        Path(path).write_text(json.dumps(self.get_manifest(features),indent=2)+"\n",encoding="utf-8")

    def save_run(self, path):
        """Export the search configuration, per-round audit and elapsed time."""
        Path(path).write_text(json.dumps(self.run_metadata_, indent=2) + "\n", encoding="utf-8")

    def transform_new(self, data, features=None):
        return transform_manifest(data,self.get_manifest(features))

    def transform(self,X_train,X_test,new_features_list,n_jobs=1,name=""):
        """Upstream-compatible augmented output, computed independently per batch.

        Use transform_new for the generated-only representation. This method
        never reuses the temporary training file removed when fit completed.
        """
        if n_jobs<1:
            raise ValueError("n_jobs must be positive")
        output=[]
        for data in (X_train,X_test):
            generated=self.transform_new(data,new_features_list)
            generated.columns=[f"autoFE_f_{i}{name}" for i in range(generated.shape[1])]
            if set(generated.columns)&set(data.columns):
                raise ValueError("Generated feature names collide with source columns")
            output.append(pd.concat([data,generated],axis=1))
        return tuple(output)
