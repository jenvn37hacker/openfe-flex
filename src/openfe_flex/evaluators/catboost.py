"""CatBoost evaluator, imported without importing CatBoost itself."""
from copy import deepcopy
from importlib.util import find_spec

import numpy as np

from .base import numeric_data


class CatBoostEvaluator:
    supported_importances = ("shap", "prediction_values_change")

    def __init__(self, params=None, *, stage1_params=None, stage2_params=None):
        self.params = deepcopy(dict(params or {}))
        self.stage1_params = deepcopy(dict(stage1_params or {}))
        self.stage2_params = deepcopy(dict(stage2_params or {}))

    def check_available(self):
        if find_spec("catboost") is None:
            raise ImportError("Install the CatBoost backend: pip install 'openfe-flex[catboost]'")

    @staticmethod
    def pool(data, labels=None, baseline=None):
        from catboost import Pool

        return Pool(numeric_data(data), label=None if labels is None else np.asarray(labels).ravel(),
                    baseline=None if baseline is None else np.asarray(baseline, dtype=float))

    def fit(self, data, labels, *, validation, baseline, context, stage, early_stopping_rounds):
        from catboost import CatBoostClassifier, CatBoostRegressor

        objective = ("RMSE" if context.task == "regression" else
                     "MultiClass" if context.n_classes > 2 else "Logloss")
        params = dict(iterations=1000, depth=5, learning_rate=.03, l2_leaf_reg=5,
                      random_seed=context.seed, thread_count=1, verbose=False,
                      allow_writing_files=False)
        params.update(deepcopy(self.params))
        if stage == "stage1":
            params.update(iterations=100)
            params.update(deepcopy(self.stage1_params))
        elif stage == "stage2":
            params.update(deepcopy(self.stage2_params))
        params.update(loss_function=objective, eval_metric={
            "rmse":"RMSE", "multi_logloss":"MultiClass",
            "binary_logloss":"Logloss", "auc":"AUC",
        }[context.metric])
        if objective == "MultiClass":
            params["classes_count"] = context.n_classes
        model = (CatBoostRegressor if context.task == "regression" else CatBoostClassifier)(**params)
        val_x, val_y, val_baseline = validation
        model.fit(self.pool(data, labels, baseline),
                  eval_set=self.pool(val_x, val_y, val_baseline),
                  early_stopping_rounds=early_stopping_rounds, use_best_model=True)
        return model

    def predict_raw(self, model, data, *, baseline=None):
        # CatBoost includes Pool.baseline in RawFormulaVal already.
        return np.asarray(model.predict(self.pool(data, baseline=baseline), prediction_type="RawFormulaVal"))

    def importance(self, model, data, *, method, context):
        if method == "shap":
            values = np.asarray(model.get_feature_importance(self.pool(data), type="ShapValues", thread_count=1))
            return np.abs(values[..., :-1]).mean(axis=tuple(range(values.ndim - 1)))
        if method == "prediction_values_change":
            return np.asarray(model.get_feature_importance(type="PredictionValuesChange", thread_count=1))
        raise ValueError(f"Unsupported CatBoost importance: {method}")

