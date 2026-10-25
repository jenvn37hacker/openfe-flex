"""LightGBM evaluator with explicit raw-margin FeatureBoost correction."""
from copy import deepcopy
from importlib.util import find_spec

import numpy as np

from .base import numeric_data, with_baseline


class LightGBMEvaluator:
    supported_importances = ("shap", "gain_importance")

    def __init__(self, params=None, *, stage1_params=None, stage2_params=None):
        self.params = deepcopy(dict(params or {}))
        self.stage1_params = deepcopy(dict(stage1_params or {}))
        self.stage2_params = deepcopy(dict(stage2_params or {}))

    def check_available(self):
        if find_spec("lightgbm") is None:
            raise ImportError("Install the LightGBM backend: pip install 'openfe-flex[lightgbm]'")

    @classmethod
    def paper(cls, *, params=None):
        """D.3 iteration/rate/leaves settings for numerical reference experiments.

        Set stage2_metric='gain_importance' and the experiment's data-block count
        on OpenFE.fit. This does not enable unsupported global/categorical operators.
        """
        common = dict(n_estimators=1000, learning_rate=.1, num_leaves=16, min_child_samples=20)
        common.update(dict(params or {}))
        return cls(params=common, stage1_params={"n_estimators":common["n_estimators"]})

    def fit(self, data, labels, *, validation, baseline, context, stage, early_stopping_rounds):
        import lightgbm as lgb

        params = dict(n_estimators=1000, num_leaves=16, learning_rate=.03,
                      random_state=context.seed, n_jobs=1, verbosity=-1,
                      min_child_samples=2, deterministic=True, force_col_wise=True)
        params.update(deepcopy(self.params))
        if stage == "stage1":
            params.update(n_estimators=100)
            params.update(deepcopy(self.stage1_params))
        elif stage == "stage2":
            params.update(deepcopy(self.stage2_params))
        objective = ("regression" if context.task == "regression" else
                     "multiclass" if context.n_classes > 2 else "binary")
        params.update(objective=objective, metric=context.metric)
        if objective == "multiclass":
            params["num_class"] = context.n_classes
        model = (lgb.LGBMRegressor if context.task == "regression" else lgb.LGBMClassifier)(**params)
        val_x, val_y, val_baseline = validation
        def margins(value):
            if value is None:
                return None
            array = np.asarray(value)
            return array if objective == "multiclass" else array.ravel()
        model.fit(numeric_data(data), np.asarray(labels).ravel(), init_score=margins(baseline),
                  eval_set=[(numeric_data(val_x), np.asarray(val_y).ravel())],
                  eval_init_score=[margins(val_baseline)],
                  callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
        return model

    def predict_raw(self, model, data, *, baseline=None):
        # LightGBM predictions exclude fit-time init_score; add it exactly once.
        raw = model.predict(numeric_data(data), raw_score=True)
        return with_baseline(raw, baseline)

    def importance(self, model, data, *, method, context):
        if method == "shap":
            values = np.asarray(model.booster_.predict(
                numeric_data(data), pred_contrib=True, num_iteration=model.best_iteration_))
            values = values.reshape(len(data), -1, data.shape[1] + 1)
            return np.abs(values[..., :-1]).mean(axis=(0, 1))
        if method == "gain_importance":
            return np.asarray(model.booster_.feature_importance(importance_type="gain"))
        raise ValueError(f"Unsupported LightGBM importance: {method}")

