"""Compatibility entry point; new code should inject CatBoostEvaluator into OpenFE."""
from .engine import OpenFE
from .evaluators import CatBoostEvaluator
from .features import numeric_candidates, transform_manifest


class CatBoostOpenFE(OpenFE):
    """Compatibility facade over the shared engine, retaining the old fit API."""

    def __init__(self):
        super().__init__(evaluator=CatBoostEvaluator())

    def fit(self, data, label, *, model_params=None, stage1_params=None, stage2_params=None, **kwargs):
        kwargs.setdefault("n_new_features", None)
        self.evaluator = CatBoostEvaluator(
            params=model_params, stage1_params=stage1_params, stage2_params=stage2_params)
        self._stage2_metrics = self.evaluator.supported_importances
        return super().fit(data, label, **kwargs)
