"""Optional entry point for the pinned original LightGBM evaluation backend."""
from importlib.util import find_spec

from ._upstream.openfe import OpenFE


class LightGBMOpenFE(OpenFE):
    """Original evaluator; install ``openfe-flex[lightgbm]`` to use it.

    Unlike CatBoostOpenFE, this keeps upstream operators, scoring defaults,
    temporary-file behavior, and transform semantics. It is not a drop-in
    implementation of the generated-only CatBoost API.
    """

    def fit(self, *args, **kwargs):
        if find_spec("lightgbm") is None:
            raise ImportError(
                "LightGBMOpenFE requires the optional backend: "
                "pip install 'openfe-flex[lightgbm]' (or pip install -e '.[lightgbm]' from source)"
            )
        return super().fit(*args, **kwargs)

