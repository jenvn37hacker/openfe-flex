# Changelog

## 0.4.0

- Enumerate all supported numerical formulas by default; a finite candidate
  budget explicitly enables reproducible reservoir sampling.
- Select top-k features per round and optionally expand higher-order formulas
  from accepted features with updated FeatureBoost baselines.
- Remove nonpositive fallback in the shared engine and support empty results.
- Export per-round pruning scores, configuration and elapsed time with `save_run`.
- Add LightGBM paper-parameter configuration and independent upstream comparisons.
- Build once, test the installed universal wheel across 12 OS/Python combinations,
  verify checksums, and publish GitHub Releases for version-matched tags.

## 0.3.0

- Rename the project and distribution from `openfe-catboost` to `openfe-flex`
  and the public Python package from `openfe_catboost` to `openfe_flex`.
- Update source paths, imports, install hints, examples and CI to the new name.
- Keep the injected evaluator API and formula manifest format unchanged.

## 0.2.0

- Introduce `OpenFE(evaluator=...)` with a shared numeric search engine and an
  exported evaluator protocol for custom model backends.
- Add `CatBoostEvaluator` and `LightGBMEvaluator` for baseline, candidate and
  stage-2 model evaluation; both support native SHAP ranking.
- Make both model libraries optional extras and import them only when used.
- Retain the old CatBoost facade and original LightGBM entry point for compatibility.
- Update examples to select either evaluator with `--backend`.

## 0.1.0

- Standalone `src/` package with the public import `openfe_catboost`.
- CatBoost out-of-fold raw baseline, candidate correction scoring, and native
  SHAP ranking over the pinned OpenFE engine.
- Bounded pointwise numeric candidate generation and generated-only transforms.
- JSON formula manifests with explicit source-feature dependencies.
- Optional original LightGBM evaluation backend.

