# OpenFE-Flex

**Automated Feature Engineering with Pluggable Model Evaluators**

A numeric adaptation of OpenFE with a shared search engine and an **injected
model evaluator**. Choose CatBoost or LightGBM without changing the candidate
search, validation loss calculation, formula manifest or transform API.

```python
from openfe_flex import OpenFE, CatBoostEvaluator

generator = OpenFE(
    evaluator=CatBoostEvaluator(params={"iterations": 1000, "depth": 5}),
)
```

This is an independent project alongside `recursive-selection` and `shm_ml`.
It adapts OpenFE 0.0.12 at commit
`3e3bdf9f5b972e214e018c22f6f9f456a47b416e`; it is not an official OpenFE backend
or a reproduction of the paper's original configuration. See `LICENSE` and
`UPSTREAM.json` for the original MIT notice, source checksums and engine changes.

## Structure

```text
openfe-flex/
├── pyproject.toml
├── src/openfe_flex/
│   ├── engine.py
│   ├── features.py
│   ├── evaluators/
│   │   ├── base.py
│   │   ├── catboost.py
│   │   └── lightgbm.py
│   └── _upstream/
├── tests/
├── examples/classification.py
└── .github/workflows/tests.yml
```

The distribution is `openfe-flex`; the Python import is `openfe_flex`.
This project builds a pure Python wheel. Model libraries are external optional
dependencies, imported when used. A tag-triggered GitHub Release workflow is included; PyPI publication is not configured.

Version 0.3.0 renames the former `openfe-catboost` distribution and
`openfe_catboost` import. Uninstall that old distribution, install `openfe-flex`,
and update imports and editable source paths. Formula JSON manifests are unchanged.

## Install from source

Use CPython 3.11, 3.12 or 3.13. From this project's root on Linux/macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[catboost,dev]"
.venv/bin/python examples/classification.py --backend catboost
```

Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[catboost,dev]"
.venv\Scripts\python.exe examples/classification.py --backend catboost
```

Choose extras for the evaluator you use:

| Extra | Model dependencies |
| --- | --- |
| `[catboost]` | CatBoost |
| `[lightgbm]` | LightGBM |
| `[all]` | Both |
| `[dev]` | Build tool; tests use standard `unittest` |

For example, `python -m pip install -e ".[lightgbm]"` supports LightGBM without
installing CatBoost. Installing the core package alone supports formula transforms
and custom evaluators without either model library. A missing backend produces
an installation hint when `fit` is called.

From sibling `shm_ml`, after `./envstart`, use its kernel environment:
`.venv/bin/python -m pip install -r requirements-openfe.txt`. That file installs
`../openfe-flex[catboost]`. Copy/clone both directories together when migrating,
or install this package's wheel with the appropriate backend extra.

## Choose an evaluator

```python
from openfe_flex import OpenFE, CatBoostEvaluator, LightGBMEvaluator

catboost_generator = OpenFE(
    evaluator=CatBoostEvaluator(
        params={"iterations": 1000, "depth": 5, "thread_count": 1},
        stage1_params={"iterations": 100},
        stage2_params={"iterations": 500},
    ),
)

lightgbm_generator = OpenFE(
    evaluator=LightGBMEvaluator(
        params={"n_estimators": 1000, "num_leaves": 16, "n_jobs": 1},
        stage1_params={"n_estimators": 100},
        stage2_params={"n_estimators": 500},
    ),
)
```

Model parameters belong to the evaluator; search parameters belong to `OpenFE.fit`.
Each evaluator copies its configuration and creates a fresh model for each fit.
Stage-specific settings override common parameters. The default stage-1 iteration
budget is 100; override it explicitly with `stage1_params`. The engine supplies
the task, objective/metric convention, class count and default seed.

Both evaluators support `stage2_metric="shap"` (the shared default).
CatBoost additionally supports `"prediction_values_change"`; LightGBM additionally
supports `"gain_importance"`. Unsupported combinations are rejected. Mean absolute
SHAP averages over samples and classes and excludes the bias column. Keeping the
importance method the same does not make the fitted models or results identical.

An evaluator controls feature discovery. The classifier trained afterwards on
those features is an independent choice.

## Fit and export generated features

Fit using **outer Train only**. Internal Train/Validation IDs must be disjoint
and contain every class. If omitted, the engine uses an internal stratified split
for classification. Labels must share the feature table's index and row order.

```python
from sklearn.model_selection import train_test_split
from openfe_flex import OpenFE, CatBoostEvaluator, transform_manifest

# All rows here belong to outer Train. Use recording-disjoint partitions
# instead of this sample split when multiple rows share a recording.
inner_train, inner_val = train_test_split(
    train_x.index, test_size=0.2, stratify=train_y, random_state=42,
)
generator = OpenFE(evaluator=CatBoostEvaluator(
    params={"iterations": 1000, "depth": 5, "thread_count": 1},
    stage1_params={"iterations": 100},
))
ranked = generator.fit(
    train_x, train_y, task="classification", metric="multi_logloss",
    train_index=inner_train, val_index=inner_val,
    feature_boosting=True, stage2_metric="shap",
    max_candidates=2000, n_new_features=50, n_jobs=1, seed=42,
)
chosen = ranked[:50]
manifest = generator.get_manifest(chosen)
x_train_generated = generator.transform_new(train_x, chosen)
x_val_generated = transform_manifest(val_x, manifest)
x_test_generated = transform_manifest(test_x, manifest)
generator.save_manifest("formulas.json", chosen)
```

`fit` returns formula nodes, not the estimator. `n_new_features` selects at most
that many surviving features per order (default 10); use None for all survivors
in a single order. Zero survivors is valid and means no candidate improved
validation loss; skip any generated-only model fit in that case. `transform_new`
returns generated columns only. `transform(train, test, features)` is the augmented
output API; each batch is still computed independently.

For correlated recordings, also pass
`baseline_cv=list(grouped_splitter.split(train_x, train_y, groups=recording_ids))`.
These OOF folds use positional indices, must be disjoint within each fold, and
must validate every row exactly once. No recording identities are inferred.
Use the same grouping principle for the internal Train/Validation split.

The frozen manifest can transform QUGS Data B without fitting. It records source
feature dependencies, e.g. `(S2_rms/S7_rms)` requires both source statistics and
both channels even though neither original column is a direct model input.
Nonfinite values become NaN; any further imputation must be fitted on outer Train.

## Algorithm and scope

The shared engine reuses OpenFE's formula AST and successive feature-wise halving:

1. Enumerate all formulas in the supported first-order numeric space by default
   (`max_candidates=None`). A finite budget explicitly enables reproducible
   reservoir sampling. Exhaustive enumeration can be expensive on wide tables.
2. Build OOF raw margins using the injected evaluator (`feature_boosting=True`).
3. Fit small correction models for each candidate. Stage-1 score is baseline
   validation loss minus corrected loss; AUC uses the reversed direction.
4. Fit a correction model on surviving formulas plus the current full basis and
   select the top `n_new_features` by the configured importance method.
5. If `max_order>1`, add accepted features to the basis, recompute OOF margins,
   and expand formulas touching the accepted frontier. Stop when no feature
   improves. Returned formulas retain their original-source dependencies.

`save_run("search.json")` exports per-round scores, counts, selected formulas,
search/evaluator parameters and elapsed time. Scores are ranked within each order;
importance scores from different orders are not globally comparable. `save_manifest`
remains the smaller representation used for transforming new data.

See [algorithm alignment](docs/algorithm_alignment.md) for the pseudocode-to-code
mapping, pruning edge cases and the controlled numerical upstream comparison.
`LightGBMEvaluator.paper()` provides D.3's model settings; pass
`stage2_metric="gain_importance"` and set the experiment's block count on fit.

The evaluator handles model fitting, raw-margin prediction and importance. The
engine owns folds, losses, pruning and manifests. CatBoost uses `Pool.baseline`;
its raw predictions already include that baseline. LightGBM uses `init_score`;
its raw predictions exclude it, so its evaluator adds it exactly once.

Stage-2 SHAP describes the **correction ensemble**. The external OOF baseline's
own contributions are not part of those SHAP values. This ranking is not the SHAP
ranking of the final generated-only classifier or a later Proposed model.

The numeric engine admits pointwise operators only: `abs`, `log`, `sqrt`,
`square`, `sigmoid`, `round`, `residual`, `+`, `-`, `*`, `/`, `min`, `max`.
The default generator uses all of these supported pointwise operators. It includes both directions of subtraction and division. Operator
semantics follow the pinned upstream definitions: for example, `sqrt` takes the
absolute value first and division by zero produces missing values. Global
frequency, grouping and categorical operators are rejected. Feature IDs must be
simple unique strings such as `S1_rms`.

`n_jobs=1` evaluates sequentially. Parallel jobs use `spawn`; custom evaluators
must be pickleable and scripts need an `if __name__ == "__main__":` guard.
Model threading is configured independently on the evaluator.

## Custom evaluators and compatibility

Implement the exported `Evaluator` protocol (`check_available`, `fit`,
`predict_raw`, `importance`, `supported_importances`) and inject that object.
See `src/openfe_flex/evaluators/base.py` for the signatures and
`EvaluationContext`. No backend name registry or engine subclass is required.
Raw predictions must include any supplied baseline exactly once; importance must
return one finite score per input column.

`CatBoostOpenFE()` remains a compatibility facade over this shared engine and
accepts the old `model_params`/`stage1_params`/`stage2_params` fit arguments.
`LightGBMOpenFE()` retains the original upstream evaluator, defaults and transform
semantics for older experiments. New code should use
`OpenFE(evaluator=LightGBMEvaluator(...))` for the shared numeric/manifest API.
The latter is an adaptation, not the stock OpenFE benchmark configuration.

## Verification and build

```bash
.venv/bin/python -m pip install -e ".[all,dev]"
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python examples/classification.py --backend lightgbm --classes 17
.venv/bin/python -m pip check
.venv/bin/python -m build
```

Tests cover custom evaluator injection, optional backend imports, binary/17-class/
31-class tasks, regression, raw baseline correction against native validation
losses, serial/parallel execution, manifests, schema validation and held-out batch
invariance. Tests requiring an absent model library are skipped.

CI builds one universal wheel and source archive, generates `SHA256SUMS`, then
installs **that wheel** (never the editable checkout) on 12 combinations:
CPython 3.11–3.13 on Linux x86-64, Windows x64, macOS Intel and macOS Apple Silicon.
Each job tests core-only, LightGBM-only, then both model extras. macOS jobs install
OpenMP for LightGBM. Remote CI results require running the workflow on GitHub;
local checks cover the available Linux environment.

The distributions are uploaded as an Actions artifact. Pushing a version-matched
tag such as `v0.4.0` publishes a GitHub Release **only after all wheel-test jobs
pass**. The version in `pyproject.toml`, `__version__` and the tag must agree;
release artifacts are verified against their checksums. A failed job prevents
the release. No remote repository or tag is created by this local implementation.

After committing this independent project and adding your GitHub remote:

```bash
git push -u origin HEAD
# Once ready to release the current 0.4.0 version:
git tag v0.4.0
git push origin v0.4.0
```

Build outputs are a `py3-none-any` wheel and a source archive under `dist/`.
The source archive includes tests, examples and alignment documentation. The
wheel's model extras must be selected when installing on another machine.
For an install from source use `pip install ".[catboost]"` or `".[lightgbm]"`.

Synthetic checks demonstrate implementation behavior, not improved Z24/QUGS
accuracy or speed. SHM notebooks' sensor catalogs and `reshape(sensors, 10)` masks
still need dependency-aware integration before using generated features. The
library does not run those experiments. For Proposed selection on generated
features, define the Full reference in that representation and also report
quality relative to the original statistical-feature Full model.
