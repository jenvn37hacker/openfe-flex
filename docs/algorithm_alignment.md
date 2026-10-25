# Alignment with OpenFE

Reference: Zhang et al., *OpenFE: Automated Feature Generation with Expert-level
Performance*, ICML 2023. [Paper](https://proceedings.mlr.press/v202/zhang23ay/zhang23ay.pdf).
Upstream source is pinned in `UPSTREAM.json`. The original MIT notice is retained.

This document describes the **shared numeric engine**, not the legacy
`LightGBMOpenFE` entry point. It does not claim reproduction of the original
benchmark scores, full operator universe, or speedups.

| Reference | Implementation | Scope and differences |
| --- | --- | --- |
| Algorithm 1: expansion and reduction | `engine.OpenFE.fit`, `features.expand_candidates` | First order by default. `max_order>1` expands accepted top-k expressions, adds them to the basis and recomputes OOF margins. |
| Section 3.1: enumerate transformations | `numeric_candidates`, `expand_candidates` | All supported pointwise numeric formulas by default. Both directions of division/subtraction. A finite cap samples explicitly. Frequency/GroupBy/categorical operators are unsupported. |
| Algorithm 2: FeatureBoost | `get_init_score`, `_evaluate`, evaluators | Raw-margin baseline plus a fresh correction model; stage-1 improvement computed independently on validation data. |
| Algorithm 3: coarse pruning | `stage1_select` | Uses upstream's nested random blocks, halving schedule, initial score-based deduplication and optional minimum-candidate early jump. Removes final scores <=0 without upstream's top-100 fallback. |
| Algorithm 4: attribution | `stage2_select` | Correction model fitted on the current full basis plus surviving candidates. Importance ranks only the newly generated candidates. |
| Appendix D.3: model parameters | `LightGBMEvaluator.paper()` | 1,000 estimators, learning rate 0.1, 16 leaves. Stage-1/stage-2 stopping patience remains 3/50. Select gain importance and the appropriate data-block count explicitly. |

## Pruning details

The paper's pseudocode and released implementation are not identical in all
edge cases. Appendix D.3 describes no halving when there is one data block.
OpenFE-Flex follows that behavior and the released code's schedule for larger
power-of-two block counts: evaluate the initial subset, halve before the next
evaluation, and end with all internal training/validation rows. The pseudocode
can be read as halving after the final evaluation too; that additional final
halving is not applied here. This difference is intentional and testable.

Deduplication follows the upstream implementation's initial score-based
comparison (threshold 1e-20); this is not a guarantee of equality of candidate
values. Setting `min_candidate_features>1` can jump to full data earlier and
changes the search budget. The default is 1. Tiny classification blocks that
omit training classes are enlarged before evaluation, because sklearn model
wrappers must retain the global label encoding. AUC evaluation also requires
both validation classes. These enlargements can change the block schedule.

If all candidates have nonpositive improvement, the result is empty, attribution
is skipped, and higher-order expansion stops. The original source's top-100
fallback remains only in the explicitly legacy entry point.

## Higher orders and output

`n_new_features=10` selects at most ten ranked survivors **per order**. Use a
different top-k to match an experiment. `n_new_features=None` returns all ranked
survivors for a single order. With multiple orders, returned features are
concatenated in generation order, ranked within each order. Importance scores
from different fitted correction models are not globally comparable.

Higher-order formulas must touch the newest accepted frontier. Their ASTs refer
to the original source columns; internal basis aliases never enter manifests.
The definition of order is operator-tree depth, not the total operator count.
Unselected expressions are not used as a frontier. This hierarchical search
does not enumerate every mathematically possible high-order expression.

`transform_new` returns generated columns only; `transform` returns original plus
generated columns. The latter representation is closest to the paper's
base-plus-new-feature objective. A generated-only downstream model is a separate
experiment: FeatureBoost measures incremental value conditional on the basis,
not optimality when that basis is removed.

## Numerical reference checks

`tests/test_paper_core.py` compares:

- nested block sizes, retained formulas and halving with the pinned upstream;
- independently recomputed LightGBM improvement against original native scores;
- retained candidate sets and stage-2 gain under matched candidates, splits,
  supplied raw margins, parameters and a serial reference wrapper;
- empty results, hierarchy, dependencies and transform invariance.

The serial reference wrapper removes only multiprocessing; it uses original
scoring/pruning/attribution methods. The comparison uses upstream's actual stage-1
100-tree implementation for numerical parity, not the paper's 1,000-tree D.3
setting. The `paper()` configuration is checked separately. Loss comparisons
allow small numerical differences from LightGBM's internal casts. End-to-end
benchmark parity has not been established by these synthetic tests.

## Example reference configuration

```python
from openfe_flex import OpenFE, LightGBMEvaluator

generator = OpenFE(evaluator=LightGBMEvaluator.paper())
ranked = generator.fit(
    train_x, train_y, task="classification", feature_boosting=True,
    stage2_metric="gain_importance", n_data_blocks=1,
    max_candidates=None, max_order=1, n_new_features=50, seed=42,
)
generator.save_run("openfe_search.json")
```

Set recording-disjoint internal partitions and OOF folds when rows share a
recording. Keep outer validation/test and Data B outside discovery. Neither
the paper settings nor the injectable interface make accuracy improvements
automatic.
