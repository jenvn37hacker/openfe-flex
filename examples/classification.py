"""Run a small generated-only example: python examples/classification.py.

The held-out test split is never passed to OpenFE.fit. Replace the synthetic
tables with numeric feature tables for your application; use recording/group
disjoint outer and inner partitions when multiple rows share one recording.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from openfe_flex import OpenFE, CatBoostEvaluator, LightGBMEvaluator, transform_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("catboost", "lightgbm"), default="catboost")
    parser.add_argument("--classes", type=int, default=3)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--keep", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output", type=Path, help="Optional JSON formula manifest")
    args = parser.parse_args()
    if args.classes < 2 or min(args.candidates, args.keep, args.jobs) < 1:
        parser.error("Require classes >= 2 and positive candidates, keep and jobs")

    values, labels = make_classification(
        n_samples=max(240, args.classes * 20), n_features=10, n_informative=8,
        n_redundant=0, n_classes=args.classes, n_clusters_per_class=1, random_state=42,
    )
    x = pd.DataFrame(values, columns=[f"f_{i}" for i in range(values.shape[1])])
    y = pd.Series(labels, index=x.index)
    outer_train, test = train_test_split(x.index, test_size=.2, stratify=y, random_state=42)
    train_x, train_y = x.loc[outer_train], y.loc[outer_train]
    inner_train, inner_val = train_test_split(
        train_x.index, test_size=.25, stratify=train_y, random_state=42,
    )

    if args.backend == "catboost":
        evaluator = CatBoostEvaluator(
            params={"iterations":30, "depth":3, "thread_count":1},
            stage1_params={"iterations":15})
    else:
        evaluator = LightGBMEvaluator(
            params={"n_estimators":30, "num_leaves":7, "n_jobs":1},
            stage1_params={"n_estimators":15})
    generator = OpenFE(evaluator=evaluator)
    ranked = generator.fit(
        train_x, train_y, task="classification",
        train_index=inner_train, val_index=inner_val,
        max_candidates=args.candidates,
        n_new_features=args.keep,
        min_candidate_features=4, n_data_blocks=1, n_jobs=args.jobs, seed=42,
        feature_boosting=True, stage2_metric="shap", verbose=True,
    )
    chosen = ranked[:args.keep]
    manifest = generator.get_manifest(chosen)
    train_generated = generator.transform_new(train_x, chosen)
    test_generated = transform_manifest(x.loc[test], manifest)
    if not chosen:
        print(json.dumps({"backend":args.backend,"n_generated":0,
                          "status":"No candidate improved validation loss; no generated-only model fitted."}))
        if args.output is not None:
            generator.save_manifest(args.output, chosen)
        return
    if args.backend == "catboost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(
            iterations=60, depth=3, random_seed=42, thread_count=1,
            verbose=False, allow_writing_files=False)
    else:
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(n_estimators=60, num_leaves=7, random_state=42,
                               n_jobs=1, verbosity=-1, min_child_samples=2)
    # Fixed training budget; no early stopping or tuning on the held-out test.
    model.fit(train_generated, train_y)
    predictions = np.asarray(model.predict(test_generated)).ravel()
    print(json.dumps({
        "backend": args.backend,
        "representation": "generated features only",
        "n_generated": len(chosen),
        "test_accuracy": accuracy_score(y.loc[test], predictions),
        "test_f1_macro": f1_score(y.loc[test], predictions, average="macro"),
        "formulas": manifest,
    }, indent=2))
    if args.output is not None:
        generator.save_manifest(args.output, chosen)


if __name__ == "__main__":
    main()

