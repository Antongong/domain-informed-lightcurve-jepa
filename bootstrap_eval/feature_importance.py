#!/usr/bin/env python3
"""
Extracts scikit-learn's built-in global feature importance (.feature_importances_)
from Random Forest, once per training truncation level (100/50/25).

Feature importance depends only on the TRAINING data -- it's a byproduct of how
the trees were built during .fit(), never touching the test set -- so this only
needs 3 refits (one per train_pct), not 9 (one per combo). Confirmed empirically
that best_params.json is identical across all 3 test_pct siblings for a given
train_pct, since the hyperparameter search itself never touches the test set
either (PredefinedSplit only uses train+validation).

Reuses get_predictions.py's load_combo_data() rather than duplicating it --
does NOT modify or rerun get_predictions.py, and does not touch
bootstrap_metrics.py or bootstrap_metrics_per_class.py at all.

Output: feature_importance.csv, one row per (train_pct, feature_name),
sorted by importance within each train_pct.
"""
import os
import glob
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import manifest as M
from get_predictions import load_combo_data
from feature_categories import category_of

RF_SEEDS = [42, 100, 158 + 42]  # matches get_predictions.py / rf.py convention -> [42, 100, 200]


def get_best_params(train_pct):
    """For just the train levels, not both, find the saved best parameters"""
    for test_pct in M.TEST_LEVELS:
        out_base = M.COMBO_OUT_BASE.get((train_pct, test_pct))
        if out_base is None:
            continue
        matches = glob.glob(os.path.join(out_base, "rf", "*", "best_params.json"))
        if matches:
            with open(matches[0]) as f:
                return json.load(f)["best_params"]
    raise FileNotFoundError(f"No best_params.json found for train_pct={train_pct}")


def get_feature_names(train_pct):
    """load all of the features"""
    train_dir = M.PCT_TO_FEATURES_DIR[train_pct]
    path = os.path.join(train_dir, "handcrafted_features_train.npz")
    with np.load(path, allow_pickle=True) as z:
        return list(z["feature_names"])


def main():
    """Load all of the data, get feature names and best parameters,
    run the rf but specifically with **best_params activated, save and store the
    data of the best parameters,"""
    rows = []
    for train_pct in M.TRAIN_LEVELS:
        print(f"\n=== train{train_pct} ===")
        # test_pct passed here is irrelevant to the fit -- only X_train/y_train get used
        X_train_full, y_train_full, _X_test, _y_test, _labels = load_combo_data(train_pct, train_pct)

        feature_names = get_feature_names(train_pct)
        best_params = get_best_params(train_pct)
        print(f"  best_params: {best_params}")

        importances_per_seed = []
        for seed in RF_SEEDS:
            clf = RandomForestClassifier(random_state=seed, n_jobs=-1, **best_params)
            clf.fit(X_train_full, y_train_full)
            importances_per_seed.append(clf.feature_importances_)

        importance_mean = np.mean(importances_per_seed, axis=0)

        for fname, imp in zip(feature_names, importance_mean):
            rows.append({"train_pct": train_pct, "feature_name": fname, "importance": imp})

        top5 = sorted(zip(feature_names, importance_mean), key=lambda x: -x[1])[:5]
        print("  top 5 features:", [f"{n}[{category_of(n)}]={v:.4f}" for n, v in top5])

    df = pd.DataFrame(rows)
    df = df.sort_values(["train_pct", "importance"], ascending=[True, False])
    out_path = os.path.join(M.THIS_DIR, "feature_importance.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
