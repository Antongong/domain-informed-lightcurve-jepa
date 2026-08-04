#!/usr/bin/env python3
"""
Per-class extension of the bootstrap pipeline, scoped to Random Forest only
(the classifier singled out for this deeper check).

Reuses the SAME rf prediction files get_predictions.py already saved --
no retraining, no new predictions, no change to Stage 1 at all. For each
combo, resamples the test set 2000 times using the same convention as
bootstrap_metrics.py (one shared resampled index per iteration, scores
averaged across the 3 seeds) -- but instead of collapsing precision/recall/F1
down to one macro number per iteration, keeps all 7 classes' numbers
separately, so each class ends up with its own mean +/- std, per combo.

Output: bootstrap_results_per_class.csv, one row per (train_pct, test_pct, class).
"""
""""""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

import manifest as M

N_BOOT = 2000
SEED = 42


def bootstrap_per_class(y_true, y_preds_by_seed, n_classes, n_boot=N_BOOT, seed=SEED):
    """take different seeds that later will be averaged """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    seed_preds = list(y_preds_by_seed.values())
    n_seeds = len(seed_preds)

    """make empty matricies holding the metrics for each class"""
    precs = np.empty((n_boot, n_classes))
    recs = np.empty((n_boot, n_classes))
    f1s = np.empty((n_boot, n_classes))

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)  # same shared resample for every seed, this iteration
        yt = y_true[idx]

        prec_sum = np.zeros(n_classes)
        rec_sum = np.zeros(n_classes)
        f1_sum = np.zeros(n_classes)
        for y_pred in seed_preds:
            """calculates the metrics for each seed and then takes
            the averave"""
            yp = y_pred[idx]
            p, r, f1, _ = precision_recall_fscore_support(
                yt, yp, labels=list(range(n_classes)), average=None, zero_division=0
            )
            prec_sum += p
            rec_sum += r
            f1_sum += f1

        precs[b] = prec_sum / n_seeds
        recs[b] = rec_sum / n_seeds
        f1s[b] = f1_sum / n_seeds

    return precs, recs, f1s


def main():
    """Runs bootstrap for all combinations of test and training set, 
    and classification type"""
    rows = []

    """For all train and test percentages, find and load that specific script"""
    for train_pct, test_pct in M.COMBOS:
        pred_path = os.path.join(M.PRED_DIR, f"train{train_pct}_test{test_pct}_rf.npz")
        if not os.path.exists(pred_path):
            print(f"[skip] missing {pred_path}")
            continue

        with np.load(pred_path, allow_pickle=True) as z:
            y_true = z["y_true"]
            text_labels = list(z["text_labels"])
            y_preds_by_seed = {k: z[k] for k in z.files if k.startswith("y_pred_seed")}

        """calculate the metrics"""
        precs, recs, f1s = bootstrap_per_class(y_true, y_preds_by_seed, len(text_labels))

        "Calculate mean and std and append to rows"
        for ci, cname in enumerate(text_labels):
            row = {
                "classifier": "rf",
                "train_pct": train_pct,
                "test_pct": test_pct,
                "class_name": cname,
                "precision_mean": precs[:, ci].mean() * 100,
                "precision_std": precs[:, ci].std(ddof=0) * 100,
                "recall_mean": recs[:, ci].mean() * 100,
                "recall_std": recs[:, ci].std(ddof=0) * 100,
                "f1_mean": f1s[:, ci].mean() * 100,
                "f1_std": f1s[:, ci].std(ddof=0) * 100,
            }
            rows.append(row)
            print(
                f"train{train_pct}/test{test_pct} {cname:8s} "
                f"f1={row['f1_mean']:.1f}+/-{row['f1_std']:.1f}"
            )
    """Make rows a file and save it"""
    df = pd.DataFrame(rows)
    out_path = os.path.join(M.THIS_DIR, "bootstrap_results_per_class.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved consolidated per-class results to: {out_path}")


if __name__ == "__main__":
    main()
