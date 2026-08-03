#!/usr/bin/env python3

import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import manifest as M

N_BOOT = 2000
SEED = 42


def bootstrap_one(y_true, y_pred, n_boot=N_BOOT, seed=SEED):
    """Storage for metrics, run a loop 2000 times, pick 8000 ich with replacement
    then see accuracy (does converge to same mean but shows some variane)"""
    rng = np.random.default_rng(seed)
    n = len(y_true)

    accs = np.empty(n_boot)
    recs = np.empty(n_boot)
    precs = np.empty(n_boot)
    f1s = np.empty(n_boot)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)  # with replacement, same size as original test set
        yt = y_true[idx]
        yp = y_pred[idx]  # same idx as yt -- keeps each star's (true, predicted) pair together

        accs[b] = accuracy_score(yt, yp)
        p, r, f1, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
        precs[b] = p
        recs[b] = r
        f1s[b] = f1

    return {
        "accuracy_mean": accs.mean(), "accuracy_std": accs.std(ddof=0),
        "recall_mean": recs.mean(), "recall_std": recs.std(ddof=0),
        "precision_mean": precs.mean(), "precision_std": precs.std(ddof=0),
        "f1_mean": f1s.mean(), "f1_std": f1s.std(ddof=0),
    }


def bootstrap_multi_seed(y_true, y_preds_by_seed, n_boot=N_BOOT, seed=SEED):
    """Same idea as bootstrap_one, but for classifiers (rf) whose reported
    number is a MEAN across several seeds' models. Each iteration draws ONE
    shared resampled index set, scores every seed's predictions on that same
    resample, and averages across seeds before moving to the next iteration --
    so the resulting mean matches "mean across seeds", while std still reflects
    test-set resampling variance (not seed-to-seed variance, which is a
    separate, already-reported axis of uncertainty)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    seed_preds = list(y_preds_by_seed.values())

    accs = np.empty(n_boot)
    recs = np.empty(n_boot)
    precs = np.empty(n_boot)
    f1s = np.empty(n_boot)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]

        acc_this, rec_this, prec_this, f1_this = [], [], [], []
        for y_pred in seed_preds:
            yp = y_pred[idx]
            acc_this.append(accuracy_score(yt, yp))
            p, r, f1, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
            prec_this.append(p)
            rec_this.append(r)
            f1_this.append(f1)

        accs[b] = np.mean(acc_this)
        recs[b] = np.mean(rec_this)
        precs[b] = np.mean(prec_this)
        f1s[b] = np.mean(f1_this)

    return {
        "accuracy_mean": accs.mean(), "accuracy_std": accs.std(ddof=0),
        "recall_mean": recs.mean(), "recall_std": recs.std(ddof=0),
        "precision_mean": precs.mean(), "precision_std": precs.std(ddof=0),
        "f1_mean": f1s.mean(), "f1_std": f1s.std(ddof=0),
    }


def main():
    """Hunts down all paths to all mixed trainings and testings,
    takes out real y and guessed y, runs bootstrap function, converts
    clculated metrics to percentage,stores in panda excell isch"""
    rows = []
    for train_pct, test_pct in M.COMBOS:
        for clf_name in M.CLASSIFIERS:
            pred_path = os.path.join(M.PRED_DIR, f"train{train_pct}_test{test_pct}_{clf_name}.npz")
            if not os.path.exists(pred_path):
                print(f"[skip] missing {pred_path}")
                continue

            with np.load(pred_path, allow_pickle=True) as z:
                y_true = z["y_true"]
                if clf_name == "rf":
                    y_preds_by_seed = {
                        key: z[key] for key in z.files if key.startswith("y_pred_seed")
                    }
                else:
                    y_pred = z["y_pred"]

            if clf_name == "rf":
                stats = bootstrap_multi_seed(y_true, y_preds_by_seed)
            else:
                stats = bootstrap_one(y_true, y_pred)
            row = {"classifier": clf_name, "train_pct": train_pct, "test_pct": test_pct}
            row.update({k: v * 100 for k, v in stats.items()})  # store as percentages
            rows.append(row)
            print(
                f"train{train_pct}/test{test_pct} {clf_name:10s} "
                f"acc={row['accuracy_mean']:.1f}+/-{row['accuracy_std']:.1f}  "
                f"f1={row['f1_mean']:.1f}+/-{row['f1_std']:.1f}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(M.RESULTS_CSV, index=False)
    print(f"\nSaved consolidated results to: {M.RESULTS_CSV}")


if __name__ == "__main__":
    main()
