#!/usr/bin/env python3

import os

import numpy as np
import pandas as pd

import manifest as M
from feature_categories import category_of


IN_CSV = os.path.join(M.THIS_DIR, "feature_importance.csv")
OUT_CSV = os.path.join(M.THIS_DIR, "feature_importance_deviation.csv")

TRAIN_LEVELS = M.TRAIN_LEVELS  # [100, 50, 25]
MIN_IMPORTANCE_FOR_RELATIVE = 0.001  # ~1/5th of the "equal share" baseline (1/207 ~ 0.0048)


def main():
    df = pd.read_csv(IN_CSV)
    """Merges everything with same name to combat duplicants."""
    df = df.groupby(["feature_name", "train_pct"], as_index=False)["importance"].sum()
    pivot = df.pivot(index="feature_name", columns="train_pct", values="importance")
    pivot = pivot[TRAIN_LEVELS]  # fixed column order
    pivot.columns = [f"importance_{p}" for p in TRAIN_LEVELS]


    """For each feature compute mean, absulute effect and relative effect"""
    grand_mean = pivot.mean(axis=1)
    pivot["grand_mean"] = grand_mean
    for pct in TRAIN_LEVELS:
        pivot[f"effect_{pct}"] = pivot[f"importance_{pct}"] - grand_mean
        pivot[f"rel_effect_{pct}"] = pivot[f"effect_{pct}"] / grand_mean

    """Coefficient of variation across the 3 truncations: std / grand_mean.
    Low cv = importance barely moves across train100/50/25 -> a stable feature."""
    level_cols = [f"importance_{pct}" for pct in TRAIN_LEVELS]
    pivot["std_across_levels"] = pivot[level_cols].std(axis=1, ddof=0)
    pivot["cv"] = pivot["std_across_levels"] / grand_mean

    """Saves it"""
    pivot = pivot.reset_index()
    pivot.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}\n")

    eligible = pivot[pivot["grand_mean"] >= MIN_IMPORTANCE_FOR_RELATIVE]


    """Prints out top features for each percentage for raw effect and relative effect."""
    for pct in TRAIN_LEVELS:
        print(f"=== train{pct}%: top 10 by RAW effect (importance_{pct} - grand_mean) ===")
        top_raw = pivot.sort_values(f"effect_{pct}", ascending=False).head(10)
        for _, row in top_raw.iterrows():
            print(
                f"  {row['feature_name']:35s} [{category_of(row['feature_name'])}] effect={row[f'effect_{pct}']:+.4f}  "
                f"(importance_{pct}={row[f'importance_{pct}']:.4f}, grand_mean={row['grand_mean']:.4f})"
            )

        print(f"\n=== train{pct}%: top 10 by RELATIVE effect (excluding grand_mean < {MIN_IMPORTANCE_FOR_RELATIVE}) ===")
        top_rel = eligible.sort_values(f"rel_effect_{pct}", ascending=False).head(10)
        for _, row in top_rel.iterrows():
            print(
                f"  {row['feature_name']:35s} [{category_of(row['feature_name'])}] rel_effect={row[f'rel_effect_{pct}']:+.2f}x  "
                f"(importance_{pct}={row[f'importance_{pct}']:.4f}, grand_mean={row['grand_mean']:.4f})"
            )
        print()

    """Prints out the features whose importance is most stable across all 3 truncations."""
    print(f"=== top 15 MOST STABLE features across train100/50/25 (lowest cv, excluding grand_mean < {MIN_IMPORTANCE_FOR_RELATIVE}) ===")
    most_stable = eligible.sort_values("cv", ascending=True).head(15)
    for _, row in most_stable.iterrows():
        print(
            f"  {row['feature_name']:35s} [{category_of(row['feature_name'])}] cv={row['cv']:.3f}  "
            f"(importance_100={row['importance_100']:.4f}, importance_50={row['importance_50']:.4f}, "
            f"importance_25={row['importance_25']:.4f}, grand_mean={row['grand_mean']:.4f})"
        )
    print()


if __name__ == "__main__":
    main()
