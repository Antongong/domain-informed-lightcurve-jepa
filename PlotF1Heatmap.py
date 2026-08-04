import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# reads straight from the bootstrap pipeline's output
CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "bootstrap_eval", "bootstrap_results.csv",
)

CLASSIFIER_ORDER = ["logistic", "knn", "rf", "mlp"]
CLASSIFIER_DISPLAY = {
    "logistic": "Logistic Regression",
    "knn": "k-NN",
    "rf": "Random Forest",
    "mlp": "MLP",
}

# train stays descending (100 -> 25, top to bottom); test is ascending
# (25 -> 100, left to right) so the matched train=test diagonal is visible
TRAIN_LEVELS = [100, 50, 25]
TEST_LEVELS = [25, 50, 100]

df = pd.read_csv(CSV_PATH)

# one shared color scale across all 4 panels, so "brighter" always means
# "higher F1" no matter which classifier you're looking at
vmin = df["f1_mean"].min()
vmax = df["f1_mean"].max()

sns.set_theme(style="white", context="talk")

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
axes = axes.flatten()

for ax, clf_name in zip(axes, CLASSIFIER_ORDER):
    clf_df = df[df["classifier"] == clf_name]
    pivot = clf_df.pivot(index="train_pct", columns="test_pct", values="f1_mean")
    pivot = pivot.reindex(index=TRAIN_LEVELS, columns=TEST_LEVELS)

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        vmin=vmin,
        vmax=vmax,
        cbar=False,
        linewidths=1,
        linecolor="white",
        ax=ax,
        annot_kws={"fontsize": 13},
    )
    ax.set_title(CLASSIFIER_DISPLAY[clf_name], fontsize=18, fontweight="bold")
    ax.set_xlabel("Test truncation", fontsize=12)
    ax.set_ylabel("Train truncation", fontsize=12)
    ax.set_xticklabels([f"{t}%" for t in TEST_LEVELS], fontsize=11)
    ax.set_yticklabels([f"{t}%" for t in TRAIN_LEVELS], fontsize=11, rotation=0)

# one shared colorbar for the whole figure instead of 4 separate ones
fig.subplots_adjust(right=0.9, wspace=0.35, hspace=0.4)
cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(vmin=vmin, vmax=vmax))
fig.colorbar(sm, cax=cbar_ax, label="Macro F1 (%)")

fig.suptitle("Macro F1 by train/test truncation", fontsize=16, y=0.98)
fig.savefig("f1_heatmap_matrix.png", dpi=300, bbox_inches="tight")
plt.show()
