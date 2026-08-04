import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# reads straight from the per-class bootstrap output (Random Forest only)
CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "bootstrap_eval", "bootstrap_results_per_class.csv",
)

CLASS_ORDER = ["EW", "EA", "RRab", "RRc", "RRd", "RS_CVn", "LPV"]

train_levels = [100, 50, 25]     # one line per training truncation
test_levels = [25, 50, 100]      # x-axis, ascending

df = pd.read_csv(CSV_PATH)

sns.set_theme(style="whitegrid", context="talk")
palette = sns.color_palette("colorblind", n_colors=3)
train_colors = dict(zip(train_levels, palette))

# 2x4 grid: 7 classes used, 1 panel left blank
fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharey=False)
axes = axes.flatten()

for ax, class_name in zip(axes, CLASS_ORDER):
    class_df = df[df["class_name"] == class_name]
    for train_pct in train_levels:
        sub = class_df[class_df["train_pct"] == train_pct].set_index("test_pct")
        y = [sub.loc[t, "f1_mean"] for t in test_levels]
        yerr = [sub.loc[t, "f1_std"] for t in test_levels]
        ax.errorbar(
            test_levels, y, yerr=yerr,
            marker="o", markersize=8, linewidth=2, capsize=4,
            color=train_colors[train_pct],
            label=f"Train {train_pct}%",
        )
    ax.set_title(class_name, fontsize=18, fontweight="bold")
    ax.set_xticks(test_levels)
    ax.set_xticklabels([f"{t}%" for t in test_levels])
    ax.set_xlabel("Test truncation", fontsize=12)
    ax.set_ylabel("F1 (%)", fontsize=12)
    ax.tick_params(axis="both", labelsize=11, pad=2)
    ax.grid(True, alpha=0.3)

axes[-1].axis("off")  # unused 8th slot in the 2x4 grid

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig("f1_per_class_vs_test_truncation.png", dpi=300, bbox_inches="tight")
plt.show()
