import matplotlib.pyplot as plt
import seaborn as sns

# ---- your data: macro F1 (%) for each classifier, by (train%, test%) ----
data = {
    "Logistic Regression": {
        100: {100: 71.0, 50: 66.1, 25: 61.5},
        50:  {100: 70.0, 50: 66.9, 25: 64.4},
        25:  {100: 66.4, 50: 65.9, 25: 63.7},
    },
    "k-NN": {
        100: {100: 68.9, 50: 62.7, 25: 55.9},
        50:  {100: 65.0, 50: 62.7, 25: 58.6},
        25:  {100: 58.6, 50: 57.3, 25: 55.9},
    },
    "Random Forest": {
        100: {100: 80.4, 50: 76.9, 25: 69.0},
        50:  {100: 80.5, 50: 78.1, 25: 71.5},
        25:  {100: 79.8, 50: 77.6, 25: 73.2},
    },
    "MLP": {
        100: {100: 74.4, 50: 68.9, 25: 64.4},
        50:  {100: 72.0, 50: 68.5, 25: 63.1},
        25:  {100: 68.9, 50: 68.1, 25: 66.4},
    },
}

train_levels = [100, 50, 25]     # one line per training truncation
test_levels = [25, 50, 100]      # x-axis, ascending

sns.set_theme(style="whitegrid", context="talk")
palette = sns.color_palette("colorblind", n_colors=3)
train_colors = dict(zip(train_levels, palette))

fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharey=True)
axes = axes.flatten()

for ax, (clf_name, clf_data) in zip(axes, data.items()):
    for train_pct in train_levels:
        y = [clf_data[train_pct][t] for t in test_levels]
        ax.plot(
            test_levels, y,
            marker="o", markersize=8, linewidth=2,
            color=train_colors[train_pct],
            label=f"Train {train_pct}%",
        )
    ax.set_title(clf_name, fontsize=18, fontweight="bold")
    ax.set_xticks(test_levels)
    ax.set_xticklabels([f"{t}%" for t in test_levels])
    ax.set_xlabel("Test truncation", fontsize=12)
    ax.set_ylabel("Macro F1 (%)", fontsize=12)
    ax.tick_params(axis="both", labelsize=11, pad=2)
    ax.grid(True, alpha=0.3)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig("f1_vs_test_truncation.png", dpi=300, bbox_inches="tight")
plt.show()
