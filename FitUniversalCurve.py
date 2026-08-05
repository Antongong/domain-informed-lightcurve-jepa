"""
Fits three candidate models to the per-class bootstrap results
(bootstrap_eval/bootstrap_results_per_class.csv), one training truncation
level (100/50/25) at a time, to compare which model/sharing structure best
describes the F1-vs-test-truncation relationship:

  1. power_shared      F1(x) = alpha_c - beta   * x^(-gamma)     (beta, gamma shared across classes)
  2. power_class_beta  F1(x) = alpha_c - beta_c * x^(-gamma)     (only gamma shared; alpha AND beta per class)
  3. exponential       F1(x) = alpha_c - beta   * exp(-gamma*x)  (beta, gamma shared; different family)

x = test truncation fraction (0.25/0.50/1.00). Model 1 is the original fit;
models 2 and 3 were added because model 1 fit some classes (RS_CVn in
particular) poorly -- model 2 tests whether letting beta vary per class (not
just alpha) fixes that, model 3 tests whether a different functional family
fits better while keeping model 1's sharing structure.

Outputs:
  - universal_fit_params.csv: fitted parameters + R^2, all 3 models x all 3 train% levels
  - f1_fit_power_shared.png / f1_fit_power_class_beta.png / f1_fit_exponential.png:
    one 7-panel figure per model, real data points + that model's fitted curve
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "bootstrap_eval", "bootstrap_results_per_class.csv",
)

CLASS_ORDER = ["EW", "EA", "RRab", "RRc", "RRd", "RS_CVn", "LPV"]
TRAIN_LEVELS = [100, 50, 25]
TEST_LEVELS = [25, 50, 100]
N_CLASSES = len(CLASS_ORDER)

df = pd.read_csv(CSV_PATH)


def _stack_data(train_pct):
    """Common data-gathering step shared by every model's fit function."""
    sub = df[df["train_pct"] == train_pct]
    xs, class_idxs, ys = [], [], []
    for ci, cname in enumerate(CLASS_ORDER):
        crow = sub[sub["class_name"] == cname].set_index("test_pct")
        for test_pct in TEST_LEVELS:
            xs.append(test_pct / 100.0)
            class_idxs.append(ci)
            ys.append(crow.loc[test_pct, "f1_mean"])
    return np.array(xs), np.array(class_idxs), np.array(ys)


def _r2(y, y_pred):
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot


def _alpha0(train_pct):
    """Initial guess for each class's alpha: its own test100 F1."""
    sub = df[df["train_pct"] == train_pct]
    return [
        sub[(sub["class_name"] == c) & (sub["test_pct"] == 100)]["f1_mean"].values[0]
        for c in CLASS_ORDER
    ]


# ---------------------------------------------------------------
# Model 1 (original): F1 = alpha_c - beta * x^(-gamma), beta/gamma shared
# ---------------------------------------------------------------
def _model_power_shared(X, beta, gamma, *alphas):
    x, class_idx = X
    alphas = np.asarray(alphas)
    a = alphas[class_idx.astype(int)]
    return a - beta * np.power(x, -gamma)


def fit_power_shared(train_pct):
    xs, class_idxs, ys = _stack_data(train_pct)
    p0 = [10.0, 1.0] + _alpha0(train_pct)
    bounds = ([0, 0] + [0] * N_CLASSES, [np.inf, np.inf] + [100] * N_CLASSES)
    popt, _ = curve_fit(_model_power_shared, (xs, class_idxs), ys, p0=p0, bounds=bounds, maxfev=20000)
    beta, gamma, alphas = popt[0], popt[1], dict(zip(CLASS_ORDER, popt[2:]))
    r2 = _r2(ys, _model_power_shared((xs, class_idxs), *popt))
    return {"beta": beta, "gamma": gamma, "alphas": alphas, "betas": None, "r2": r2}


# ---------------------------------------------------------------
# Model 2: F1 = alpha_c - beta_c * x^(-gamma), only gamma shared
# ---------------------------------------------------------------
def _model_power_class_beta(X, gamma, *alpha_beta):
    x, class_idx = X
    n = len(alpha_beta) // 2
    alphas = np.asarray(alpha_beta[:n])
    betas = np.asarray(alpha_beta[n:])
    ci = class_idx.astype(int)
    return alphas[ci] - betas[ci] * np.power(x, -gamma)


def fit_power_class_beta(train_pct):
    xs, class_idxs, ys = _stack_data(train_pct)
    p0 = [1.0] + _alpha0(train_pct) + [10.0] * N_CLASSES
    bounds = (
        [0] + [0] * N_CLASSES + [0] * N_CLASSES,
        [np.inf] + [100] * N_CLASSES + [np.inf] * N_CLASSES,
    )
    popt, _ = curve_fit(_model_power_class_beta, (xs, class_idxs), ys, p0=p0, bounds=bounds, maxfev=20000)
    gamma = popt[0]
    alphas = dict(zip(CLASS_ORDER, popt[1:1 + N_CLASSES]))
    betas = dict(zip(CLASS_ORDER, popt[1 + N_CLASSES:]))
    r2 = _r2(ys, _model_power_class_beta((xs, class_idxs), *popt))
    return {"beta": None, "gamma": gamma, "alphas": alphas, "betas": betas, "r2": r2}


# ---------------------------------------------------------------
# Model 3: F1 = alpha_c - beta * exp(-gamma * x), beta/gamma shared
# ---------------------------------------------------------------
def _model_exponential(X, beta, gamma, *alphas):
    x, class_idx = X
    alphas = np.asarray(alphas)
    a = alphas[class_idx.astype(int)]
    return a - beta * np.exp(-gamma * x)


def fit_exponential(train_pct):
    xs, class_idxs, ys = _stack_data(train_pct)
    p0 = [10.0, 1.0] + _alpha0(train_pct)
    bounds = ([0, 0] + [0] * N_CLASSES, [np.inf, np.inf] + [100] * N_CLASSES)
    popt, _ = curve_fit(_model_exponential, (xs, class_idxs), ys, p0=p0, bounds=bounds, maxfev=20000)
    beta, gamma, alphas = popt[0], popt[1], dict(zip(CLASS_ORDER, popt[2:]))
    r2 = _r2(ys, _model_exponential((xs, class_idxs), *popt))
    return {"beta": beta, "gamma": gamma, "alphas": alphas, "betas": None, "r2": r2}


MODELS = {
    "power_shared": (fit_power_shared, "F1 = alpha - beta*x^(-gamma)  [beta,gamma shared]"),
    "power_class_beta": (fit_power_class_beta, "F1 = alpha_c - beta_c*x^(-gamma)  [gamma shared only]"),
    "exponential": (fit_exponential, "F1 = alpha - beta*exp(-gamma*x)  [beta,gamma shared]"),
}


def eval_curve(model_name, fit_result, cname, x_smooth):
    """Smooth curve values along x_smooth, for one class, for plotting."""
    a = fit_result["alphas"][cname]
    gamma = fit_result["gamma"]
    if model_name == "power_class_beta":
        b = fit_result["betas"][cname]
        return a - b * np.power(x_smooth, -gamma)
    elif model_name == "exponential":
        return a - fit_result["beta"] * np.exp(-gamma * x_smooth)
    else:  # power_shared
        return a - fit_result["beta"] * np.power(x_smooth, -gamma)


def make_plot(model_name, fitted_by_train, out_path, title):
    """Makes the plot"""
    sns.set_theme(style="whitegrid", context="talk")
    palette = sns.color_palette("colorblind", n_colors=3)
    train_colors = dict(zip(TRAIN_LEVELS, palette))

    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharey=False)
    axes = axes.flatten()
    x_smooth = np.linspace(0.20, 1.0, 200)

    for ax, cname in zip(axes, CLASS_ORDER):
        for train_pct in TRAIN_LEVELS:
            fit_result = fitted_by_train[train_pct]

            crow = df[(df["train_pct"] == train_pct) & (df["class_name"] == cname)].set_index("test_pct")
            y_real = [crow.loc[t, "f1_mean"] for t in TEST_LEVELS]
            x_real = [t / 100.0 for t in TEST_LEVELS]
            ax.scatter(x_real, y_real, color=train_colors[train_pct], s=60, zorder=3)

            y_smooth = eval_curve(model_name, fit_result, cname, x_smooth)
            ax.plot(
                x_smooth, y_smooth, color=train_colors[train_pct], linewidth=2,
                label=f"Train {train_pct}%", zorder=2,
            )

        ax.set_title(cname, fontsize=18, fontweight="bold")
        ax.set_xlabel("Test truncation", fontsize=12)
        ax.set_ylabel("F1 (%)", fontsize=12)
        ax.set_xticks([0.25, 0.5, 1.0])
        ax.set_xticklabels(["25%", "50%", "100%"])
        ax.tick_params(axis="both", labelsize=11, pad=2)
        ax.grid(True, alpha=0.3)

    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, fontsize=15, y=1.0)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    """For all models for all of the training percentages, get results from fitting,
    use panda and save to outdir and make plots using make_plot"""
    all_rows = []
    here = os.path.dirname(os.path.abspath(__file__))

    for model_name, (fit_fn, description) in MODELS.items():
        fitted_by_train = {}
        for train_pct in TRAIN_LEVELS:
            result = fit_fn(train_pct)
            fitted_by_train[train_pct] = result

            for cname in CLASS_ORDER:
                all_rows.append({
                    "model": model_name,
                    "train_pct": train_pct,
                    "class_name": cname,
                    "alpha": result["alphas"][cname],
                    "beta": result["betas"][cname] if result["betas"] is not None else result["beta"],
                    "gamma": result["gamma"],
                    "r2": result["r2"],
                })

        out_png = os.path.join(here, f"f1_fit_{model_name}.png")
        make_plot(model_name, fitted_by_train, out_png, description)

    fit_df = pd.DataFrame(all_rows)
    out_csv = os.path.join(here, "universal_fit_params.csv")
    fit_df.to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
