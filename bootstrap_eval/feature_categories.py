#!/usr/bin/env python3
"""
Maps each feature_name (band-prefixed, e.g. "g_amplitude") to the type of
light-curve property it measures: statistical, periodicity, stochastic
(incl. the phase-folded Psi_CS/Psi_eta variants), or trend/slope.

Kept per-band on purpose: g_amplitude, r_amplitude and i_amplitude are three
distinct features that just happen to share a category label, so the
exported mapping has one explicit entry per (band, base) pair rather than a
lookup that strips the band prefix off first.
"""

BANDS = ["g", "r", "i"]

# base feature name (post band-prefix) -> category
_CATEGORY_BY_BASE = {
    # statistical / distributional
    "mean": "statistical",
    "median": "statistical",
    "weighted_mean": "statistical",
    "standard_deviation": "statistical",
    "skew": "statistical",
    "kurtosis": "statistical",
    "amplitude": "statistical",
    "percent_amplitude": "statistical",
    "median_absolute_deviation": "statistical",
    "inter_percentile_range_10": "statistical",
    "inter_percentile_range_25": "statistical",
    "magnitude_percentage_ratio_40_5": "statistical",
    "median_buffer_range_percentage_10": "statistical",
    "beyond_1_std": "statistical",
    "beyond_2_std": "statistical",
    "beyond_3_std": "statistical",
    "anderson_darling_normal": "statistical",
    "otsu_mean_diff": "statistical",
    "otsu_std_lower": "statistical",
    "otsu_std_upper": "statistical",
    "otsu_lower_to_all_ratio": "statistical",

    # periodicity / frequency-domain
    "PeriodLS": "periodicity",
    "Period_fit": "periodicity",
    **{f"Freq{f}_harmonics_amplitude_{h}": "periodicity" for f in (1, 2, 3) for h in range(4)},
    **{f"Freq{f}_harmonics_rel_phase_{h}": "periodicity" for f in (1, 2, 3) for h in range(4)},

    # stochastic-process / autocorrelation (incl. folded-curve Psi_* variants)
    "Autocor_length": "stochastic",
    "CAR_mean": "stochastic",
    "CAR_sigma": "stochastic",
    "CAR_tau": "stochastic",
    "eta": "stochastic",
    "eta_e": "stochastic",
    "stetson_K": "stochastic",
    "chi2": "stochastic",
    "Psi_CS": "stochastic",
    "Psi_eta": "stochastic",

    # trend / slope
    "linear_trend": "trend",
    "linear_trend_noise": "trend",
    "linear_trend_sigma": "trend",
    "linear_fit_slope": "trend",
    "linear_fit_slope_sigma": "trend",
    "linear_fit_reduced_chi2": "trend",
    "PairSlopeTrend": "trend",
    "maximum_slope": "trend",
    "Con": "trend",
    "cusum": "trend",
}

# feature_name (e.g. "g_amplitude") -> category, one explicit entry per band.
FEATURE_CATEGORY = {
    f"{band}_{base}": category
    for band in BANDS
    for base, category in _CATEGORY_BY_BASE.items()
}


def category_of(feature_name: str) -> str:
    return FEATURE_CATEGORY[feature_name]
