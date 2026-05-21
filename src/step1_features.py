from step0_eda import *
"""
HBAAC 2026 — Stage 1: Feature Engineering Pipeline
====================================================
Continuation of the same Kaggle notebook immediately after Stage 0.

HOW TO USE
----------
Paste this entire cell below the Stage 0 cell.  Stage 0 must have already
produced the following variables in the notebook's global scope:

    df             pd.DataFrame   parsed transaction data
    panel          pd.DataFrame   daily (Date × ItemCode) panel
    sku_stats      pd.DataFrame   per-SKU stats + profit weights + taxonomy
    folds          list[dict]     CV fold boundaries (5 folds)
    naive_results  dict           naive baseline WRMSSE results
    CONFIG         dict           Stage 0 configuration (will be updated here)

Stage 1 writes to CONFIG["OUTPUT_DIR"] and returns these variables for later
stages in the same session:

    sku_features      pd.DataFrame   extended SKU static feature table
    panel_wide        dict           wide-format DataFrames (qty/price/gross/return)
    hard_zero_skus    set[str]       SKUs predicted 0 for all 56 days
    train_datasets    dict           {week_idx (1-8): pd.DataFrame}
    inference_ds      pd.DataFrame   inference features (origin = TRAIN_END)
"""

# ==============================================================================
# 0 — STAGE 1 IMPORTS  (only what Stage 0 did not already import)
# ==============================================================================
import gc
import itertools
import os
import warnings
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")


# ==============================================================================
# 1 — STAGE 1 CONFIGURATION  (merged into existing CONFIG)
# ==============================================================================
# NOTE: CONFIG was created in Stage 0.  We extend it here with Stage 1 keys.
# All Stage 0 keys (TRAIN_START, TRAIN_END, OUTPUT_DIR, …) remain accessible.

CONFIG.update({
    # Update OUPUT DIR
    "OUTPUT_DIR": Path("/kaggle/working/feature_outputs"),
    
    # ── Updated cutoff based on EDA findings ───────────────────────────────
    # Pre-2022 data is anomalously sparse (< 700 tx/month) and shows an
    # "invisible wall" in the SKU lifespan scatter.  All feature computation
    # and CV training use only post-cutoff data.
    "EARLY_DATA_CUTOFF": pd.Timestamp("2022-01-01"),   # overrides Stage 0 value

    # ── Lag depths ─────────────────────────────────────────────────────────
    # IMPORTANT: ALL lags are relative to the forecast ORIGIN DATE (the last
    # known day before prediction starts).  For direct/non-recursive models,
    # every lag is valid regardless of the forecast horizon h because features
    # are computed at origin, not at t+h.
    # ACF/PACF confirmed strong weekly AR(1) structure (PACF ≈ 0.35 at lag-7).
    "LAGS": [1, 7, 14, 21, 28, 35, 42, 56],

    # ── Rolling mean configs: (window W, offset O) ─────────────────────────
    # rmean_{W}_{O} = mean of qty values from (origin − O − W) to (origin − O − 1)
    # i.e. the W-day window that ends O days before the origin.
    #   rmean_7_0   : last week from origin        → short-term level
    #   rmean_7_28  : same week 4 weeks ago         → recent same-period comp
    #   rmean_7_364 : same week last year           → annual comp
    #   rmean_28_0  : last 4 weeks from origin      → monthly baseline
    #   rmean_28_28 : 4-week window 4 weeks ago     → MoM change signal
    #   rmean_28_364: 4-week window last year        → YoY change signal
    "ROLLING_CONFIGS": [
        (7,   0),    # rmean_7_0  — short-term level
        (14,  0),    # rmean_14_0 — 2-week recency anchor (NEW)
        (7,   28),   # rmean_7_28 — same-week 4-weeks-ago comp
        (28,  0),    # rmean_28_0 — monthly baseline
        (28,  28),   # rmean_28_28 — MoM change signal
    ],

    # ── Rolling stat configs: (stat, window W, offset O) ───────────────────
    # Same window/offset semantics as ROLLING_CONFIGS.
    "ROLLING_STAT_CONFIGS": [
        ("std",       28, 0),   # rstd_28_0   — demand volatility
        ("max",       28, 0),   # rmax_28_0   — peak demand signal
        ("zero_rate", 28, 0),   # rzero_28_0  — recent sparsity state
    ],

    # ── Price / return windows ──────────────────────────────────────────────
    "PRICE_LAGS":      [7, 28],
    "RETURN_WINDOWS":  [7, 28],   # return_fraction_{W}

    # ── Training dataset construction ───────────────────────────────────────
    # N_ORIGIN_SAMPLE: how many origin dates to sample from each training fold.
    # Reducing this is the primary memory lever; 150 gives reliable estimates.
    "N_ORIGIN_SAMPLE":   150,
    "ORIGIN_BATCH_SIZE":  30,    # origins processed simultaneously (memory control)

    # Minimum days of history required at the origin date for a sample to be
    # valid.  Equals max lag (56) + safety margin so features are available.
    "MIN_HISTORY_DAYS": 60,

    # ── Forecast structure ──────────────────────────────────────────────────
    "N_WEEKS":          8,        # total week-specialist models
    "DAYS_PER_WEEK":    7,
    "HORIZON_TOTAL":    56,

    # ── Hard-zero rules ─────────────────────────────────────────────────────
    # SKU-level: if post-2022 zero_rate > this AND last-28d mean == 0 → predict 0.
    # Empirically confirmed for SKU-09458 (96.3 % zeros) and SKU-00005 (75.7 %).
    "HARD_ZERO_ZERO_RATE": 0.90,
    "HARD_ZERO_LAST28_MEAN": 0.0,

    # Regime-change thresholds
    "DECLINING_THRESHOLD":     0.60,   # rmean_28_0 < threshold × hist_mean
    "ACCELERATING_THRESHOLD":  1.20,   # rmean_7_0  > threshold × rmean_28_0


})

CONFIG["OUTPUT_DIR"].mkdir(parents=True, exist_ok=True)

# ── Derived constants (computed once from CONFIG) ───────────────────────────
_CUTOFF        = CONFIG["EARLY_DATA_CUTOFF"]
_TRAIN_END     = CONFIG["TRAIN_END"]
_ORIGIN_FINAL  = _TRAIN_END           # last possible origin for final submission
_OUTPUT_DIR    = CONFIG["OUTPUT_DIR"]

print("Stage 1 configuration loaded.")
print(f"  Effective training window : {_CUTOFF.date()} → {_TRAIN_END.date()}")
print(f"  Usable calendar days      : "
      f"{(pd.date_range(_CUTOFF, _TRAIN_END, freq='D')).shape[0]}")


# ==============================================================================
# 2 — EXTENDED SKU STATIC FEATURES  (Stage 1A)
# ==============================================================================

def extend_sku_static_features(
    panel: pd.DataFrame,
    sku_stats: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extend the Stage 0 sku_stats table with post-cutoff-specific features
    and encode categorical columns for model consumption.

    New columns added
    -----------------
    zero_rate_post2022     : float32  — zero-rate computed on post-2022 data only
    historical_mean_post22 : float32  — mean net_qty on non-zero post-2022 days
    last_28d_mean          : float32  — rmean_28_0 from the final TRAIN_END origin
    demand_declining       : int8     — 1 if last_28d_mean < 0.6 × historical_mean
    demand_accelerating    : int8     — computed at TRAIN_END; updated per origin at train time
    profit_tier_enc        : int8     — A=0, B=1, C=2, D=3
    demand_density_enc     : int8     — Dense=0, Intermittent=1, Sparse=2
    ets_available          : int8     — 1 if ETS model will be fitted for this SKU
                                        (A-tier Dense or Intermittent, not hard-zero)

    Parameters
    ----------
    panel     : daily panel from Stage 0 (Date, ItemCode, net_qty, …)
    sku_stats : per-SKU statistics from Stage 0 compute_sku_stats()

    Returns
    -------
    sku_features : pd.DataFrame indexed by ItemCode (sorted), ready for fast lookup
    """
    df = sku_stats.copy()

    # ── Post-2022 zero rate ─────────────────────────────────────────────────
    post_panel = panel[panel["Date"] >= _CUTOFF].copy()
    n_post_days = post_panel["Date"].nunique()

    post_active = (
        post_panel[post_panel["net_qty"] > 0]
        .groupby("ItemCode", observed=True)["net_qty"]
        .agg(["count", "mean"])
        .rename(columns={"count": "_post_active_days", "mean": "historical_mean_post22"})
    )
    df = df.merge(post_active, on="ItemCode", how="left")
    df["_post_active_days"]   = df["_post_active_days"].fillna(0)
    df["historical_mean_post22"] = df["historical_mean_post22"].fillna(0).astype("float32")
    df["zero_rate_post2022"]  = (
        1.0 - df["_post_active_days"] / max(n_post_days, 1)
    ).astype("float32")
    df.drop(columns=["_post_active_days"], inplace=True)

    # ── Last-28-day mean: rmean_28_0 evaluated at TRAIN_END origin ──────────
    # Build a temporary wide-qty slice for the last 28 training days.
    last28_panel = panel[
        (panel["Date"] > _TRAIN_END - pd.Timedelta(days=28)) &
        (panel["Date"] <= _TRAIN_END)
    ]
    last28_mean = (
        last28_panel.groupby("ItemCode", observed=True)["net_qty"]
        .mean()
        .rename("last_28d_mean")
    )
    df = df.merge(last28_mean, on="ItemCode", how="left")
    df["last_28d_mean"] = df["last_28d_mean"].fillna(0).astype("float32")

    # ── Regime-change flags (computed at TRAIN_END; re-derived per origin at train time) ──
    df["demand_declining"] = (
        df["last_28d_mean"] < CONFIG["DECLINING_THRESHOLD"] * df["historical_mean_qty"]
    ).astype("int8")

    # demand_accelerating cannot be computed statically without rmean_7_0;
    # set placeholder — overwritten per-origin in the batch feature computation.
    df["demand_accelerating"] = np.int8(0)

    # ── Categorical encoding ────────────────────────────────────────────────
    tier_map    = {"A": 0, "B": 1, "C": 2, "D": 3}
    density_map = {"Dense": 0, "Intermittent": 1, "Sparse": 2}
    df["profit_tier_enc"]     = df["profit_tier"].map(tier_map).fillna(3).astype("int8")
    df["demand_density_enc"]  = (
        df["demand_density"].map(density_map).fillna(2).astype("int8")
    )

    # ── ETS availability flag ───────────────────────────────────────────────
    # ETS fitted for A-tier SKUs that are Dense or Intermittent (not Sparse)
    # and not flagged as hard-zero candidates.
    is_a_tier   = df["profit_tier"] == "A"
    not_sparse  = df["demand_density"].isin(["Dense", "Intermittent"])
    not_hzero   = ~(
        (df["zero_rate_post2022"] >= CONFIG["HARD_ZERO_ZERO_RATE"]) &
        (df["last_28d_mean"] <= CONFIG["HARD_ZERO_LAST28_MEAN"])
    )
    df["ets_available"] = (is_a_tier & not_sparse & not_hzero).astype("int8")

    # ── Final float casts for model consumption ─────────────────────────────
    for col in ["profit_weight", "historical_mean_qty", "historical_std_qty",
                "mean_unit_price_global", "mean_margin_rate"]:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    df["days_since_first_sale"] = df["days_since_first_sale"].fillna(0).astype("float32")
    df["is_discontinued"]       = df["is_discontinued"].fillna(False).astype("int8")
    df["has_pre_cutoff_only"]   = df["has_pre_cutoff_only"].fillna(False).astype("int8")

    # ── Index by ItemCode for O(1) lookup ───────────────────────────────────
    df = df.set_index("ItemCode").sort_index()

    print(
        f"  SKU features extended: {len(df):,} SKUs | "
        f"ETS-eligible: {df['ets_available'].sum():,} | "
        f"Demand-declining: {df['demand_declining'].sum():,}"
    )
    return df


# ==============================================================================
# 3 — WIDE PANEL CONSTRUCTION  (Stage 1B)
# ==============================================================================

def build_panel_wide(
    panel: pd.DataFrame,
    date_range: pd.DatetimeIndex,
) -> Dict[str, pd.DataFrame]:
    """
    Convert the long daily panel to wide-format DataFrames for O(1) date-based
    feature lookups during batch feature computation.

    All wide DataFrames share the same index (date_range) and columns (sorted
    ItemCodes).  Missing (Date, ItemCode) combinations are filled as follows:
      - qty    : 0   (no transaction = zero net sales)
      - gross  : 0
      - return : 0
      - price  : NaN → forward-filled per SKU (price persists until next observed)

    Parameters
    ----------
    panel      : Stage 0 daily panel (Date, ItemCode, net_qty, …)
    date_range : full calendar date range for the training period

    Returns
    -------
    dict with keys 'qty', 'price', 'gross', 'return'
    Each value is a pd.DataFrame of shape (len(date_range), n_skus), float32.
    """
    # MASTER SKU ORDER
    all_skus = sorted(panel["ItemCode"].unique())

    def _pivot(col, fill):
        return (
            panel.pivot_table(
                index="Date",
                columns="ItemCode",
                values=col,
                aggfunc="sum",
                fill_value=fill,
            )
            .reindex(index=date_range, fill_value=fill)
            .reindex(columns=all_skus, fill_value=fill)
            .astype("float32")
        )

    print("  Building wide qty panel …")
    qty_wide = _pivot("net_qty", 0)

    print("  Building wide gross-qty panel …")
    gross_wide = _pivot("gross_qty", 0)

    print("  Building wide return-qty panel …")
    return_wide = _pivot("return_qty", 0)

    print("  Building wide price panel (with forward-fill) …")

    price_wide = (
        panel.pivot_table(
            index="Date",
            columns="ItemCode",
            values="mean_unit_price",
            aggfunc="mean",
        )
        .reindex(index=date_range)
        .reindex(columns=all_skus)   # <<< CRITICAL FIX
        .astype("float32")
        .ffill()
    )

    # optional defensive fill
    price_wide = price_wide.fillna(0)

    # HARD ASSERTS
    assert qty_wide.shape == price_wide.shape
    assert qty_wide.shape == gross_wide.shape
    assert qty_wide.shape == return_wide.shape

    print(
        f"  Wide panels ready: {qty_wide.shape[0]} dates × "
        f"{qty_wide.shape[1]:,} SKUs | "
        f"memory ≈ {qty_wide.memory_usage(deep=True).sum() / 1e6:.0f} MB each"
    )

    return {
        "qty": qty_wide,
        "price": price_wide,
        "gross": gross_wide,
        "return": return_wide,
    }


# ==============================================================================
# 4 — HARD-ZERO SKU IDENTIFICATION  (Stage 1C)
# ==============================================================================

def identify_hard_zero_skus(sku_features: pd.DataFrame) -> Set[str]:
    """
    Identify SKUs whose forecasts should be overridden to 0 for all 56 days.

    Rules applied (OR logic — any matching rule triggers the override)
    ----------------------------------------------------------------
    Rule 1 — Extreme sparsity + silent recent period:
              zero_rate_post2022 ≥ HARD_ZERO_ZERO_RATE AND last_28d_mean == 0
    Rule 2 — No post-cutoff activity at all:
              has_pre_cutoff_only == 1
    Rule 3 — Discontinued SKU:
              is_discontinued == 1

    NOTE: Sunday overrides and Tết overrides are applied in post-processing
    (Stage 5), not here, because they are day-level, not SKU-level.

    Parameters
    ----------
    sku_features : extended SKU feature table, indexed by ItemCode

    Returns
    -------
    set of ItemCode strings
    """
    sf = sku_features

    rule1 = (
        (sf["zero_rate_post2022"] >= CONFIG["HARD_ZERO_ZERO_RATE"]) &
        (sf["last_28d_mean"] <= CONFIG["HARD_ZERO_LAST28_MEAN"])
    )
    rule2 = sf["has_pre_cutoff_only"] == 1
    rule3 = sf["is_discontinued"] == 1

    mask = rule1 | rule2 | rule3
    hard_zero_set = set(sf.index[mask].tolist())

    print(f"  Hard-zero SKUs identified: {len(hard_zero_set):,}")
    print(f"    Rule 1 (extreme sparse + silent) : {rule1.sum():,}")
    print(f"    Rule 2 (pre-cutoff only)          : {rule2.sum():,}")
    print(f"    Rule 3 (discontinued)             : {rule3.sum():,}")

    # Print the top-7 by weight for auditing
    top7 = ["SKU-00003", "SKU-00002", "SKU-09458", "SKU-00005", "SKU-08589", "SKU-12534", "SKU-09760"]
    for sku in top7:
        if sku in sf.index:
            flag = "HARD-ZERO" if sku in hard_zero_set else "will be modelled"
            w = sf.at[sku, "profit_weight"]
            print(f"    {sku}: weight={w*100:.2f}%  → {flag}")

    return hard_zero_set



# ==============================================================================
# 6 — CORE FEATURE COMPUTATION  (Stage 1D)
# ==============================================================================

_LAG_FEATURE_NAMES   = [f"lag_{k}"   for k in CONFIG["LAGS"]]
_RMEAN_FEATURE_NAMES = [f"rmean_{W}_{O}" for W, O in CONFIG["ROLLING_CONFIGS"]]
_RSTAT_FEATURE_NAMES = [f"r{stat}_{W}_{O}"
                        for stat, W, O in CONFIG["ROLLING_STAT_CONFIGS"]]
_PRICE_FEATURE_NAMES = [f"price_lag{k}" for k in CONFIG["PRICE_LAGS"]] + ["price_change"]
_RETURN_FEATURE_NAMES= [f"return_fraction_{W}" for W in CONFIG["RETURN_WINDOWS"]]
_MOMENTUM_FEATURES   = ["mom_signal", "demand_declining", "demand_accelerating",
                         "last_28d_mean", "days_since_last_sale",
                         "recent_zero_streak"]   # NEW — consecutive zero-sales days before origin
_ORIGIN_CAL_FEATURES = ["origin_dow", "origin_month",
                         "origin_day_of_month", "origin_week_of_year"]

# Static SKU features used directly in the model (categorical-encoded)
_SKU_STATIC_FEATURES = [
    "profit_weight", "profit_tier_enc", "demand_density_enc",
    "historical_mean_qty", "historical_mean_post22",
    "historical_std_qty", "zero_rate_post2022",
    "mean_unit_price_global", "mean_margin_rate",
    "days_since_first_sale", "is_discontinued", "has_pre_cutoff_only",
    "ets_available",
]

# All non-target, non-metadata feature column names in the training dataset
ALL_BASE_FEATURE_NAMES = (
    _LAG_FEATURE_NAMES +
    _RMEAN_FEATURE_NAMES +
    _RSTAT_FEATURE_NAMES +
    _PRICE_FEATURE_NAMES +
    _RETURN_FEATURE_NAMES +
    _MOMENTUM_FEATURES +
    _ORIGIN_CAL_FEATURES +
    _SKU_STATIC_FEATURES
)
N_BASE_FEATURES = len(ALL_BASE_FEATURE_NAMES)

# Target-date calendar features (added per-row in the week-dataset builder)
_TARGET_CAL_FEATURES = [
    "target_dow", "target_day_of_month", "target_week_of_year",
    "target_month", "target_quarter",
    "target_is_weekend", "target_is_sunday",
    "forecast_day_index",   # 1-7 within the week
    "forecast_week_index",  # 1-8 global week
]

ALL_FEATURE_NAMES = ALL_BASE_FEATURE_NAMES + _TARGET_CAL_FEATURES


def compute_base_features_at_origins(
    origin_indices:  List[int],
    panels:          Dict[str, np.ndarray],   # pre-extracted .values arrays
    skus_sorted:     List[str],
    sku_features:    pd.DataFrame,            # indexed by ItemCode
    all_dates:       pd.DatetimeIndex,
) -> Tuple[np.ndarray, List[str], List[pd.Timestamp]]:
    """
    Vectorised computation of all base features for a batch of origin dates.

    For each origin date t (identified by its integer position in all_dates),
    features for ALL SKUs are computed simultaneously as 1-D numpy arrays,
    then stacked into a (n_origins × n_skus, N_BASE_FEATURES) float32 matrix.

    Leakage contract
    ----------------
    • Every lag, rolling mean, and price feature uses only indices ≤ t−1.
      The earliest index accessed per feature:
        lag_k        → t − k
        rmean_W_O    → t − O − W   (window ends at t − O − 1)
        price_lag_k  → t − k
        return_W     → t − W
      All are strictly before t → zero leakage for direct forecasting models.
    • SKU static features do not contain future information.

    Parameters
    ----------
    origin_indices : list of integer positions in all_dates
    panels         : dict of numpy arrays {qty, price, gross, return},
                     each shape (n_dates, n_skus), float32, SKUs sorted
    skus_sorted    : list of ItemCode strings matching panels column order
    sku_features   : extended SKU feature table indexed by ItemCode (sorted)
    all_dates      : DatetimeIndex of all training calendar dates

    Returns
    -------
    feature_matrix : np.ndarray shape (n_origins × n_skus, N_BASE_FEATURES), float32
    col_names      : list[str] matching ALL_BASE_FEATURE_NAMES
    origin_dates   : list[Timestamp] of length n_origins (for metadata)
    """
    qty_arr    = panels["qty"]
    price_arr  = panels["price"]
    gross_arr  = panels["gross"]
    return_arr = panels["return"]

    n_dates, n_skus = qty_arr.shape
    n_origins       = len(origin_indices)
    n_rows_total    = n_origins * n_skus

    # Pre-allocate the full output matrix with NaN (missing = NaN, handled by LGBM)
    out = np.full((n_rows_total, N_BASE_FEATURES), np.nan, dtype="float32")

    # SKU static values: shape (n_skus, n_static_features)
    # Reindex sku_features to skus_sorted to ensure column alignment.
    sf_aligned = sku_features.reindex(skus_sorted)

    # For demand_declining we need historical_mean_qty per SKU (varies by SKU)
    hist_mean = sf_aligned["historical_mean_qty"].values.astype("float32")

    origin_dates_out = []

    for batch_i, t_idx in enumerate(origin_indices):
        row_s = batch_i * n_skus
        row_e = row_s + n_skus
        feat_col = 0                    # column pointer within out[row_s:row_e, :]
        origin_date = all_dates[t_idx]
        origin_dates_out.append(origin_date)

        # ── 6A: Lag features ──────────────────────────────────────────────
        # LEAKAGE CHECK: lag_k uses index t_idx − k ≤ t_idx − 1. ✓
        for lag_k in CONFIG["LAGS"]:
            src = t_idx - lag_k
            if src >= 0:
                out[row_s:row_e, feat_col] = qty_arr[src, :]
            # else: remains NaN (insufficient history for this lag)
            feat_col += 1

        # ── 6B: Rolling mean features ────────────────────────────────────
        # rmean_{W}_{O}: indices [t_idx − O − W : t_idx − O], exclusive end.
        # LEAKAGE CHECK: window end = t_idx − O − 1 ≤ t_idx − 1. ✓
        for W, O in CONFIG["ROLLING_CONFIGS"]:
            start = t_idx - O - W
            end   = t_idx - O          # exclusive; last included = t_idx−O−1
            if start >= 0 and end > start:
                window = qty_arr[start:end, :]          # shape (W, n_skus)
                out[row_s:row_e, feat_col] = window.mean(axis=0)
            feat_col += 1

        # ── 6C: Rolling stat features ─────────────────────────────────────
        for stat, W, O in CONFIG["ROLLING_STAT_CONFIGS"]:
            start = t_idx - O - W
            end   = t_idx - O
            if start >= 0 and end > start:
                window = qty_arr[start:end, :]
                if stat == "std":
                    out[row_s:row_e, feat_col] = window.std(axis=0)
                elif stat == "max":
                    out[row_s:row_e, feat_col] = window.max(axis=0)
                elif stat == "zero_rate":
                    # Fraction of window days with net_qty <= 0
                    out[row_s:row_e, feat_col] = (window <= 0).mean(axis=0)
            feat_col += 1

        # ── 6D: Price features ────────────────────────────────────────────
        # LEAKAGE CHECK: price at t_idx − k < t_idx. ✓
        p_vals = {}
        for lag_k in CONFIG["PRICE_LAGS"]:
            src = t_idx - lag_k
            if src >= 0:
                p_vals[lag_k] = price_arr[src, :].copy()
                out[row_s:row_e, feat_col] = p_vals[lag_k]
            feat_col += 1
        # price_change = price_lag7 / (price_lag28 + 1e-3) − 1
        if len(CONFIG["PRICE_LAGS"]) >= 2:
            k0, k1 = CONFIG["PRICE_LAGS"][0], CONFIG["PRICE_LAGS"][1]  # 7, 28
            if k0 in p_vals and k1 in p_vals:
                out[row_s:row_e, feat_col] = (
                    p_vals[k0] / (p_vals[k1] + 1e-3) - 1.0
                )
        feat_col += 1   # for price_change column

        # ── 6E: Return fraction features ──────────────────────────────────
        # return_fraction_W = sum(returns[t-W:t]) / (sum(returns) + sum(gross) + ε)
        # LEAKAGE CHECK: window ends at t_idx − 1. ✓
        for W in CONFIG["RETURN_WINDOWS"]:
            if t_idx >= W:
                ret_sum   = return_arr[t_idx - W : t_idx, :].sum(axis=0)
                gross_sum = gross_arr[ t_idx - W : t_idx, :].sum(axis=0)
                out[row_s:row_e, feat_col] = (
                    ret_sum / (ret_sum + gross_sum + 1e-6)
                ).astype("float32")
            feat_col += 1

        # ── 6F: Momentum and regime features ─────────────────────────────
        # These are derived from rolling means computed above.
        # We re-read from the already-filled output columns instead of
        # recomputing, using the column index map.
        rmean_7_0_idx  = (len(CONFIG["LAGS"]) +
                          _RMEAN_FEATURE_NAMES.index("rmean_7_0"))
        rmean_28_0_idx = (len(CONFIG["LAGS"]) +
                          _RMEAN_FEATURE_NAMES.index("rmean_28_0"))

        r70  = out[row_s:row_e, rmean_7_0_idx]   # rmean_7_0 values (n_skus,)
        r280 = out[row_s:row_e, rmean_28_0_idx]  # rmean_28_0 values

        # mom_signal
        out[row_s:row_e, feat_col] = r70 / (r280 + 1e-3)
        feat_col += 1

        # demand_declining: rmean_28_0 < 0.6 × historical_mean_qty
        out[row_s:row_e, feat_col] = (
            r280 < CONFIG["DECLINING_THRESHOLD"] * hist_mean
        ).astype("float32")
        feat_col += 1

        # demand_accelerating: rmean_7_0 > 1.2 × rmean_28_0
        out[row_s:row_e, feat_col] = (
            r70 > CONFIG["ACCELERATING_THRESHOLD"] * r280
        ).astype("float32")
        feat_col += 1

        # last_28d_mean: alias of rmean_28_0 (explicit for interpretability)
        out[row_s:row_e, feat_col] = r280
        feat_col += 1

        # days_since_last_sale (look back up to 100 days)
        dsls = np.full(n_skus, 999.0, dtype="float32")
        for look_i in range(1, 101):
            if t_idx - look_i < 0:
                break
            mask = (qty_arr[t_idx - look_i, :] > 0) & (dsls == 999.0)
            dsls[mask] = look_i
        out[row_s:row_e, feat_col] = dsls
        feat_col += 1

        # recent_zero_streak — consecutive zero-sales days immediately before origin
        # Vectorised: still_counting stays True for each SKU until it hits a sale day.
        # LEAKAGE CHECK: only accesses qty_arr[t_idx - look_i] with look_i >= 1. ✓
        streak       = np.zeros(n_skus, dtype="float32")
        still_counting = np.ones(n_skus,  dtype=bool)
        for look_i in range(1, 101):
            if t_idx - look_i < 0:
                break
            has_sale = qty_arr[t_idx - look_i, :] > 0
            still_counting &= ~has_sale   # stop once a sale is found
            streak += still_counting.astype("float32")
        out[row_s:row_e, feat_col] = streak
        feat_col += 1

        # ── 6G: Origin calendar features ─────────────────────────────────
        out[row_s:row_e, feat_col] = origin_date.dayofweek          ; feat_col += 1
        out[row_s:row_e, feat_col] = origin_date.month              ; feat_col += 1
        out[row_s:row_e, feat_col] = origin_date.day                ; feat_col += 1
        out[row_s:row_e, feat_col] = origin_date.isocalendar()[1]   ; feat_col += 1

        # ── 6H: SKU static features (same for all origin dates) ──────────
        for sf_col in _SKU_STATIC_FEATURES:
            vals = sf_aligned[sf_col].values
            out[row_s:row_e, feat_col] = vals.astype("float32")
            feat_col += 1

        assert feat_col == N_BASE_FEATURES, (
            f"Column count mismatch at origin {origin_date}: "
            f"wrote {feat_col}, expected {N_BASE_FEATURES}"
        )

    return out, ALL_BASE_FEATURE_NAMES, origin_dates_out


# ==============================================================================
# 7 — ORIGIN DATE SAMPLING  (Stage 1E)
# ==============================================================================

def sample_origin_dates(
    all_dates:         pd.DatetimeIndex,
    train_end_idx:     int,              # inclusive upper bound in all_dates
    max_forecast_days: int,              # must leave room for targets (= week_end × 7)
    min_history_idx:   int,              # minimum date index with sufficient history
    n_sample:          int,
    seed:              int = CONFIG["SEED"],
) -> List[int]:
    """
    Stratified sample of origin date indices for training dataset construction.

    Stratification ensures every calendar month (across all years) is
    represented, so the model sees all seasonal contexts.

    Parameters
    ----------
    all_dates         : DatetimeIndex of all dates in training range
    train_end_idx     : last valid index such that origin + max_forecast_days ≤ train_end
    max_forecast_days : horizon end for the week being modelled (e.g., 56 for week 8)
    min_history_idx   : first valid origin (≥ MIN_HISTORY_DAYS from start)
    n_sample          : total number of origin dates to return
    seed              : random seed

    Returns
    -------
    Sorted list of valid integer indices into all_dates
    """
    rng = np.random.default_rng(seed)

    # Valid origin range: [min_history_idx, train_end_idx − max_forecast_days]
    last_valid_idx = train_end_idx - max_forecast_days
    if last_valid_idx < min_history_idx:
        raise ValueError(
            f"No valid origin dates: train_end_idx={train_end_idx}, "
            f"max_forecast_days={max_forecast_days}, "
            f"min_history_idx={min_history_idx}"
        )

    valid_indices = np.arange(min_history_idx, last_valid_idx + 1)

    if len(valid_indices) <= n_sample:
        return sorted(valid_indices.tolist())

    # Stratify by (year, month) bucket to ensure seasonal coverage
    valid_dates = all_dates[valid_indices]
    buckets     = valid_dates.to_period("M").astype(str)
    unique_buckets, bucket_ids = np.unique(buckets, return_inverse=True)

    n_buckets         = len(unique_buckets)
    per_bucket_sample = max(1, n_sample // n_buckets)
    sampled_indices   = []

    for b_id in range(n_buckets):
        in_bucket = valid_indices[bucket_ids == b_id]
        k         = min(per_bucket_sample, len(in_bucket))
        sampled_indices.extend(rng.choice(in_bucket, size=k, replace=False).tolist())

    # Top-up to exactly n_sample if buckets were uneven
    remaining = list(set(valid_indices.tolist()) - set(sampled_indices))
    shortfall  = n_sample - len(sampled_indices)
    if shortfall > 0 and remaining:
        extra = rng.choice(remaining, size=min(shortfall, len(remaining)), replace=False)
        sampled_indices.extend(extra.tolist())

    return sorted(sampled_indices[:n_sample])


# ==============================================================================
# 8 — WEEK-SPECIALIST TRAINING DATASET BUILDER  (Stage 1F)
# ==============================================================================

def build_week_training_dataset(
    week_idx:        int,                # 1-indexed week (1–8)
    base_feature_matrix: np.ndarray,    # (n_origins × n_skus, N_BASE_FEATURES)
    origin_dates:    List[pd.Timestamp],
    panels:          Dict[str, np.ndarray],
    skus_sorted:     List[str],
    all_dates:       pd.DatetimeIndex,
    sku_features:    pd.DataFrame,
    hard_zero_skus:  Set[str],
) -> pd.DataFrame:
    """
    Construct the training dataset for week-specialist model A_{week_idx}.

    For each (origin_date, SKU) pair in the base feature matrix, this function
    generates 7 training rows — one per forecast day within the week:

        target_date = origin_date + (week_idx − 1) × 7 + day_in_week
        target_qty  = net_qty at target_date (from qty panel)

    Rows where target_qty is NaN (target_date outside training range) are
    dropped.  Hard-zero SKUs are excluded (they do not need model training).

    Parameters
    ----------
    week_idx             : 1-indexed week number (1 = days 1–7, …, 8 = days 50–56)
    base_feature_matrix  : pre-computed base features from compute_base_features_at_origins
    origin_dates         : list of Timestamps matching the first dimension of base_feature_matrix
    panels               : dict of numpy arrays (qty, price, gross, return)
    skus_sorted          : ItemCode list matching column order of panels
    all_dates            : full calendar DatetimeIndex for integer lookup
    sku_features         : extended SKU feature table (indexed by ItemCode)
    hard_zero_skus       : set of SKUs excluded from training

    Returns
    -------
    pd.DataFrame with columns = ALL_FEATURE_NAMES + ['target_qty', 'ItemCode',
    '_origin_date', '_target_date'] and one row per (origin, SKU, day_in_week).
    """
    qty_arr = panels["qty"]
    n_skus  = len(skus_sorted)
    h_start = (week_idx - 1) * CONFIG["DAYS_PER_WEEK"]   # 0-indexed offset start
    days    = list(range(1, CONFIG["DAYS_PER_WEEK"] + 1)) # 1..7

    # Date → index map for fast lookup
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    # Index of hard-zero SKUs in skus_sorted (for efficient row removal)
    hz_mask = np.array(
        [sku in hard_zero_skus for sku in skus_sorted], dtype=bool
    )

    n_origins   = len(origin_dates)
    all_chunks  = []

    for origin_i, origin_date in enumerate(origin_dates):
        row_s = origin_i * n_skus
        row_e = row_s + n_skus
        base_rows = base_feature_matrix[row_s:row_e, :]  # (n_skus, N_BASE_FEATURES)

        for day_in_week in days:
            h = h_start + day_in_week                    # total horizon offset
            target_date = origin_date + pd.Timedelta(days=h)

            # Target qty: look up in qty_arr
            t_idx = date_to_idx.get(target_date, None)
            if t_idx is None:
                # target_date beyond training data — skip this day
                continue

            target_qty_all_skus = qty_arr[t_idx, :].copy()  # (n_skus,)

            # ── Target calendar features ──────────────────────────────────
            dow        = target_date.dayofweek
            is_sunday  = int(dow == 6)
            is_weekend = int(dow >= 5)
            quarter    = (target_date.month - 1) // 3 + 1

            # Build target-calendar row (same for all SKUs at this target_date)
            target_cal = np.array([
                dow,
                target_date.day,
                target_date.isocalendar()[1],
                target_date.month,
                quarter,
                is_weekend,
                is_sunday,
                day_in_week,             # forecast_day_index (1-7)
                week_idx,                # forecast_week_index (1-8)
            ], dtype="float32")          # (N_TARGET_CAL,)

            # Tile target_cal to (n_skus, N_TARGET_CAL)
            n_tc = len(_TARGET_CAL_FEATURES)
            target_cal_tiled = np.tile(target_cal, (n_skus, 1))  # (n_skus, n_tc)

            # Concatenate base + target-calendar + target_qty
            full_features = np.hstack([base_rows, target_cal_tiled])  # (n_skus, N_all)

            # Build DataFrame for this (origin, day) slice
            chunk = pd.DataFrame(
                full_features,
                columns=ALL_FEATURE_NAMES,
                dtype="float32",
            )
            chunk["target_qty"]    = target_qty_all_skus.astype("float32")
            chunk["ItemCode"]      = skus_sorted
            chunk["_origin_date"]  = origin_date
            chunk["_target_date"]  = target_date

            # Remove hard-zero SKUs — they need no training data
            chunk = chunk[~hz_mask].copy()

            all_chunks.append(chunk)

    if not all_chunks:
        raise RuntimeError(f"No valid training rows for week {week_idx}.")

    train_df = pd.concat(all_chunks, ignore_index=True)

    # ── Downcast to save memory ───────────────────────────────────────────
    for col in _SKU_STATIC_FEATURES:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype("float32")

    print(
        f"  Week {week_idx} training dataset: "
        f"{len(train_df):,} rows × {len(train_df.columns)} cols | "
        f"~{train_df.memory_usage(deep=True).sum() / 1e6:.0f} MB"
    )
    return train_df


# ==============================================================================
# 9 — (NOT IN USE) RECURSIVE MODEL B TRAINING DATASET BUILDER  (Stage 1G)    
# ==============================================================================

def build_model_b_training_dataset(
    panels:         Dict[str, np.ndarray],
    skus_sorted:    List[str],
    all_dates:      pd.DatetimeIndex,
    sku_features:   pd.DataFrame,
    hard_zero_skus: Set[str],
    train_end_idx:  int,
) -> pd.DataFrame:
    """
    Build the training dataset for the recursive one-step-ahead Model B.

    Model B predicts net_qty at t+1 from features available at t.
    Because the forecast horizon is only h=1, short lags (lag_1 through lag_7)
    are legal without leakage — all values are known at the origin t.

    Training range: last MODEL_B_LOOKBACK_DAYS from TRAIN_END.
    Each row: (SKU, origin_date t) → target = net_qty at t+1.

    Short lags added (not in non-recursive models' base feature set):
        lag_1, lag_2, lag_3, lag_4, lag_5, lag_6  (already lag_7 in base set)
    These are the primary signal for the recursive model and its main
    advantage over direct models at h=1.

    Parameters
    ----------
    panels, skus_sorted, all_dates, sku_features, hard_zero_skus,
    holiday_lookup : same as build_week_training_dataset
    train_end_idx  : integer index of TRAIN_END in all_dates

    Returns
    -------
    pd.DataFrame with columns: MODEL_B_FEATURES + ['target_qty', 'ItemCode',
    '_origin_date', '_target_date']
    """
    qty_arr = panels["qty"]
    n_skus  = len(skus_sorted)
    hz_mask = np.array(
        [sku in hard_zero_skus for sku in skus_sorted], dtype=bool
    )

    # Lookback window: up to MODEL_B_LOOKBACK_DAYS before TRAIN_END
    lookback_start_idx = max(
        0, train_end_idx - CONFIG["MODEL_B_LOOKBACK_DAYS"]
    )
    # Minimum index: need lag_28 at minimum (to match global feature set)
    min_valid_idx = lookback_start_idx + 28

    # Sample origin dates (all valid dates in the lookback window)
    # For Model B we use all dates (no subsampling) since the window is small
    origin_indices = list(range(min_valid_idx, train_end_idx))  # exclude train_end itself

    short_lags = [l for l in CONFIG["MODEL_B_SHORT_LAGS"] if l not in CONFIG["LAGS"]]

    short_lag_names = [f"lag_{k}" for k in short_lags]
    MODEL_B_EXTRA_FEATURES = short_lag_names  # extra lags not in base set

    all_chunks = []
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    for t_idx in origin_indices:
        origin_date = all_dates[t_idx]
        target_date = origin_date + pd.Timedelta(days=1)

        t_target_idx = date_to_idx.get(target_date, None)
        if t_target_idx is None:
            continue

        target_qty_all = qty_arr[t_target_idx, :].copy()

        # ── Base features at t_idx ────────────────────────────────────────
        # Re-use lags from CONFIG["LAGS"] + add Model-B-specific short lags
        row_dict = {}

        # Standard lags
        for k in CONFIG["LAGS"]:
            src = t_idx - k
            row_dict[f"lag_{k}"] = qty_arr[src, :].copy() if src >= 0 else np.full(n_skus, np.nan, dtype="float32")

        # Model-B extra short lags (leak-safe for h=1)
        for k in short_lags:
            src = t_idx - k
            row_dict[f"lag_{k}"] = qty_arr[src, :].copy() if src >= 0 else np.full(n_skus, np.nan, dtype="float32")

        # rmean_7_0 and rmean_28_0 (most important for recursive model)
        if t_idx >= 7:
            row_dict["rmean_7_0"]  = qty_arr[t_idx - 7  : t_idx, :].mean(axis=0)
        if t_idx >= 28:
            row_dict["rmean_28_0"] = qty_arr[t_idx - 28 : t_idx, :].mean(axis=0)

        # ── Target calendar features ──────────────────────────────────────
        dow        = target_date.dayofweek
        row_dict["target_dow"]        = np.full(n_skus, dow,                    dtype="float32")
        row_dict["target_month"]      = np.full(n_skus, target_date.month,      dtype="float32")
        row_dict["target_is_sunday"]  = np.full(n_skus, int(dow == 6),          dtype="float32")
        row_dict["target_is_weekend"] = np.full(n_skus, int(dow >= 5),          dtype="float32")

        # ── SKU static ────────────────────────────────────────────────────
        sf_aligned = sku_features.reindex(skus_sorted)
        for sf_col in ["profit_weight", "profit_tier_enc", "historical_mean_qty",
                        "zero_rate_post2022", "last_28d_mean"]:
            row_dict[sf_col] = sf_aligned[sf_col].values.astype("float32")

        # ── Assemble chunk ────────────────────────────────────────────────
        chunk = pd.DataFrame(row_dict)
        chunk.index = skus_sorted
        chunk.index.name = "ItemCode"
        chunk = chunk.reset_index()
        chunk["target_qty"]   = target_qty_all
        chunk["_origin_date"] = origin_date
        chunk["_target_date"] = target_date
        chunk = chunk[~hz_mask].copy()

        all_chunks.append(chunk)

        if len(all_chunks) % 200 == 0:
            print(f"    Model B: processed {len(all_chunks)} origin dates …", end="\r")

    train_b = pd.concat(all_chunks, ignore_index=True)
    print(
        f"\n  Model B training dataset: "
        f"{len(train_b):,} rows × {len(train_b.columns)} cols | "
        f"~{train_b.memory_usage(deep=True).sum() / 1e6:.0f} MB"
    )
    return train_b


# ==============================================================================
# 10 — INFERENCE DATASET BUILDER  (Stage 1H)
# ==============================================================================

def build_inference_dataset(
    panels:        Dict[str, np.ndarray],
    skus_sorted:   List[str],
    all_dates:     pd.DatetimeIndex,
    sku_features:  pd.DataFrame,
    origin_date:   pd.Timestamp,
) -> pd.DataFrame:
    """
    Build the feature dataset for final submission inference.

    The origin date is CONFIG["TRAIN_END"] (2025-09-05).
    Target dates span the full 56-day horizon:
        Public  (weeks 1–4): 2025-09-06 → 2025-10-03
        Private (weeks 5–8): 2025-10-04 → 2025-10-31

    One row per (SKU, target_day_index) — same structure as training data
    but without 'target_qty' (unknown).

    Returns
    -------
    pd.DataFrame with columns = ALL_FEATURE_NAMES + ['ItemCode', 'week_idx',
    'day_in_week', 'target_date', 'is_hard_zero']
    """
    qty_arr    = panels["qty"]
    price_arr  = panels["price"]
    gross_arr  = panels["gross"]
    return_arr = panels["return"]
    n_skus     = len(skus_sorted)

    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    origin_idx  = date_to_idx[origin_date]

    # ── Compute base features at origin (single call) ─────────────────────
    panels_np = {
        "qty":    qty_arr,
        "price":  price_arr,
        "gross":  gross_arr,
        "return": return_arr,
    }
    base_matrix, _, _ = compute_base_features_at_origins(
        origin_indices  = [origin_idx],
        panels          = panels_np,
        skus_sorted     = skus_sorted,
        sku_features    = sku_features,
        all_dates       = all_dates,
    )
    base_rows = base_matrix        # shape (n_skus, N_BASE_FEATURES)

    # ── Build one row per (SKU × horizon day) ─────────────────────────────
    all_rows = []
    for week_idx in range(1, CONFIG["N_WEEKS"] + 1):
        for day_in_week in range(1, CONFIG["DAYS_PER_WEEK"] + 1):
            h           = (week_idx - 1) * CONFIG["DAYS_PER_WEEK"] + day_in_week
            target_date = origin_date + pd.Timedelta(days=h)

            dow        = target_date.dayofweek
            is_sunday  = int(dow == 6)
            quarter    = (target_date.month - 1) // 3 + 1

            target_cal = np.array([
                dow,
                target_date.day,
                target_date.isocalendar()[1],
                target_date.month,
                quarter,
                int(dow >= 5),
                is_sunday,
                day_in_week,
                week_idx,
            ], dtype="float32")

            target_cal_tiled = np.tile(target_cal, (n_skus, 1))
            full_feats       = np.hstack([base_rows, target_cal_tiled])

            chunk = pd.DataFrame(full_feats, columns=ALL_FEATURE_NAMES, dtype="float32")
            chunk["ItemCode"]    = skus_sorted
            chunk["week_idx"]    = np.int8(week_idx)
            chunk["day_in_week"] = np.int8(day_in_week)
            chunk["target_date"] = target_date
            chunk["horizon_day"] = np.int8(h)         # 1–56
            all_rows.append(chunk)

    inference_ds = pd.concat(all_rows, ignore_index=True)
    print(
        f"  Inference dataset: {len(inference_ds):,} rows "
        f"({len(skus_sorted):,} SKUs × 56 days) | "
        f"~{inference_ds.memory_usage(deep=True).sum() / 1e6:.0f} MB"
    )
    return inference_ds


# ==============================================================================
# 11 — LEAKAGE AUDIT  (Stage 1I)
# ==============================================================================

def leakage_audit(
    df:        pd.DataFrame,
    week_idx:  Optional[int] = None,
    label:     str = "",
) -> None:
    """
    Automated leakage detection for a training dataset.

    Checks performed
    ----------------
    1. target_qty not present in any feature column (name match).
    2. Feature column names do not contain 'target_qty'.
    3. '_target_date' > '_origin_date' for every row.
    4. NaN rate per feature — high NaN rates flag potential data issues.
    5. Target variable distribution: mean, std, max, % negative.

    Parameters
    ----------
    df       : training DataFrame from build_week_training_dataset
    week_idx : week index label for display (optional)
    label    : additional label string (optional)
    """
    tag = f"Week {week_idx}" if week_idx is not None else label
    print(f"\n  ── Leakage Audit: {tag} ──")

    # Check 1: target column not in feature names
    feature_cols = [c for c in ALL_FEATURE_NAMES if c in df.columns]
    if "target_qty" in feature_cols:
        print("  !! LEAKAGE DETECTED: 'target_qty' appears in feature columns!")
    else:
        print("  ✓ 'target_qty' not in feature column list")

    # Check 2: temporal ordering
    if "_origin_date" in df.columns and "_target_date" in df.columns:
        bad_rows = (df["_target_date"] <= df["_origin_date"]).sum()
        if bad_rows > 0:
            print(f"  !! LEAKAGE: {bad_rows} rows where target_date ≤ origin_date")
        else:
            print("  ✓ All target_dates > origin_dates")

    # Check 3: NaN rates
    nan_rates = df[feature_cols].isna().mean().sort_values(ascending=False)
    high_nan  = nan_rates[nan_rates > 0.30]
    if len(high_nan) > 0:
        print(f"  ⚠ Features with >30% NaN (expected for long lags on short-history SKUs):")
        for col, rate in high_nan.head(8).items():
            print(f"      {col}: {rate*100:.1f}%")
    else:
        print("  ✓ No features with >30% NaN")

    # Check 4: target distribution
    tgt = df["target_qty"]
    print(f"  Target stats: mean={tgt.mean():.2f}, std={tgt.std():.2f}, "
          f"max={tgt.max():.0f}, "
          f"% negative={(tgt < 0).mean()*100:.2f}%")


# ==============================================================================
# 12 — FEATURE DISTRIBUTION PLOTS  (Stage 1J)
# ==============================================================================

def plot_feature_diagnostics(
    df:        pd.DataFrame,
    week_idx:  int,
    sku_features: pd.DataFrame,
) -> None:
    """
    Produce diagnostic plots for the training dataset of a week-specialist model.

    Plots
    -----
    F1  Distribution of key lag features (lag_1, lag_7, lag_28, lag_56)
    F2  Distribution of rolling means (rmean_7_0, rmean_28_0, rmean_7_28, rmean_28_28)
    F3  Target quantity distribution (all SKUs, then top A-tier only)
    F4  Feature NaN-rate bar chart
    F5  Target mean by day-of-week (from forecast_day_index + target_dow)
    """
    set_plot_style()
    print(f"\n  Plotting feature diagnostics for Week {week_idx} …")

    # ── F1: Lag feature distributions ────────────────────────────────────
    lag_cols = [c for c in ["lag_1", "lag_7", "lag_28", "lag_56"] if c in df.columns]
    if lag_cols:
        fig, axes = plt.subplots(1, len(lag_cols), figsize=(5 * len(lag_cols), 4))
        if len(lag_cols) == 1:
            axes = [axes]
        fig.suptitle(f"Fig F1-W{week_idx} — Lag Feature Distributions (clipped at 99th pct)",
                     fontweight="bold")
        for ax, col in zip(axes, lag_cols):
            vals = df[col].dropna()
            clip = vals.quantile(0.99)
            ax.hist(vals.clip(upper=clip), bins=60, color="steelblue",
                    edgecolor="none", alpha=0.8)
            ax.set_title(col)
            ax.set_xlabel("Net qty")
        plt.tight_layout()
        save_fig(fig, f"F1_W{week_idx}_lag_distributions")
        plt.show()

    # ── F2: Rolling mean distributions ───────────────────────────────────
    roll_cols = [c for c in ["rmean_7_0", "rmean_28_0", "rmean_7_28", "rmean_28_28"] if c in df.columns]
    if roll_cols:
        fig, axes = plt.subplots(1, len(roll_cols), figsize=(5 * len(roll_cols), 4))
        if len(roll_cols) == 1:
            axes = [axes]
        fig.suptitle(f"Fig F2-W{week_idx} — Rolling Mean Feature Distributions",
                     fontweight="bold")
        for ax, col in zip(axes, roll_cols):
            vals = df[col].dropna()
            clip = vals.quantile(0.99)
            ax.hist(vals.clip(upper=clip), bins=60, color="darkorange",
                    edgecolor="none", alpha=0.8)
            ax.set_title(col)
            ax.set_xlabel("Net qty")
        plt.tight_layout()
        save_fig(fig, f"F2_W{week_idx}_rmean_distributions")
        plt.show()

    # ── F3: Target quantity distribution ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f"Fig F3-W{week_idx} — Target Quantity Distribution", fontweight="bold")

    tgt = df["target_qty"].clip(upper=df["target_qty"].quantile(0.99))
    axes[0].hist(tgt, bins=80, color="steelblue", edgecolor="none", alpha=0.8, log=True)
    axes[0].set_title("All SKUs (log Y)")
    axes[0].set_xlabel("target_qty")

    # A-tier only
    if "profit_tier_enc" in df.columns:
        tgt_a = df[df["profit_tier_enc"] == 0]["target_qty"].clip(
            upper=df["target_qty"].quantile(0.995)
        )
        axes[1].hist(tgt_a, bins=60, color="navy", edgecolor="none", alpha=0.8)
        axes[1].set_title("A-Tier SKUs only")
        axes[1].set_xlabel("target_qty")
    plt.tight_layout()
    save_fig(fig, f"F3_W{week_idx}_target_distribution")
    plt.show()

    # ── F4: NaN rate bar chart ────────────────────────────────────────────
    feat_cols = [c for c in ALL_FEATURE_NAMES if c in df.columns]
    nan_rates = df[feat_cols].isna().mean().sort_values(ascending=False).head(20)
    if nan_rates.max() > 0:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(range(len(nan_rates)), nan_rates.values, color="tomato",
               edgecolor="none", alpha=0.8)
        ax.set_xticks(range(len(nan_rates)))
        ax.set_xticklabels(nan_rates.index, rotation=45, ha="right", fontsize=9)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_title(f"Fig F4-W{week_idx} — Top-20 NaN Rates by Feature")
        plt.tight_layout()
        save_fig(fig, f"F4_W{week_idx}_nan_rates")
        plt.show()

    # ── F5: Mean target by target day-of-week ────────────────────────────
    if "target_dow" in df.columns:
        dow_mean = df.groupby("target_dow")["target_qty"].mean()
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(dow_mean.index, dow_mean.values, color="steelblue",
               edgecolor="none", alpha=0.8)
        ax.set_xticks(range(7))
        ax.set_xticklabels(day_names)
        ax.set_title(f"Fig F5-W{week_idx} — Mean Target Qty by Day-of-Week")
        ax.set_ylabel("Mean target_qty")
        plt.tight_layout()
        save_fig(fig, f"F5_W{week_idx}_target_by_dow")
        plt.show()


# ==============================================================================
# 13 — MAIN STAGE 1 EXECUTION
# ==============================================================================

def run_stage1(
    panel:        pd.DataFrame,
    sku_stats:    pd.DataFrame,
) -> Dict:
    """
    Execute the full Stage 1 pipeline in order.

    Steps
    -----
    1A  Extend SKU static features
    1B  Build wide panels
    1C  Identify hard-zero SKUs
    1D  Build holiday lookup
    1E  Build panel numpy arrays for fast indexing
    1F  Compute full date range and CV indices
    1G  For each week (1–8): sample origins → compute base features →
        build week training dataset → leakage audit → plot → save
    1H  Build Model B training dataset
    1I  Build inference dataset (origin = TRAIN_END)
    1J  Save all artifacts

    Parameters
    ----------
    panel      : Stage 0 daily panel
    sku_stats  : Stage 0 SKU statistics

    Returns
    -------
    dict with keys:
        sku_features, panel_wide, hard_zero_skus,
        train_datasets, inference_ds
    """
    print("\n" + "=" * 60)
    print(" HBAAC 2026 — Stage 1: Feature Engineering Pipeline")
    print("=" * 60)

    # ── 1A: Extended SKU features ────────────────────────────────────────
    print("\n--- 1A: Extending SKU static features ---")
    sku_features = extend_sku_static_features(panel, sku_stats)
    sku_features.to_csv(_OUTPUT_DIR / "sku_features.csv")

    # ── 1B: Wide panels ──────────────────────────────────────────────────
    print("\n--- 1B: Building wide panels ---")
    # Full calendar range covering training only
    # (inference uses the same panels with origin at TRAIN_END)
    train_date_range = pd.date_range(
        CONFIG["TRAIN_START"], CONFIG["TRAIN_END"], freq="D"
    )
    panel_wide_dfs = build_panel_wide(panel, train_date_range)

    # ── 1C: Hard-zero SKUs ───────────────────────────────────────────────
    print("\n--- 1C: Identifying hard-zero SKUs ---")
    hard_zero_skus = identify_hard_zero_skus(sku_features)

    # ── 1E: Convert wide DataFrames to numpy for fast indexing ────────────
    print("\n--- 1E: Extracting numpy arrays from wide panels ---")
    skus_sorted  = panel_wide_dfs["qty"].columns.tolist()  # consistent SKU order
    all_dates    = train_date_range

    panels_np = {
        "qty":    panel_wide_dfs["qty"].values.astype("float32"),
        "price":  panel_wide_dfs["price"].values.astype("float32"),
        "gross":  panel_wide_dfs["gross"].values.astype("float32"),
        "return": panel_wide_dfs["return"].values.astype("float32"),
    }
    n_all_dates = len(all_dates)

    # Verify sku_features is aligned to skus_sorted
    sku_features = sku_features.reindex(skus_sorted)

    # ── 1F: Date index helpers ────────────────────────────────────────────
    print("\n--- 1F: Computing date indices ---")
    date_to_idx  = {d: i for i, d in enumerate(all_dates)}
    cutoff_idx   = date_to_idx.get(_CUTOFF, 0)
    train_end_idx= date_to_idx[_TRAIN_END]

    # Minimum history index: at least MIN_HISTORY_DAYS after cutoff so that
    # the maximum lag (56 days) and rolling features have sufficient backing data.
    min_hist_idx = cutoff_idx + CONFIG["MIN_HISTORY_DAYS"]

    print(f"  Training date range : {all_dates[0].date()} → {all_dates[-1].date()}")
    print(f"  Cutoff index        : {cutoff_idx} ({all_dates[cutoff_idx].date()})")
    print(f"  Min-history index   : {min_hist_idx} ({all_dates[min_hist_idx].date()})")
    print(f"  Train-end index     : {train_end_idx} ({all_dates[train_end_idx].date()})")

    # ── 1G: Week-specialist training datasets ─────────────────────────────
    print("\n--- 1G: Building week-specialist training datasets ---")
    train_datasets = {}

    for week_idx in range(1, CONFIG["N_WEEKS"] + 1):
        print(f"\n  ── Week {week_idx} / {CONFIG['N_WEEKS']} ──")

        # For week k, the furthest target is train_end - k*7 days from origin
        max_forecast = week_idx * CONFIG["DAYS_PER_WEEK"]

        # Sample origin dates
        try:
            origin_indices = sample_origin_dates(
                all_dates         = all_dates,
                train_end_idx     = train_end_idx,
                max_forecast_days = max_forecast,
                min_history_idx   = min_hist_idx,
                n_sample          = CONFIG["N_ORIGIN_SAMPLE"],
                seed              = CONFIG["SEED"] + week_idx,
            )
        except ValueError as e:
            print(f"  [WARNING] Skipping week {week_idx}: {e}")
            continue

        print(f"  Origin dates sampled: {len(origin_indices)} "
              f"(range: {all_dates[origin_indices[0]].date()} → "
              f"{all_dates[origin_indices[-1]].date()})")

        # Compute base features in batches
        print(f"  Computing base features (batch size={CONFIG['ORIGIN_BATCH_SIZE']}) …")
        batch_size    = CONFIG["ORIGIN_BATCH_SIZE"]
        batches       = [origin_indices[i:i + batch_size]
                         for i in range(0, len(origin_indices), batch_size)]
        base_chunks   = []
        origin_dates_all = []

        for b_i, batch in enumerate(batches):
            chunk_matrix, _, batch_dates = compute_base_features_at_origins(
                origin_indices = batch,
                panels         = panels_np,
                skus_sorted    = skus_sorted,
                sku_features   = sku_features,
                all_dates      = all_dates,
            )
            base_chunks.append(chunk_matrix)
            origin_dates_all.extend(batch_dates)
            print(f"    Batch {b_i+1}/{len(batches)} done.", end="\r")

        print()
        base_matrix = np.vstack(base_chunks)   # (n_origins × n_skus, N_BASE_FEATURES)
        del base_chunks
        gc.collect()

        # Build week training dataset
        week_df = build_week_training_dataset(
            week_idx            = week_idx,
            base_feature_matrix = base_matrix,
            origin_dates        = origin_dates_all,
            panels              = panels_np,
            skus_sorted         = skus_sorted,
            all_dates           = all_dates,
            sku_features        = sku_features,
            hard_zero_skus      = hard_zero_skus,
        )

        del base_matrix
        gc.collect()

        # Leakage audit
        leakage_audit(week_df, week_idx=week_idx)

        # Feature distribution plots (only for week 1 to save runtime;
        # other weeks share the same base feature distributions)
        if week_idx == 1:
            plot_feature_diagnostics(week_df, week_idx=1, sku_features=sku_features)

        # Save to parquet (preserves float32, efficient I/O)
        out_path = _OUTPUT_DIR / f"train_week_{week_idx}.parquet"
        week_df.to_parquet(out_path, index=False)
        print(f"  Saved → {out_path}")

        train_datasets[week_idx] = week_df
        del week_df
        gc.collect()



    # ── 1I: Inference dataset ─────────────────────────────────────────────
    print("\n--- 1I: Building inference dataset (origin = TRAIN_END) ---")
    inference_ds = build_inference_dataset(
        panels         = panels_np,
        skus_sorted    = skus_sorted,
        all_dates      = all_dates,
        sku_features   = sku_features,
        origin_date    = _ORIGIN_FINAL,
    )
    inf_path = _OUTPUT_DIR / "inference_dataset.parquet"
    inference_ds.to_parquet(inf_path, index=False)
    print(f"  Saved → {inf_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" Stage 1 Complete — Summary")
    print("=" * 60)
    print(f"  SKUs in feature table     : {len(sku_features):,}")
    print(f"  Hard-zero SKUs            : {len(hard_zero_skus):,}")
    print(f"  Base features per row     : {N_BASE_FEATURES}")
    print(f"  Total features (w/ target-cal) : {len(ALL_FEATURE_NAMES)}")
    print(f"  Week datasets built       : {len(train_datasets)}")
    print(f"  Inference rows            : {len(inference_ds):,}")
    print(f"  All artifacts saved to    : {_OUTPUT_DIR}")
    print("=" * 60)

    return {
        "sku_features":    sku_features,
        "panel_wide":      panel_wide_dfs,
        "hard_zero_skus":  hard_zero_skus,
        "train_datasets":  train_datasets,
        "inference_ds":    inference_ds,
    }


# ==============================================================================
# ENTRY POINT
# ==============================================================================
# In the Kaggle notebook, call:
#
#   if __name__ == "__main__":
#       artifacts = run_eda()
#       panel = artifacts["panel"]
#       sku_stats = artifacts["sku_stats"]
#       stage1_artifacts = run_stage1(panel, sku_stats)
#
# Then unpack for convenience:
#   sku_features    = stage1_artifacts["sku_features"]
#   panel_wide      = stage1_artifacts["panel_wide"]
#   hard_zero_skus  = stage1_artifacts["hard_zero_skus"]
#   train_datasets  = stage1_artifacts["train_datasets"]
#   inference_ds    = stage1_artifacts["inference_ds"]


# entry point feature engineering
if __name__ == "__main__":
    artifacts = run_eda()
    panel = artifacts["panel"]
    sku_stats = artifacts["sku_stats"]
    stage1_artifacts = run_stage1(panel, sku_stats)
