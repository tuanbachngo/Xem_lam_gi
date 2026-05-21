from step0_eda import *
from step1_features import *

"""
HBAAC 2026 — Stages 2–6: Training, CV, Ensemble, Post-processing, Submission
=============================================================================
Continuation of the same Kaggle notebook after Stage 1.

REQUIRED variables already in notebook scope (from Stages 0 and 1)
-------------------------------------------------------------------
  CONFIG                 dict
  panel                  pd.DataFrame   daily panel (Date, ItemCode, net_qty, …)
  sku_stats              pd.DataFrame   Stage 0 SKU statistics
  folds                  list[dict]     Stage 0 CV fold boundaries
  sub_raw                pd.DataFrame   sample_submission.csv (loaded in Stage 0)

  stage1_artifacts       dict  with keys:
      sku_features       pd.DataFrame (indexed by ItemCode)
      panel_wide         dict  {qty, price, gross, return} DataFrames
      hard_zero_skus     set[str]
      train_datasets     dict  {1..8 : pd.DataFrame}
      inference_ds       pd.DataFrame

  ALL_FEATURE_NAMES      list[str]   (Stage 1 module-level constant)
  _TARGET_CAL_FEATURES   list[str]
  _OUTPUT_DIR            Path
  _TRAIN_END             pd.Timestamp
  _CUTOFF                pd.Timestamp

Execution
---------
  artifacts_26 = run_pipeline(stage1_artifacts, panel, sub_raw)
"""

# ==============================================================================
# 0 — IMPORTS
# ==============================================================================
import gc
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

# Global variables and fallbacks to avoid NameError
naive_results = None
_TRAIN_END = CONFIG["TRAIN_END"]
_CUTOFF = CONFIG["EARLY_DATA_CUTOFF"]
_OUTPUT_DIR = CONFIG["OUTPUT_DIR"]

print("lightgbm version :", lgb.__version__)

# ==============================================================================
# 1 — STAGE 2 CONFIGURATION
# ==============================================================================
CONFIG.update(
    {
        # Update OUPUT DIR
        "OUTPUT_DIR": Path("output/submission_outputs"),
        # ── LightGBM base parameters shared by all tree models ──────────────────
        "LGBM_BASE": {
            "n_jobs": -1,
            "device": "gpu",
            "random_state": CONFIG["SEED"],
            "verbose": -1,
            "min_child_samples": 20,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
        },
        # ── Model A (Non-recursive, Tweedie) ─────────────────────────────────────
        "LGBM_A": {
            "objective": "tweedie",
            "tweedie_variance_power": 1.5,
            "metric": "rmse",
            "num_leaves": 64,
            "learning_rate": 0.05,
            "n_estimators": 750,
        },
        # # ── Model C (Non-recursive, Poisson) ─────────────────────────────────────
        # "LGBM_C": {
        #     "objective": "poisson",
        #     "metric": "rmse",
        #     "num_leaves": 64,
        #     "learning_rate": 0.05,
        #     "n_estimators": 500,
        # },
        # # ── Model B (Recursive, Tweedie) ─────────────────────────────────────────
        # "LGBM_B": {
        #     "objective": "tweedie",
        #     "tweedie_variance_power": 1.5,
        #     "metric": "rmse",
        #     "num_leaves": 64,
        #     "learning_rate": 0.05,
        #     "n_estimators": 700,  # reduced vs A-models for speed
        # },
        # ── CV training: aligned with final models for honest evaluation ─────────
        "CV_N_ESTIMATORS": 300,
        "EARLY_STOPPING_ROUNDS": 50,
        # ── Ensemble weights: {week_idx: {model_key: base_weight}} ───────────────
        # Keys: 'A'=Tweedie NR, 'D'=Naive, 'E'=ETS
        # These are BASE weights applied to A-tier Dense SKUs.
        # Lower-tier SKUs follow the TIER_BLEND_WEIGHTS override below.
        "WEEK_WEIGHTS": {
            1: {"A": 0.96, "E": 0.04},  # Tier A Dense w14
            2: {"A": 0.96, "E": 0.04},
            3: {"A": 0.96, "E": 0.04},
            4: {"A": 0.96, "E": 0.04},
            # Tier A Dense w58 (Blended Naive for safety)
            5: {"A": 0.30, "E": 0.50, "D": 0.20},
            6: {"A": 0.30, "E": 0.50, "D": 0.20},
            7: {"A": 0.30, "E": 0.50, "D": 0.20},
            8: {"A": 0.30, "E": 0.50, "D": 0.20},
        },
        # Tier-level blend overrides  (applied instead of WEEK_WEIGHTS for non-A tiers)
        # Structure: {profit_tier_enc: {density_enc: {model_key: weight}}}
        # profit_tier_enc: A=0, B=1, C=2, D=3
        # demand_density_enc: Dense=0, Intermittent=1, Sparse=2
        "TIER_BLEND": {
            # B-tier Dense: heavily rely on A; mix in some Naive for long-term stability
            (1, 0): {"w14": {"A": 0.90, "D": 0.10}, "w58": {"A": 0.70, "D": 0.30}},
            # B-tier Intermittent: A + D (Model A handles intermittency well if given days_since_last_sale)
            (1, 1): {"w14": {"A": 0.70, "D": 0.30}, "w58": {"A": 0.60, "D": 0.40}},
            # B-tier Sparse: A + D
            (1, 2): {"w14": {"A": 0.60, "D": 0.40}, "w58": {"A": 0.50, "D": 0.50}},
            # C-tier any: A + D
            (2, 0): {"w14": {"A": 0.60, "D": 0.40}, "w58": {"A": 0.50, "D": 0.50}},
            (2, 1): {"w14": {"A": 0.60, "D": 0.40}, "w58": {"A": 0.50, "D": 0.50}},
            (2, 2): {"w14": {"A": 0.50, "D": 0.50}, "w58": {"A": 0.40, "D": 0.60}},
            # D-tier any: pure Naive
            (3, 0): {"w14": {"D": 1.0}, "w58": {"D": 1.0}},
            (3, 1): {"w14": {"D": 1.0}, "w58": {"D": 1.0}},
            (3, 2): {"w14": {"D": 1.0}, "w58": {"D": 1.0}},
        },
    }
)

CONFIG["OUTPUT_DIR"].mkdir(parents=True, exist_ok=True)

_OUTPUT_DIR = CONFIG["OUTPUT_DIR"]

# ── Categorical feature names for LightGBM ──────────────────────────────────
LGBM_CAT_FEATURES = [
    "profit_tier_enc",
    "demand_density_enc",
    "target_dow",
    "target_month",
    "target_quarter",
    "origin_dow",
    "origin_month",
]

# Metadata columns not used as features
_META_COLS = {"target_qty", "ItemCode", "_origin_date", "_target_date"}


# ==============================================================================
# 2 — SETUP: UNPACK STAGE 1 ARTIFACTS
# ==============================================================================


def setup_stage2(stage1_artifacts: Dict) -> Dict:
    """
    Unpack Stage 1 artifacts and reconstruct numpy arrays needed for Stages 2-6.

    Also recomputes CV folds using the updated EARLY_DATA_CUTOFF (2022-01-01)
    so the anchor fold is grounded in clean data.

    Returns
    -------
    ctx : dict with keys:
        sku_features, panel_wide, hard_zero_skus,
        train_datasets, inference_ds,
        panels_np, skus_sorted, all_dates, date_to_idx,
        origin_idx, week_feature_cols,
        folds_updated, weights_series
    """
    print("\n--- Stage 2 Setup ---")

    sku_features = stage1_artifacts["sku_features"]
    panel_wide = stage1_artifacts["panel_wide"]
    hard_zero_skus = stage1_artifacts["hard_zero_skus"]
    train_datasets = stage1_artifacts["train_datasets"]
    inference_ds = stage1_artifacts["inference_ds"]

    # Reconstruct numpy arrays from wide DataFrames
    skus_sorted = panel_wide["qty"].columns.tolist()
    all_dates = pd.date_range(CONFIG["TRAIN_START"], CONFIG["TRAIN_END"], freq="D")
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    origin_idx = date_to_idx[_TRAIN_END]

    panels_np = {k: v.values.astype("float32") for k, v in panel_wide.items()}
    print(
        f"  panels_np shapes: qty={panels_np['qty'].shape}, "
        f"price={panels_np['price'].shape}"
    )

    # Feature column lists
    sample_ds = train_datasets[min(train_datasets.keys())]
    week_feature_cols = [
        c for c in sample_ds.columns if c not in _META_COLS and c != "target_qty"
    ]

    # SKU profit weights as a Series (ItemCode → weight)
    weights_series = sku_features["profit_weight"].reindex(skus_sorted).fillna(0.0)

    # Recompute CV folds anchored to 2022-01-01
    folds_updated = compute_cv_fold_dates(
        train_start=_CUTOFF,  # 2022-01-01
        train_end=_TRAIN_END,  # 2025-09-05
        horizon=CONFIG["HORIZON_DAYS"],
        n_folds=CONFIG["N_FOLDS"],
        anchor_gap_days=365,
    )
    print(f"\n  Updated CV folds (anchored to {_CUTOFF.date()}):")
    print(
        f"  {'Fold':>5}  {'Train start':>12}  {'Train end':>12}  "
        f"{'Val start':>12}  {'Val end':>12}"
    )
    print("  " + "-" * 62)
    for f in folds_updated:
        print(
            f"  {f['fold_id']:>5}  {str(f['train_start'].date()):>12}  "
            f"{str(f['train_end'].date()):>12}  "
            f"{str(f['val_start'].date()):>12}  "
            f"{str(f['val_end'].date()):>12}"
        )

    print(f"\n  Total SKUs           : {len(skus_sorted):,}")
    print(f"  Hard-zero SKUs       : {len(hard_zero_skus):,}")
    print(f"  Week feature cols    : {len(week_feature_cols)}")

    return {
        "sku_features": sku_features,
        "panel_wide": panel_wide,
        "hard_zero_skus": hard_zero_skus,
        "train_datasets": train_datasets,
        "inference_ds": inference_ds,
        "panels_np": panels_np,
        "skus_sorted": skus_sorted,
        "all_dates": all_dates,
        "date_to_idx": date_to_idx,
        "origin_idx": origin_idx,
        "week_feature_cols": week_feature_cols,
        "folds_updated": folds_updated,
        "weights_series": weights_series,
    }


# ==============================================================================
# 3 — LGBM TRAINING UTILITY  (Models A and C)
# ==============================================================================


def _make_lgbm_params(base_key: str) -> Dict:
    """Merge LGBM_BASE with model-specific params and return a clean dict."""
    params = {**CONFIG["LGBM_BASE"], **CONFIG[base_key]}
    # Remove None values (e.g. tweedie_variance_power for Poisson)
    return {k: v for k, v in params.items() if v is not None}


def train_lgbm(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    param_key: str,  # 'LGBM_A', 'LGBM_B', or 'LGBM_C'
    label: str = "",
    n_estimators_override: Optional[int] = None,
    eval_df: Optional[pd.DataFrame] = None,
    param_overrides: Optional[Dict] = None,
) -> lgb.LGBMRegressor:
    """
    Train a LightGBM model on the provided training DataFrame.

    Sample weights are taken from the 'profit_weight' column — this directly
    proxies the WRMSSE optimisation objective (higher-weight SKUs dominate loss).

    Early stopping is applied if eval_df is provided, using
    EARLY_STOPPING_ROUNDS rounds without improvement on the eval set.

    Parameters
    ----------
    train_df     : training DataFrame with feature_cols + 'target_qty' + 'profit_weight'
    feature_cols : list of feature column names (no target, no metadata)
    param_key    : CONFIG key for model-specific hyperparameters
    label        : display label for logging
    n_estimators_override : if set, overrides CONFIG n_estimators (used in CV)
    eval_df      : optional hold-out DataFrame for early stopping
    param_overrides: optional dict of parameters to override the CONFIG ones

    Returns
    -------
    Fitted LGBMRegressor
    """
    params = _make_lgbm_params(param_key)
    if n_estimators_override is not None:
        params["n_estimators"] = n_estimators_override
    if param_overrides is not None:
        params.update(param_overrides)

    # Filter available categorical features
    cat_feats = [c for c in LGBM_CAT_FEATURES if c in feature_cols]

    X_train = train_df[feature_cols].astype("float32")
    y_train = train_df["target_qty"].astype("float32").clip(lower=0.0)
    w_train = train_df["profit_weight"].astype("float32")

    model = lgb.LGBMRegressor(**params)

    t0 = time.time()
    if eval_df is not None:
        X_eval = eval_df[feature_cols].astype("float32")
        y_eval = eval_df["target_qty"].astype("float32").clip(lower=0.0)
        w_eval = eval_df["profit_weight"].astype("float32")

        callbacks = [
            lgb.early_stopping(CONFIG["EARLY_STOPPING_ROUNDS"], verbose=False),
            lgb.log_evaluation(period=200),
        ]
        model.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=[(X_eval, y_eval)],
            eval_sample_weight=[w_eval],
            callbacks=callbacks,
            categorical_feature=cat_feats if cat_feats else "auto",
        )
    else:
        model.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            categorical_feature=cat_feats if cat_feats else "auto",
        )

    elapsed = time.time() - t0
    n_trees = model.best_iteration_ if model.best_iteration_ else params["n_estimators"]
    print(
        f"  [{label}] trained {n_trees} trees in {elapsed:.1f}s | "
        f"rows={len(train_df):,}"
    )
    return model


def split_train_eval(
    df: pd.DataFrame,
    eval_frac: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal train/eval split for early stopping within a single model training.

    The LAST eval_frac fraction of unique origin dates is used as the eval set.
    This respects temporal ordering — the eval set is strictly more recent.

    Parameters
    ----------
    df         : training DataFrame with '_origin_date' column
    eval_frac  : fraction of most-recent origin dates to use as eval

    Returns
    -------
    (train_split, eval_split) DataFrames
    """
    if "_origin_date" not in df.columns:
        # Cannot split temporally; return full data as train, empty eval
        return df, pd.DataFrame()

    sorted_origins = sorted(df["_origin_date"].unique())
    n_eval = max(1, int(len(sorted_origins) * eval_frac))
    eval_origins = set(sorted_origins[-n_eval:])

    eval_mask = df["_origin_date"].isin(eval_origins)
    return df[~eval_mask].copy(), df[eval_mask].copy()


# ==============================================================================
# 4 — NAIVE MODEL D
# ==============================================================================


def predict_naive(
    panels_np: Dict[str, np.ndarray],
    skus_sorted: List[str],
    all_dates: pd.DatetimeIndex,
    origin_idx: int,
    horizon: int = 56,
) -> np.ndarray:
    """
    Naive seasonal forecast: for each forecast day, predict the mean of
    the same day-of-week over the last 4 weeks before the origin.

    Fallback: if same-DOW observations are all zero, use the mean of the
    last 28 days. If that is also zero, predict 0.

    Parameters
    ----------
    panels_np   : numpy panel arrays (qty used)
    skus_sorted : SKU list (column order)
    all_dates   : DatetimeIndex for index-to-date mapping
    origin_idx  : integer index of the forecast origin in all_dates
    horizon     : number of forecast days (default 56)

    Returns
    -------
    np.ndarray of shape (n_skus, horizon), float32, clipped ≥ 0
    """
    n_skus = len(skus_sorted)
    qty_arr = panels_np["qty"]
    preds = np.zeros((n_skus, horizon), dtype="float32")

    # Recent 28 days fallback
    recent_start = max(0, origin_idx - 28)
    recent_28 = qty_arr[recent_start:origin_idx]  # (≤28, n_skus)
    fallback_mean = np.maximum(0, recent_28.mean(axis=0))  # (n_skus,)

    # Last 4 same-DOW observations (up to 28 days prior)
    look_back_dates = all_dates[max(0, origin_idx - 28) : origin_idx]
    look_back_vals = qty_arr[max(0, origin_idx - 28) : origin_idx]

    for h in range(1, horizon + 1):
        target_date = all_dates[origin_idx] + pd.Timedelta(days=h)
        target_dow = target_date.dayofweek

        # Indices within look_back_dates with same DOW
        dow_mask = np.array([d.dayofweek == target_dow for d in look_back_dates])
        if dow_mask.sum() > 0:
            same_dow_vals = look_back_vals[dow_mask]  # (k, n_skus)
            same_dow_mean = np.maximum(0, same_dow_vals.mean(axis=0))
            preds[:, h - 1] = same_dow_mean
        else:
            preds[:, h - 1] = fallback_mean

    return preds


# ==============================================================================
# 5 — ETS MODEL E
# ==============================================================================


def fit_ets_models(
    panel: pd.DataFrame,
    sku_features: pd.DataFrame,
    hard_zero_skus: set,
) -> Dict[str, object]:
    """
    Fit ExponentialSmoothing (Holt-Winters) models for eligible SKUs.

    Eligibility (from Stage 1 sku_features.ets_available == 1):
        A-tier Dense or Intermittent, NOT in hard_zero_skus.

    Model configuration:
        - trend='add', damped_trend=True  — additive damped trend to avoid
          explosive long-range extrapolation (critical for week 7-8 predictions)
        - seasonal='add', seasonal_periods=7  — weekly seasonality (ACF confirmed)
        - Additive seasonality handles zero-containing series; multiplicative
          would fail on intermittent A-tier SKUs.

    For SKUs where fitting fails (rare convergence errors), falls back to
    a simple exponential smoothing (no trend, no seasonality).

    Parameters
    ----------
    panel         : Stage 0 daily panel
    sku_features  : Stage 1 extended SKU feature table (indexed by ItemCode)
    hard_zero_skus: set of SKUs with hard-zero override

    Returns
    -------
    dict: ItemCode → fitted ExponentialSmoothing result object
    """
    eligible = sku_features[
        (sku_features["ets_available"] == 1)
        & (~sku_features.index.isin(hard_zero_skus))
    ].index.tolist()

    print(f"  Fitting ETS for {len(eligible)} eligible SKUs …")
    ets_models = {}
    failed = 0

    for i, sku in enumerate(eligible):
        series = (
            panel[panel["ItemCode"] == sku].set_index("Date")["net_qty"].sort_index()
        )
        # Use post-cutoff data only
        series = series[series.index >= _CUTOFF].astype(float)

        # Need at least 2 full seasonal periods (14 days) + forecast horizon
        if len(series) < 14 + 56:
            failed += 1
            continue

        # Replace negatives with 0 for ETS (model sees net demand)
        series = series.clip(lower=0.0)

        try:
            model = ExponentialSmoothing(
                series,
                trend="add",
                damped_trend=True,
                seasonal="add",
                seasonal_periods=7,
                initialization_method="estimated",
            )
            fit = model.fit(optimized=True, use_brute=False)
            ets_models[sku] = fit

        except Exception:
            # Fallback: simple exponential smoothing
            try:
                model_simple = ExponentialSmoothing(
                    series,
                    trend=None,
                    seasonal=None,
                    initialization_method="estimated",
                )
                fit_simple = model_simple.fit(optimized=True)
                ets_models[sku] = fit_simple
            except Exception:
                failed += 1

        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(eligible)} fitted …", end="\r")

    print(f"\n  ETS fitted: {len(ets_models)} | failed/skipped: {failed}")
    return ets_models


def predict_ets(
    ets_models: Dict[str, object],
    skus_sorted: List[str],
    horizon: int = 56,
) -> np.ndarray:
    """
    Produce ETS forecasts for all SKUs.

    Non-ETS SKUs receive prediction 0 (they use A/B/C/D blend instead).

    Returns
    -------
    np.ndarray of shape (n_skus, horizon), float32, clipped ≥ 0
    """
    n_skus = len(skus_sorted)
    preds = np.zeros((n_skus, horizon), dtype="float32")

    for i, sku in enumerate(skus_sorted):
        if sku in ets_models:
            try:
                fc = ets_models[sku].forecast(horizon)
                preds[i] = np.maximum(0, fc.values).astype("float32")
            except Exception:
                pass  # stays 0 — will be excluded by blend weights

    return preds


# ==============================================================================
# 6 — RECURSIVE MODEL B INFERENCE
# ==============================================================================

# SKIP

# ==============================================================================
# 7 — CROSS-VALIDATION
# ==============================================================================


def build_fold_inference_features(
    fold: Dict,
    panels_np: Dict[str, np.ndarray],
    skus_sorted: List[str],
    sku_features: pd.DataFrame,
    all_dates: pd.DatetimeIndex,
    date_to_idx: Dict,
    week_feature_cols: List[str],
) -> Dict[int, pd.DataFrame]:
    """
    Build inference feature DataFrames at the fold's train_end origin date.

    Returns a dict {week_idx (1-8): pd.DataFrame of (n_skus × 7, features)}
    so each week-specialist model can predict directly.
    """
    origin_date = fold["train_end"]
    t_idx = date_to_idx.get(origin_date)
    if t_idx is None:
        raise KeyError(f"Fold train_end {origin_date} not found in all_dates")

    # Compute base features at this fold's origin
    base_matrix, _, _ = compute_base_features_at_origins(
        origin_indices=[t_idx],
        panels=panels_np,
        skus_sorted=skus_sorted,
        sku_features=sku_features,
        all_dates=all_dates,
    )
    # base_matrix shape: (n_skus, N_BASE_FEATURES)

    week_feat_dfs = {}
    n_skus = len(skus_sorted)

    for week_idx in range(1, 5):  # CV covers weeks 1-4 only
        h_start = (week_idx - 1) * 7
        rows = []
        for day_in_week in range(1, 8):
            h = h_start + day_in_week
            target_date = origin_date + pd.Timedelta(days=h)
            dow = target_date.dayofweek
            quarter = (target_date.month - 1) // 3 + 1

            target_cal = np.array(
                [
                    dow,
                    target_date.day,
                    target_date.isocalendar()[1],
                    target_date.month,
                    quarter,
                    int(dow >= 5),
                    int(dow == 6),
                    day_in_week,
                    week_idx,
                ],
                dtype="float32",
            )

            target_tiled = np.tile(target_cal, (n_skus, 1))
            full = np.hstack([base_matrix, target_tiled])
            chunk = pd.DataFrame(full, columns=ALL_FEATURE_NAMES, dtype="float32")
            chunk["ItemCode"] = skus_sorted
            chunk["week_idx"] = week_idx
            chunk["day_in_week"] = day_in_week
            chunk["horizon_day"] = h
            chunk["target_date"] = target_date
            rows.append(chunk)

        week_feat_dfs[week_idx] = pd.concat(rows, ignore_index=True)

    return week_feat_dfs


def run_cv(
    ctx: Dict,
    panel: pd.DataFrame,
) -> Dict:
    """
    5-fold CV evaluation of the full ensemble pipeline.

    For each fold:
      1. Train A1-A8 on filtered training data (origin < val_start)
         with reduced n_estimators for speed.
      2. Train per-fold ETS on training series (clipped to fold train_end).
      3. Predict 56 days (weeks 1-8) at origin = fold train_end.
      4. Ensemble blend as per CONFIG["WEEK_WEIGHTS"] and TIER_BLEND.
      5. Compute WRMSSE vs actual values from panel.

    Parameters
    ----------
    ctx            : setup_stage2() output dict
    panel          : Stage 0 daily panel (for actuals lookup)
    holiday_lookup : precomputed holiday lookup dict

    Returns
    -------
    cv_results dict with keys:
        fold_wrmsse     : list[float] per-fold WRMSSE
        model_wrmsse    : dict {model_key: list[float]} per-fold per-model
        oof_predictions : pd.DataFrame of all OOF predictions
        mean_wrmsse     : float
        std_wrmsse      : float
    """
    print("\n" + "=" * 55)
    print(" Stage 3: Cross-Validation")
    print("=" * 55)

    sku_features = ctx["sku_features"]
    train_datasets = ctx["train_datasets"]
    panels_np = ctx["panels_np"]
    skus_sorted = ctx["skus_sorted"]
    all_dates = ctx["all_dates"]
    date_to_idx = ctx["date_to_idx"]
    hard_zero_skus = ctx["hard_zero_skus"]
    weights_series = ctx["weights_series"]
    wfc = ctx["week_feature_cols"]

    fold_wrmsse_list = []
    model_wrmsse_dict = {m: [] for m in ["A", "D", "E", "Ensemble"]}
    oof_records = []

    # Panel wide format for actuals lookup
    panel_wide_qty = panel.pivot_table(
        index="Date", columns="ItemCode", values="net_qty", fill_value=0
    ).reindex(columns=skus_sorted, fill_value=0)

    for fold in ctx["folds_updated"]:
        fold_id = fold["fold_id"]
        val_start = fold["val_start"]
        val_end = fold["val_end"]
        train_end = fold["train_end"]

        print(f"\n  ── Fold {fold_id} | val: {val_start.date()} → {val_end.date()} ──")

        # ── Actual values for this fold ───────────────────────────────────
        val_dates = pd.date_range(val_start, val_end, freq="D")
        actuals_df = panel_wide_qty.reindex(val_dates, fill_value=0)
        h = len(val_dates)

        if h == 0:
            print("  [SKIP] Empty validation window.")
            continue

        # ── Train A1-A8 ───────────────────────────────────────────────────
        a_models_cv = {}
        for w in range(1, 5):
            if w not in train_datasets:
                continue
            ds = train_datasets[w]
            train_sub = ds[ds["_origin_date"] < val_start].copy()
            if len(train_sub) < 100:
                print(f"  [WARN] Week {w}: only {len(train_sub)} rows — skipping.")
                continue

            dynamic_params = {}
            if w <= 2:
                dynamic_params = {"num_leaves": 64, "min_child_samples": 20, "lambda_l2": 0.1, "learning_rate": 0.08}
            elif w <= 4:
                dynamic_params = {"num_leaves": 48, "min_child_samples": 50, "lambda_l2": 0.5, "learning_rate": 0.08}
            else:
                dynamic_params = {"num_leaves": 31, "min_child_samples": 100, "lambda_l2": 2.0, "learning_rate": 0.05}

            a_models_cv[w] = train_lgbm(
                train_sub,
                wfc,
                "LGBM_A",
                label=f"F{fold_id}-A{w}",
                n_estimators_override=CONFIG["CV_N_ESTIMATORS"],
                eval_df=None,
                param_overrides=dynamic_params,
            )

        # ── Models C and B disabled ───────────────────────────────────────
        c_models_cv = {}
        model_b_cv = None

        # ── Fit ETS on fold's training series ─────────────────────────────
        panel_fold = panel[panel["Date"] <= train_end].copy()
        ets_cv = fit_ets_models(panel_fold, sku_features, hard_zero_skus)
        del panel_fold

        # ── Build inference features at fold origin ───────────────────────
        fold_inf = build_fold_inference_features(
            fold=fold,
            panels_np=panels_np,
            skus_sorted=skus_sorted,
            sku_features=sku_features,
            all_dates=all_dates,
            date_to_idx=date_to_idx,
            week_feature_cols=wfc,
        )

        fold_origin_idx = date_to_idx[train_end]

        # ── Predict: each model ───────────────────────────────────────────
        n_skus = len(skus_sorted)
        preds_A = np.zeros((n_skus, h), dtype="float32")
        preds_D = predict_naive(
            panels_np, skus_sorted, all_dates, fold_origin_idx, horizon=h
        )
        preds_E = predict_ets(ets_cv, skus_sorted, horizon=h)

        for w in range(1, 5):
            day_s = (w - 1) * 7
            day_e = w * 7
            if w in fold_inf and w in a_models_cv:
                feat = fold_inf[w][wfc].astype("float32")
                raw = a_models_cv[w].predict(feat)
                # Reshape: (n_skus × 7,) → (n_skus, 7) via column-stacking
                # fold_inf[w] is ordered: all SKUs for day1, then all for day2, …
                raw_mat = raw.reshape(7, n_skus).T  # (n_skus, 7)
                preds_A[:, day_s:day_e] = np.maximum(0, raw_mat)

        # ── Ensemble blend ────────────────────────────────────────────────
        all_model_preds = {"A": preds_A, "D": preds_D, "E": preds_E}
        preds_ensemble = _blend_ensemble(
            all_model_preds,
            sku_features,
            skus_sorted,
            horizon=h,
        )

        # ── WRMSSE ────────────────────────────────────────────────────────
        train_wide_fold = panel_wide_qty.loc[:train_end]
        actuals_np = actuals_df.values.astype("float32")  # (h, n_skus)

        def _wrmsse_np(pred_mat):
            """pred_mat shape: (n_skus, h). Returns scalar WRMSSE."""
            w_vec = weights_series.values
            rmsse_vals = []
            for i in range(n_skus):
                y_true = actuals_np[:, i]
                y_pred = pred_mat[i]
                y_train = train_wide_fold.iloc[:, i].values.astype("float32")
                denom = compute_naive_rmsse_denominator(y_train)
                if denom == 0.0:
                    rmsse_vals.append(0.0)
                else:
                    rmsse_vals.append(
                        float(np.sqrt(np.mean((y_true - y_pred) ** 2) / denom))
                    )
            rmsse_arr = np.array(rmsse_vals, dtype="float32")
            return float(np.dot(rmsse_arr, w_vec))

        wrmsse_ens = _wrmsse_np(preds_ensemble)
        wrmsse_a = _wrmsse_np(preds_A)
        wrmsse_d = _wrmsse_np(preds_D[:, :h])
        wrmsse_e = _wrmsse_np(preds_E[:, :h])

        print(
            f"  WRMSSE  Ensemble={wrmsse_ens:.4f} | "
            f"A={wrmsse_a:.4f} | D={wrmsse_d:.4f} | E={wrmsse_e:.4f}"
        )

        # ── Magic Multiplier Sweep ──────────────────────────────────────────
        best_mult, best_score = 1.0, wrmsse_ens
        print("  [Magic Multiplier Sweep]")
        for mult in [0.95, 0.97, 0.98, 0.99, 1.01, 1.02, 1.03, 1.05]:
            m_score = _wrmsse_np(preds_ensemble * mult)
            diff = m_score - wrmsse_ens
            marker = "<- BEST" if m_score < best_score else ""
            if m_score < best_score:
                best_mult, best_score = mult, m_score
            print(f"    x{mult:.2f} -> WRMSSE: {m_score:.4f} (diff: {diff:+.4f}) {marker}")

        fold_wrmsse_list.append(wrmsse_ens)
        for key, val in [
            ("A", wrmsse_a),
            ("D", wrmsse_d),
            ("E", wrmsse_e),
            ("Ensemble", wrmsse_ens),
        ]:
            model_wrmsse_dict[key].append(val)

        # # ── Save OOF records (top-50 weight SKUs only to keep memory low) ─
        # top50_skus = set(weights_series.nlargest(50).index.tolist())
        # for i, sku in enumerate(skus_sorted):
        #     if sku not in top50_skus:
        #         continue
        #     for d in range(h):
        #         oof_records.append(
        #             {
        #                 "fold_id": fold_id,
        #                 "ItemCode": sku,
        #                 "target_date": val_dates[d],
        #                 "actual": float(actuals_np[d, i]),
        #                 "pred_ensemble": float(preds_ensemble[i, d]),
        #                 "pred_A": float(preds_A[i, d]),
        #                 "pred_E": float(preds_E[i, d]),
        #                 "pred_D": float(preds_D[i, d]),
        #             }
        #         )

        # ── Save OOF records efficiently for all SKUs ───────────────────────
        sku_col = np.repeat(skus_sorted, h)
        date_col = np.tile(val_dates, n_skus)
        
        fold_df = pd.DataFrame({
            "fold_id": fold_id,
            "ItemCode": sku_col,
            "target_date": date_col,
            "actual": actuals_np.T.flatten(),
            "pred_ensemble": preds_ensemble.flatten(),
            "pred_A": preds_A.flatten(),
            "pred_E": preds_E[:, :h].flatten(),
            "pred_D": preds_D[:, :h].flatten(),
        })
        oof_records.append(fold_df)

        # Free fold memory
        del a_models_cv, c_models_cv, model_b_cv, ets_cv, fold_inf
        gc.collect()

    mean_w = float(np.nanmean(fold_wrmsse_list)) if fold_wrmsse_list else np.nan
    std_w = float(np.nanstd(fold_wrmsse_list)) if fold_wrmsse_list else np.nan

    print(f"\n  CV WRMSSE: {mean_w:.4f} ± {std_w:.4f}")
    print(
        f"  Naive baseline was: "
        f"{naive_results['mean_wrmsse']:.4f} ± {naive_results['std_wrmsse']:.4f}"
    )
    print(
        f"  Improvement over naive: "
        f"{(naive_results['mean_wrmsse'] - mean_w):.4f} "
        f"({(1 - mean_w / naive_results['mean_wrmsse']) * 100:.1f}%)"
    )

    # oof_df = pd.DataFrame(oof_records)
    oof_df = pd.concat(oof_records, ignore_index=True)

    return {
        "fold_wrmsse": fold_wrmsse_list,
        "model_wrmsse": model_wrmsse_dict,
        "oof_predictions": oof_df,
        "mean_wrmsse": mean_w,
        "std_wrmsse": std_w,
    }


# ==============================================================================
# 8 — ENSEMBLE BLEND LOGIC  (shared by CV and final inference)
# ==============================================================================


def _blend_ensemble(
    model_preds: Dict[str, np.ndarray],  # {model_key: (n_skus, h)}
    sku_features: pd.DataFrame,
    skus_sorted: List[str],
    horizon: int = 56,
) -> np.ndarray:
    """
    Blend model predictions using tier-aware, week-aware weights.

    Logic
    -----
    For A-tier Dense/Intermittent SKUs:
        Use CONFIG["WEEK_WEIGHTS"] which specify per-week weights for A,D,E.
        ETS weight is only non-zero for SKUs where ets_available == 1.
    For B-tier / C-tier / D-tier SKUs:
        Use CONFIG["TIER_BLEND"] lookup by (profit_tier_enc, demand_density_enc).
    Hard-zero SKUs stay at 0 (not blended).

    Weights are normalised to sum to 1 per (SKU, week) after zeroing-out
    models that contributed 0 (e.g., ETS not available for this SKU).

    Parameters
    ----------
    model_preds  : dict mapping model key to (n_skus, horizon) array
    sku_features : SKU feature table (indexed by ItemCode)
    skus_sorted  : SKU column order matching model_preds arrays
    horizon      : number of forecast days (≤ 56)

    Returns
    -------
    np.ndarray of shape (n_skus, horizon), float32, clipped ≥ 0
    """
    n_skus = len(skus_sorted)
    blended = np.zeros((n_skus, horizon), dtype="float32")
    sf = sku_features.reindex(skus_sorted)

    tier_arr = sf["profit_tier_enc"].fillna(3).values.astype(int)  # 0=A,1=B,2=C,3=D
    density_arr = (
        sf["demand_density_enc"].fillna(2).values.astype(int)
    )  # 0=Dense,1=Int,2=Sparse
    ets_arr = sf["ets_available"].fillna(0).values.astype(int)

    for day in range(1, horizon + 1):
        week_idx = (day - 1) // 7 + 1  # 1-8
        week_key = "w14" if week_idx <= 4 else "w58"
        d = day - 1  # 0-indexed column

        for i in range(n_skus):
            tier = tier_arr[i]
            density = density_arr[i]
            has_ets = (
                bool(ets_arr[i]) and density == 0
            )  # Restrict ETS to Dense SKUs only

            if tier == 0:  # A-tier: use week weights
                raw_w = dict(CONFIG["WEEK_WEIGHTS"].get(week_idx, {"A": 1.0}))
                # Zero out E if no ETS or not Dense
                if not has_ets:
                    raw_w.pop("E", None)
            else:
                # Lower tiers: use TIER_BLEND
                blend_entry = CONFIG["TIER_BLEND"].get(
                    (tier, density), {"w14": {"D": 1.0}, "w58": {"D": 1.0}}
                )
                raw_w = dict(blend_entry[week_key])

            # Normalise weights
            total_w = sum(raw_w.values())
            if total_w <= 0:
                raw_w = {"D": 1.0}
                total_w = 1.0
            norm_w = {k: v / total_w for k, v in raw_w.items()}

            # Compute blended prediction
            val = 0.0
            for model_key, w in norm_w.items():
                if model_key in model_preds and d < model_preds[model_key].shape[1]:
                    val += w * float(model_preds[model_key][i, d])
            blended[i, d] = max(0.0, val)

    return blended


# ==============================================================================
# 9 — FINAL MODEL TRAINING  (full data)
# ==============================================================================


def train_final_models(
    ctx: Dict,
    panel: pd.DataFrame,
) -> Dict:
    """
    Train all final models on the complete training dataset.

    Models trained
    --------------
    A1–A8 : LightGBM Non-Recursive Tweedie, one per week
    C1–C4 : LightGBM Non-Recursive Poisson, weeks 1-4 only
    B     : LightGBM Recursive Tweedie, 2-year training window
    D     : Naive (no training needed, computed at inference time)
    E     : ExponentialSmoothing, one per eligible A-tier SKU

    Early stopping for A and C models uses a temporal eval split
    (last 15% of origin dates).

    Returns
    -------
    dict with keys: A1..A8, C1..C4, B, E
    """
    print("\n" + "=" * 55)
    print(" Stage 4: Final Model Training")
    print("=" * 55)

    final_models = {}
    wfc = ctx["week_feature_cols"]

    # ── A1–A8 ──────────────────────────────────────────────────────────────
    print("\n  Training A1–A8 (Non-Recursive Tweedie) …")
    for w in range(1, 9):
        if w not in ctx["train_datasets"]:
            print(f"  [WARN] Week {w} dataset missing — skipping.")
            continue
        ds = ctx["train_datasets"][w].copy()

        dynamic_params = {}
        if w <= 2:
            dynamic_params = {"num_leaves": 64, "min_child_samples": 20, "lambda_l2": 0.1, "learning_rate": 0.05}
        elif w <= 4:
            dynamic_params = {"num_leaves": 48, "min_child_samples": 50, "lambda_l2": 0.5, "learning_rate": 0.05}
        else:
            dynamic_params = {"num_leaves": 31, "min_child_samples": 100, "lambda_l2": 2.0, "learning_rate": 0.035}

        final_models[f"A{w}"] = train_lgbm(
            ds,
            wfc,
            "LGBM_A",
            label=f"A{w}-final",
            eval_df=None,
            param_overrides=dynamic_params,
        )
        del ds
        gc.collect()

    # ── Models C and B disabled ────────────────────────────────────────────
    # C1-C4 and B models are dropped for performance and stability reasons.

    # ── Model E ────────────────────────────────────────────────────────────
    print("\n  Training Model E (per-SKU ETS) …")
    final_models["E_models"] = fit_ets_models(
        panel, ctx["sku_features"], ctx["hard_zero_skus"]
    )

    print("\n  Final model training complete.")
    return final_models


# ==============================================================================
# 10 — HYPERPARAMETER SEARCH (optional, run after baseline CV)
# ==============================================================================


def search_tweedie_power(
    ctx: Dict,
    week_idx: int = 1,
    n_trials: int = 15,
) -> float:
    """
    Optuna search over Tweedie variance power using CV on a single week.

    Optimises the tweedie_variance_power parameter for the A-model.
    Only week 1 is used to keep runtime manageable; the optimal power
    is then applied to all A. Uses a weighted RMSE to match
    the WRMSSE competition metric.

    Parameters
    ----------
    ctx      : setup_stage2() dict
    week_idx : which week dataset to use for the search
    n_trials : number of Optuna trials

    Returns
    -------
    best_power : float
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"\n  Optuna Tweedie variance power search on Week {week_idx} (trials={n_trials}) …")
    ds = ctx["train_datasets"].get(week_idx)
    if ds is None:
        print("  Dataset not available — returning default 1.5")
        return 1.5

    wfc = ctx["week_feature_cols"]

    # Use fold 4 (most recent) as eval
    fold4 = ctx["folds_updated"][-1]
    tr_mask = ds["_origin_date"] < fold4["val_start"]
    ev_mask = ds["_origin_date"] >= fold4["val_start"]
    tr_ds = ds[tr_mask]
    ev_ds = ds[ev_mask]

    if len(ev_ds) < 50:
        print("  Insufficient eval data — returning default 1.5")
        return 1.5

    X_tr = tr_ds[wfc].astype("float32")
    y_tr = tr_ds["target_qty"].astype("float32").clip(lower=0.0)
    w_tr = tr_ds["profit_weight"].astype("float32")
    X_ev = ev_ds[wfc].astype("float32")
    y_ev = ev_ds["target_qty"].astype("float32").clip(lower=0.0)
    w_ev = ev_ds["profit_weight"].astype("float32").values

    def objective(trial):
        p = trial.suggest_float("tweedie_variance_power", 1.1, 1.5)
        params = {
            **CONFIG["LGBM_BASE"],
            **CONFIG["LGBM_A"],
            "tweedie_variance_power": p,
            "n_estimators": CONFIG["CV_N_ESTIMATORS"],
        }
        # Apply the same dynamic schedule used in run_cv for W1
        dynamic_params = {"num_leaves": 64, "min_child_samples": 20, "lambda_l2": 0.1, "learning_rate": 0.08}
        params.update(dynamic_params)

        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr, sample_weight=w_tr)
        pred_val = model.predict(X_ev)
        
        sq_err = (y_ev.values - pred_val) ** 2
        weight_sum = np.sum(w_ev)
        if weight_sum > 0:
            wrmse_val = float(np.sqrt(np.sum(sq_err * w_ev) / weight_sum))
        else:
            wrmse_val = float(np.sqrt(np.mean(sq_err)))
            
        return wrmse_val

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    best_power = study.best_params["tweedie_variance_power"]
    print(f"  Best Tweedie power: {best_power:.3f} (Weighted RMSE: {study.best_value:.4f})")
    
    return best_power


# ==============================================================================
# 11 — FULL INFERENCE & SUBMISSION BUILD
# ==============================================================================


def run_final_inference(
    ctx: Dict,
    final_models: Dict,
) -> np.ndarray:
    """
    Run inference for all models and produce blended ensemble predictions.

    Returns
    -------
    ensemble_preds : np.ndarray of shape (n_skus, 56), float32
    all_model_preds: dict {model_key: (n_skus, 56)} for diagnostic plots
    """
    print("\n--- Stage 5: Final Inference ---")

    inf_ds = ctx["inference_ds"]
    skus_sorted = ctx["skus_sorted"]
    sku_features = ctx["sku_features"]
    panels_np = ctx["panels_np"]
    all_dates = ctx["all_dates"]
    date_to_idx = ctx["date_to_idx"]
    origin_idx = ctx["origin_idx"]
    wfc = ctx["week_feature_cols"]
    n_skus = len(skus_sorted)

    preds_A = np.zeros((n_skus, 56), dtype="float32")

    # ── A1–A8 ──────────────────────────────────────────────────────────────
    print("  Running A1–A8 inference …")
    for w in range(1, 9):
        key = f"A{w}"
        if key not in final_models:
            continue
        mask = inf_ds["week_idx"] == w
        X = inf_ds.loc[mask, wfc].astype("float32")
        raw = np.maximum(0, final_models[key].predict(X)).astype("float32")
        # inf_ds rows for this week are ordered: all 7 days × all SKUs
        # but actually ordering is SKU-level within each day; let's use
        # ItemCode to align properly
        inf_week = inf_ds[mask].copy()
        inf_week["_pred"] = raw
        for day_in_week in range(1, 8):
            h = (w - 1) * 7 + day_in_week
            day_mask = inf_week["day_in_week"] == day_in_week
            day_rows = inf_week[day_mask].set_index("ItemCode")["_pred"]
            day_rows = day_rows.reindex(skus_sorted, fill_value=0.0)
            preds_A[:, h - 1] = day_rows.values

    # ── Model D (Naive) ────────────────────────────────────────────────────
    print("  Running Naive (D) inference …")
    preds_D = predict_naive(panels_np, skus_sorted, all_dates, origin_idx, horizon=56)

    # ── Model E (ETS) ──────────────────────────────────────────────────────
    print("  Running ETS (E) inference …")
    preds_E = predict_ets(final_models["E_models"], skus_sorted, horizon=56)

    # ── Ensemble blend ─────────────────────────────────────────────────────
    print("  Blending ensemble …")
    all_model_preds = {"A": preds_A, "D": preds_D, "E": preds_E}
    ensemble_preds = _blend_ensemble(
        all_model_preds,
        sku_features,
        skus_sorted,
        horizon=56,
    )

    print(
        f"  Ensemble shape: {ensemble_preds.shape} | "
        f"mean={ensemble_preds.mean():.3f} | "
        f"max={ensemble_preds.max():.0f}"
    )
    return ensemble_preds, all_model_preds


# ==============================================================================
# 12 — POST-PROCESSING
# ==============================================================================


def apply_post_processing(
    preds: np.ndarray,
    skus_sorted: List[str],
    hard_zero_skus: set,
    origin_date: pd.Timestamp,
) -> np.ndarray:
    """
    Apply all post-processing rules in order.

    Rules (applied in this exact sequence)
    ---------------------------------------
    1. Clip all values to ≥ 0  (required by competition rules)
    2. Sunday override: any forecast day that is a Sunday → 0
       Empirically confirmed: business closed Sundays, predicting 0 beats
       any non-zero forecast.
    3. Hard-zero SKU override: SKUs in hard_zero_skus → 0 for all 56 days.
       Covers: extreme sparsity + silent recent period, pre-cutoff-only,
       discontinued SKUs.

    Parameters
    ----------
    preds          : (n_skus, 56) float32 array from ensemble blend
    skus_sorted    : SKU list matching preds row order
    hard_zero_skus : set of hard-zero SKU codes
    origin_date    : forecast origin (CONFIG["TRAIN_END"] = 2025-09-05)

    Returns
    -------
    np.ndarray of shape (n_skus, 56), float32, fully post-processed
    """
    out = preds.copy()

    # Rule 1: clip to non-negative
    np.clip(out, 0.0, None, out=out)

    # Rule 2: Sunday override
    sunday_days = []
    for h in range(1, 57):
        target_date = origin_date + pd.Timedelta(days=h)
        if target_date.dayofweek == 6:  # Sunday = 6
            sunday_days.append(h - 1)  # 0-indexed

    if sunday_days:
        out[:, sunday_days] = 0.0
        print(
            f"  Sunday override applied: days (1-indexed) "
            f"{[d + 1 for d in sunday_days]}"
        )

    # Rule 3: hard-zero SKU override
    hz_indices = [i for i, sku in enumerate(skus_sorted) if sku in hard_zero_skus]
    if hz_indices:
        out[hz_indices, :] = 0.0
        print(f"  Hard-zero override: {len(hz_indices)} SKUs set to 0 for all 56 days")

    # Final clip for floating-point safety
    np.clip(out, 0.0, None, out=out)
    return out


# ==============================================================================
# 13 — SUBMISSION GENERATION
# ==============================================================================


def build_submission(
    preds: np.ndarray,  # (n_skus, 56), post-processed
    skus_sorted: List[str],
    sub_template: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the submission DataFrame matching the sample_submission.csv template.

    Mapping
    -------
    Validation rows (id ends in '_validation'): F1…F28 = horizon days 1…28
    Evaluation rows (id ends in '_evaluation'): F1…F28 = horizon days 29…56

    Parameters
    ----------
    preds        : post-processed (n_skus, 56) predictions array
    skus_sorted  : SKU list matching preds row order
    sub_template : sample_submission.csv DataFrame

    Returns
    -------
    pd.DataFrame with same columns as sub_template, ready to save as CSV
    """
    sku_to_idx = {sku: i for i, sku in enumerate(skus_sorted)}

    submission = sub_template.copy()

    # Parse id column
    id_parsed = submission["id"].str.extract(r"^(.+)_(validation|evaluation)$")
    submission["_sku"] = id_parsed[0]
    submission["_window"] = id_parsed[1]

    forecast_cols = [f"F{i}" for i in range(1, 29)]

    for _, row in submission.iterrows():
        sku = row["_sku"]
        window = row["_window"]
        idx = sku_to_idx.get(sku)

        if idx is None:
            # SKU not in our training data — predict 0
            for col in forecast_cols:
                submission.at[row.name, col] = 0.0
            continue

        if window == "validation":
            # F1-F28 = horizon days 1-28 (0-indexed: 0-27)
            vals = preds[idx, 0:28]
        else:
            # F1-F28 = horizon days 29-56 (0-indexed: 28-55)
            vals = preds[idx, 28:56]

        for j, col in enumerate(forecast_cols):
            submission.at[row.name, col] = float(vals[j])

    submission = submission.drop(columns=["_sku", "_window"])
    return submission


def validate_submission(sub_df: pd.DataFrame, sub_template: pd.DataFrame) -> bool:
    """
    Sanity-check the submission DataFrame before saving.

    Checks
    ------
    1. Row count matches template exactly.
    2. All IDs present and no duplicates.
    3. No NaN or Inf values.
    4. All forecast values ≥ 0.
    5. Total predicted volume is within 0.1× – 10× of template's dummy (0) baseline
       (this check only validates it's non-trivially non-zero).
    """
    ok = True
    print("\n  Submission validation:")

    # 1. Row count
    if len(sub_df) != len(sub_template):
        print(f"  !! Row count mismatch: {len(sub_df)} vs {len(sub_template)}")
        ok = False
    else:
        print(f"  ✓ Row count: {len(sub_df):,}")

    # 2. IDs
    missing = set(sub_template["id"]) - set(sub_df["id"])
    dupes = sub_df["id"].duplicated().sum()
    if missing:
        print(f"  !! Missing IDs: {len(missing)}")
        ok = False
    else:
        print("  ✓ All IDs present")
    if dupes > 0:
        print(f"  !! Duplicate IDs: {dupes}")
        ok = False

    # 3. NaN / Inf
    fc = [f"F{i}" for i in range(1, 29)]
    n_nan = sub_df[fc].isna().sum().sum()
    n_inf = np.isinf(sub_df[fc].values.astype(float)).sum()
    if n_nan > 0:
        print(f"  !! NaN values: {n_nan}")
        ok = False
    if n_inf > 0:
        print(f"  !! Inf values: {n_inf}")
        ok = False
    if n_nan == 0 and n_inf == 0:
        print("  ✓ No NaN or Inf")

    # 4. Non-negative
    n_neg = (sub_df[fc].values < 0).sum()
    if n_neg > 0:
        print(f"  !! Negative values: {n_neg}")
        ok = False
    else:
        print("  ✓ All values ≥ 0")

    total = sub_df[fc].values.sum()
    print(f"  Total predicted volume: {total:,.1f}")
    return ok


# ==============================================================================
# 14 — DIAGNOSTIC PLOTS
# ==============================================================================


def plot_cv_results(cv_results: Dict) -> None:
    """
    Plot Set T3 / CV: WRMSSE per fold per model, with ensemble highlighted.
    """
    set_plot_style()
    model_keys = ["A", "B", "C", "D", "E", "Ensemble"]
    colors = ["steelblue", "green", "darkorange", "gray", "purple", "red"]
    n_folds = len(cv_results["fold_wrmsse"])
    fold_labels = [f"Fold {i}" for i in range(n_folds)]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("CV WRMSSE Results", fontweight="bold")

    # Per-fold per-model
    ax = axes[0]
    x = np.arange(n_folds)
    w = 0.12
    for j, (key, color) in enumerate(zip(model_keys, colors)):
        vals = cv_results["model_wrmsse"].get(key, [np.nan] * n_folds)
        vals = [v if not np.isnan(v) else 0 for v in vals]
        ax.bar(
            x + j * w,
            vals,
            width=w,
            label=key,
            color=color,
            alpha=0.8,
            edgecolor="none",
        )
    ax.set_xticks(x + w * 2.5)
    ax.set_xticklabels(fold_labels)
    ax.set_ylabel("WRMSSE")
    ax.set_title("Per-Fold WRMSSE by Model")
    ax.legend(fontsize=9)
    ax.axhline(
        naive_results["mean_wrmsse"],
        color="black",
        ls="--",
        lw=1.5,
        label="Naive baseline",
    )

    # Ensemble fold scores with mean ± std
    ax = axes[1]
    ens_scores = cv_results["fold_wrmsse"]
    ax.bar(fold_labels, ens_scores, color="red", alpha=0.75, edgecolor="none")
    mean_w = cv_results["mean_wrmsse"]
    std_w = cv_results["std_wrmsse"]
    ax.axhline(mean_w, color="darkred", lw=2, ls="--", label=f"Mean={mean_w:.4f}")
    ax.axhspan(
        mean_w - std_w,
        mean_w + std_w,
        alpha=0.15,
        color="darkred",
        label=f"±1 std={std_w:.4f}",
    )
    ax.axhline(
        naive_results["mean_wrmsse"],
        color="black",
        ls=":",
        lw=1.5,
        label=f"Naive={naive_results['mean_wrmsse']:.4f}",
    )
    for i, s in enumerate(ens_scores):
        ax.text(i, s + 0.005, f"{s:.4f}", ha="center", fontsize=9)
    ax.set_title("Ensemble CV WRMSSE per Fold")
    ax.set_ylabel("WRMSSE")
    ax.legend(fontsize=9)

    plt.tight_layout()
    save_fig(fig, "T3_cv_wrmsse")
    plt.show()


def plot_feature_importance_comparison(
    final_models: Dict,
    week_feature_cols: List[str],
    top_n: int = 25,
) -> None:
    """
    Plot Set T2: Feature importance comparison for A1, A4, A8.
    Side-by-side shows how important features shift with horizon distance.
    """
    set_plot_style()
    models_to_plot = [
        (f"A{w}", f"Week {w} (days {(w - 1) * 7 + 1}–{w * 7})")
        for w in [1, 4, 8]
        if f"A{w}" in final_models
    ]

    if not models_to_plot:
        return

    fig, axes = plt.subplots(
        1, len(models_to_plot), figsize=(7 * len(models_to_plot), 9)
    )
    if len(models_to_plot) == 1:
        axes = [axes]
    fig.suptitle(
        "Fig T2 — Feature Importance: A1 vs A4 vs A8\n"
        "(shift in importance with forecast horizon)",
        fontweight="bold",
    )

    for ax, (key, title) in zip(axes, models_to_plot):
        model = final_models[key]
        imp = pd.Series(
            model.feature_importances_,
            index=week_feature_cols[: len(model.feature_importances_)],
        ).nlargest(top_n)
        ax.barh(
            range(len(imp)),
            imp.values[::-1],
            color="steelblue",
            edgecolor="none",
            alpha=0.8,
        )
        ax.set_yticks(range(len(imp)))
        ax.set_yticklabels(imp.index[::-1], fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("Importance (gain)")

    plt.tight_layout()
    save_fig(fig, "T2_feature_importance_comparison")
    plt.show()


def plot_learning_curves(
    final_models: Dict,
) -> None:
    """
    Plot Set T1: LightGBM learning curves (eval loss vs boosting round)
    for A1, A4, A8, B, C1.
    """
    set_plot_style()
    keys_to_plot = ["A1", "A4", "A8", "B", "C1"]
    keys_to_plot = [k for k in keys_to_plot if k in final_models]

    if not keys_to_plot:
        return

    fig, axes = plt.subplots(1, len(keys_to_plot), figsize=(5 * len(keys_to_plot), 4))
    if len(keys_to_plot) == 1:
        axes = [axes]
    fig.suptitle("Fig T1 — LightGBM Learning Curves", fontweight="bold")

    for ax, key in zip(axes, keys_to_plot):
        model = final_models[key]
        if hasattr(model, "evals_result_") and model.evals_result_:
            res = model.evals_result_
            for ds_name, metrics in res.items():
                for metric_name, values in metrics.items():
                    ax.plot(values, label=f"{ds_name} {metric_name}", lw=1.5)
            if model.best_iteration_:
                ax.axvline(
                    model.best_iteration_,
                    color="red",
                    ls="--",
                    lw=1,
                    label=f"best={model.best_iteration_}",
                )
        else:
            ax.text(
                0.5,
                0.5,
                "No eval results\n(no early stopping)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.set_title(f"Model {key}")
        ax.set_xlabel("Boosting round")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=7)

    plt.tight_layout()
    save_fig(fig, "T1_learning_curves")
    plt.show()


def plot_horizon_profile(
    all_model_preds: Dict,
    skus_sorted: List[str],
    sku_features: pd.DataFrame,
) -> None:
    """
    Plot Set I2: Mean predicted quantity by forecast day (1-56),
    separately for all SKUs and A-tier SKUs.
    Shows how each model's predictions evolve across the horizon.
    """
    set_plot_style()
    sf = sku_features.reindex(skus_sorted)
    a_mask = (sf["profit_tier_enc"] == 0).values

    model_colors = {
        "A": "steelblue",
        "B": "green",
        "C": "darkorange",
        "D": "gray",
        "E": "purple",
    }

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Fig I2 — Mean Predicted Quantity by Forecast Day", fontweight="bold")

    for ax, (mask, label) in zip(
        axes, [(slice(None), "All SKUs"), (a_mask, "A-Tier SKUs only")]
    ):
        for model_key, color in model_colors.items():
            if model_key not in all_model_preds:
                continue
            preds = all_model_preds[model_key]
            if isinstance(mask, np.ndarray):
                series = preds[mask].mean(axis=0)
            else:
                series = preds.mean(axis=0)
            ax.plot(range(1, 57), series, color=color, lw=2, label=model_key, alpha=0.8)

        ax.axvline(
            28.5,
            color="black",
            ls="--",
            lw=1.5,
            alpha=0.6,
            label="Public|Private split",
        )
        # Shade Sundays
        for h in range(1, 57):
            d = _TRAIN_END + pd.Timedelta(days=h)
            if d.dayofweek == 6:
                ax.axvspan(h - 0.5, h + 0.5, alpha=0.15, color="red")
        ax.set_title(label)
        ax.set_xlabel("Forecast day (1 = Sep 6, 56 = Oct 31)")
        ax.set_ylabel("Mean predicted qty")
        ax.legend(fontsize=9)

    plt.tight_layout()
    save_fig(fig, "I2_horizon_profile")
    plt.show()


def plot_top_sku_traces(
    all_model_preds: Dict,
    ensemble_preds: np.ndarray,
    skus_sorted: List[str],
    sku_features: pd.DataFrame,
    panel: pd.DataFrame,
    n_top: int = 20,
    trailing_days: int = 90,
) -> None:
    """
    Plot Sets I3 and E4: For the top-N profit-weighted SKUs, show:
      - Historical actuals (trailing_days)
      - Individual model predictions (colored lines)
      - Ensemble prediction (thick red line)
      - Naive baseline (dashed gray)
    """
    set_plot_style()

    sf = sku_features.reindex(skus_sorted)
    top_skus = sf["profit_weight"].nlargest(n_top).index.tolist()

    model_colors = {
        "A": "steelblue",
        "B": "green",
        "C": "darkorange",
        "D": "gray",
        "E": "purple",
    }

    for rank, sku in enumerate(top_skus):
        sku_idx = skus_sorted.index(sku)
        weight = float(sf.at[sku, "profit_weight"])
        tier = sf.at[sku, "profit_tier"]
        density = sf.at[sku, "demand_density"]

        # Historical actuals
        hist = (
            panel[panel["ItemCode"] == sku]
            .set_index("Date")["net_qty"]
            .sort_index()
            .iloc[-trailing_days:]
        )
        hist_dates = hist.index.tolist()
        hist_vals = hist.values

        # Forecast dates (56 days)
        fc_dates = [_TRAIN_END + pd.Timedelta(days=h) for h in range(1, 57)]

        fig, ax = plt.subplots(figsize=(16, 4))

        # Historical
        ax.plot(
            hist_dates,
            hist_vals,
            color="black",
            lw=1.2,
            alpha=0.7,
            label="Historical actuals",
        )
        ax.axvline(_TRAIN_END, color="black", lw=1.5, ls="--", alpha=0.5)

        # Individual model predictions
        for model_key, color in model_colors.items():
            if model_key not in all_model_preds:
                continue
            preds_m = all_model_preds[model_key][sku_idx]
            ax.plot(
                fc_dates,
                preds_m,
                color=color,
                lw=1.2,
                alpha=0.65,
                ls="--",
                label=model_key,
            )

        # Ensemble (thick, solid, red)
        ax.plot(
            fc_dates,
            ensemble_preds[sku_idx],
            color="red",
            lw=2.5,
            label="Ensemble",
            zorder=5,
        )

        # Shade Sundays in forecast window
        for h in range(1, 57):
            d = _TRAIN_END + pd.Timedelta(days=h)
            if d.dayofweek == 6:
                ax.axvspan(
                    d - pd.Timedelta(hours=12),
                    d + pd.Timedelta(hours=12),
                    alpha=0.2,
                    color="red",
                )

        # Public/Private split
        ax.axvline(
            _TRAIN_END + pd.Timedelta(days=28),
            color="navy",
            ls=":",
            lw=1.5,
            alpha=0.7,
            label="Public|Private split",
        )

        ax.set_title(
            f"Rank #{rank + 1}: {sku}  |  weight={weight * 100:.3f}%  |  "
            f"Tier={tier}  Density={density}"
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Net qty")
        ax.legend(fontsize=7, ncol=4)

        plt.tight_layout()
        save_fig(fig, f"I3_E4_{sku}_rank{rank + 1}")
        plt.show()
        plt.close(fig)

        if rank >= n_top - 1:
            break


def plot_ensemble_contribution(
    all_model_preds: Dict,
    ensemble_preds: np.ndarray,
    skus_sorted: List[str],
    sku_features: pd.DataFrame,
) -> None:
    """
    Plot Sets E1, E2, E3:
    E1 — Stacked area: mean model weight contribution by week
    E2 — Model agreement (std across models) by forecast day
    E3 — WRMSSE ranking: individual models vs ensemble (using proxy RMSE)
    """
    set_plot_style()
    model_colors = {
        "A": "steelblue",
        "B": "green",
        "C": "darkorange",
        "D": "gray",
        "E": "purple",
    }

    # ── E1: Mean prediction by model by week ─────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    weeks = range(1, 9)
    week_means = {m: [] for m in model_colors}
    for w in weeks:
        d_s, d_e = (w - 1) * 7, w * 7
        for m in model_colors:
            if m in all_model_preds:
                week_means[m].append(all_model_preds[m][:, d_s:d_e].mean())
            else:
                week_means[m].append(0.0)

    bottom = np.zeros(8)
    for m, color in model_colors.items():
        vals = np.array(week_means[m])
        ax.bar(
            range(1, 9),
            vals,
            bottom=bottom,
            color=color,
            label=m,
            alpha=0.8,
            edgecolor="white",
            lw=0.5,
        )
        bottom += vals

    ax.set_xticks(range(1, 9))
    ax.set_xticklabels([f"Wk {w}" for w in range(1, 9)])
    ax.set_title("Fig E1 — Mean Prediction Level by Model by Week")
    ax.set_ylabel("Mean predicted qty")
    ax.legend(fontsize=9)
    plt.tight_layout()
    save_fig(fig, "E1_model_contribution_by_week")
    plt.show()

    # ── E2: Model disagreement (std across models) ────────────────────────
    available = [m for m in model_colors if m in all_model_preds]
    if len(available) >= 2:
        stacked = np.stack([all_model_preds[m] for m in available], axis=2)
        # stacked shape: (n_skus, 56, n_models)
        disagreement = stacked.std(axis=2).mean(axis=0)  # (56,)

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(range(1, 57), disagreement, color="tomato", lw=2)
        ax.fill_between(range(1, 57), 0, disagreement, alpha=0.25, color="tomato")
        ax.axvline(28.5, color="black", ls="--", lw=1.5, label="Public|Private split")
        ax.set_title("Fig E2 — Model Disagreement (std across models) by Forecast Day")
        ax.set_xlabel("Forecast day")
        ax.set_ylabel("Mean std across models")
        ax.legend()
        plt.tight_layout()
        save_fig(fig, "E2_model_disagreement")
        plt.show()

    # ── E3: RMSE by model and ensemble ───────────────────────────────────
    # Use proxy: mean prediction vs grand mean (no actuals at final inference)
    grand_mean = ensemble_preds.mean(axis=0)
    rmse_proxy = {}
    for m in available:
        diff = all_model_preds[m] - grand_mean
        rmse_proxy[m] = float(np.sqrt(np.mean(diff**2)))
    rmse_proxy["Ensemble"] = 0.0  # ensemble is the reference

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = list(rmse_proxy.keys())
    vals = [rmse_proxy[k] for k in labels]
    colors_plot = [model_colors.get(k, "red") for k in labels]
    ax.bar(labels, vals, color=colors_plot, edgecolor="none", alpha=0.8)
    ax.set_title("Fig E3 — Mean Deviation from Ensemble (lower = closer to consensus)")
    ax.set_ylabel("RMSE vs ensemble mean")
    plt.tight_layout()
    save_fig(fig, "E3_model_deviation")
    plt.show()


def plot_prediction_distribution(
    ensemble_preds: np.ndarray,
    panel: pd.DataFrame,
) -> None:
    """
    Plot Set I1: Compare predicted quantity distribution (56-day forecast)
    against historical daily quantity distribution (post-cutoff).
    """
    set_plot_style()
    hist_vals = panel[panel["Date"] >= _CUTOFF]["net_qty"].clip(lower=0).values
    pred_vals = ensemble_preds.flatten()
    pred_vals = pred_vals[pred_vals > 0]

    clip99 = float(np.percentile(hist_vals[hist_vals > 0], 99))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Fig I1 — Prediction vs Historical Distribution", fontweight="bold")

    axes[0].hist(
        hist_vals[hist_vals > 0].clip(max=clip99),
        bins=80,
        color="steelblue",
        alpha=0.7,
        log=True,
        label="Historical (log Y)",
    )
    axes[0].hist(
        pred_vals.clip(max=clip99),
        bins=80,
        color="red",
        alpha=0.5,
        log=True,
        label="Predicted (log Y)",
    )
    axes[0].set_title("Positive values only, log-Y scale")
    axes[0].legend()

    axes[1].hist(
        hist_vals[hist_vals > 0].clip(max=clip99),
        bins=80,
        color="steelblue",
        alpha=0.7,
        label="Historical",
        density=True,
    )
    axes[1].hist(
        pred_vals.clip(max=clip99),
        bins=80,
        color="red",
        alpha=0.5,
        label="Predicted",
        density=True,
    )
    axes[1].set_title("Normalised density (non-zero values)")
    axes[1].legend()

    plt.tight_layout()
    save_fig(fig, "I1_prediction_distribution")
    plt.show()


# ==============================================================================
# 15 — MAIN PIPELINE EXECUTION
# ==============================================================================


def run_pipeline(
    stage1_artifacts: Dict,
    panel: pd.DataFrame,
    sub_raw: pd.DataFrame,
    run_cv_flag: bool = True,
    run_hparam_search: bool = False,
) -> Dict:
    """
    Execute the full Stages 2–6 pipeline.

    Parameters
    ----------
    stage1_artifacts   : dict returned by run_stage1()
    panel              : Stage 0 daily panel (for actuals and ETS fitting)
    sub_raw            : sample_submission.csv DataFrame
    run_cv_flag        : if True, run the full 5-fold CV before final training
    run_hparam_search  : if True, run Tweedie power grid search first

    Returns
    -------
    dict with keys:
        ctx, final_models, ensemble_preds, all_model_preds,
        processed_preds, submission, cv_results
    """
    # ── Setup ───────────────────────────────────────────────────────────────
    ctx = setup_stage2(stage1_artifacts)

    # ── Optional: Tweedie power search ──────────────────────────────────────
    if run_hparam_search:
        best_power = search_tweedie_power(ctx, week_idx=1)
        CONFIG["LGBM_A"]["tweedie_variance_power"] = best_power
        print(f"\n  Applying Tweedie power={best_power} to A models.")

    # ── CV (optional) ────────────────────────────────────────────────────────
    cv_results = None
    if run_cv_flag:
        cv_results = run_cv(ctx, panel)
        cv_results["oof_predictions"].to_parquet(
            _OUTPUT_DIR / "cv_oof_predictions.parquet", index=False
        )
        plot_cv_results(cv_results)

    # ── Final model training ─────────────────────────────────────────────────
    final_models = train_final_models(ctx, panel)

    # ── Training diagnostic plots ───────────────────────────────────────────
    plot_learning_curves(final_models)
    plot_feature_importance_comparison(final_models, ctx["week_feature_cols"])

    # ── Final inference ──────────────────────────────────────────────────────
    ensemble_preds, all_model_preds = run_final_inference(
        ctx, final_models
    )

    # ── Post-processing ──────────────────────────────────────────────────────
    print("\n--- Stage 6A: Post-processing ---")
    processed_preds = apply_post_processing(
        ensemble_preds,
        ctx["skus_sorted"],
        ctx["hard_zero_skus"],
        origin_date=_TRAIN_END,
    )

    # ── Inference diagnostic plots ───────────────────────────────────────────
    plot_prediction_distribution(processed_preds, panel)
    plot_horizon_profile(all_model_preds, ctx["skus_sorted"], ctx["sku_features"])
    plot_top_sku_traces(
        all_model_preds,
        processed_preds,
        ctx["skus_sorted"],
        ctx["sku_features"],
        panel,
        n_top=CONFIG["TOP_SKU_PLOTS"],
    )
    plot_ensemble_contribution(
        all_model_preds,
        processed_preds,
        ctx["skus_sorted"],
        ctx["sku_features"],
    )

    # ── Submission ───────────────────────────────────────────────────────────
    print("\n--- Stage 6B: Building submission ---")
    submission = build_submission(processed_preds, ctx["skus_sorted"], sub_raw)
    ok = validate_submission(submission, sub_raw)

    if ok:
        sub_path = _OUTPUT_DIR / "submission.csv"
        submission.to_csv(sub_path, index=False)
        print(f"  Submission saved → {sub_path}")
    else:
        print("  !! Submission validation FAILED — not saved. Fix issues above.")


    print("\n" + "=" * 55)
    print(" Pipeline Complete")
    print("=" * 55)
    if cv_results:
        print(
            f"  CV WRMSSE      : {cv_results['mean_wrmsse']:.4f} "
            f"± {cv_results['std_wrmsse']:.4f}"
        )
    print(f"  Naive baseline : {naive_results['mean_wrmsse']:.4f}")
    print(f"  Submission rows: {len(submission):,}")
    print(
        f"  Total predicted: {submission[[f'F{i}' for i in range(1, 29)]].values.sum():,.1f}"
    )
    print("=" * 55)

    return {
        "ctx": ctx,
        "final_models": final_models,
        "ensemble_preds": ensemble_preds,
        "all_model_preds": all_model_preds,
        "processed_preds": processed_preds,
        "submission": submission,
        "cv_results": cv_results,
    }


# ==============================================================================
# ENTRY POINT
# ==============================================================================
# In the Kaggle notebook, after running Stage 1, call:
#
#   if __name__ == "__main__":
#       artifacts = run_eda()
#       panel = artifacts["panel"]
#       sku_stats = artifacts["sku_stats"]
#       holiday_df = artifacts["holiday_df"]
#       sub_raw = artifacts["sub_raw"]
#       stage1_artifacts = run_stage1(panel, sku_stats, holiday_df)
#       pipeline_output = run_pipeline(
#       stage1_artifacts = stage1_artifacts,
#       panel            = panel,
#       sub_raw          = sub_raw,
#       run_cv_flag      = True,
#       run_hparam_search= False,    # set True for optional Tweedie power search
#   )
#
# Then unpack for convenience:
#   ctx            = pipeline_output["ctx"]
#   final_models   = pipeline_output["final_models"]
#   submission     = pipeline_output["submission"]
#   cv_results     = pipeline_output["cv_results"]


if __name__ == "__main__":
    artifacts = run_eda()
    panel = artifacts["panel"]
    folds = artifacts["folds"]
    naive_results = artifacts["naive_results"]
    sku_stats = artifacts["sku_stats"]
    sub_raw = artifacts["sub_raw"]
    stage1_artifacts = run_stage1(panel, sku_stats)
    pipeline_output = run_pipeline(
        stage1_artifacts=stage1_artifacts,
        panel=panel,
        sub_raw=sub_raw,
        run_cv_flag=True,
        run_hparam_search=True,  # set True for optional Tweedie power search
    )
