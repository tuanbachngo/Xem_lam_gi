"""
HBAAC 2026 — Vietnamese Auto Parts Demand Forecasting
---
MODIFICATIONS LOG:
1. CV Folds: Changed from 5 folds of 28-days to 4 folds of 56-days to perfectly mimic Public/Private LB split and validate long-horizon feature degradation.
2. Architecture simplification: Dropped Model C (Poisson) due to early stopping crashes, and dropped Model B (Recursive) due to compounding error drift on dead SKUs. 
3. Early Stopping Disabled: `eval_df=None` for A models so they fully fit the 1000 trees instead of prematurely stopping and underfitting, allowing use of 100% of the training history.
4. ETS Restrictions: Standard ETS is strictly constrained to `Dense` SKUs in blending. Intermittent/Sparse SKUs cannot use ETS to avoid massive multiplicative seasonality hallucinations.
5. Regularization: Removed `reg_alpha` and `reg_lambda` to allow the trees to fit more deeply on recent lags.
---
"""


"""
======================================================
HBAAC 2026 — Stage 0: Data Foundation + Comprehensive EDA
  0A  Raw data loading & parsing
  0B  Daily sales panel construction
  0C  Profit weight computation + SKU taxonomy
  0D  CV fold date derivation
  0E  EDA Layer 1 — Dataset-level diagnostics
  0F  EDA Layer 2 — SKU-level diagnostics
  0G  EDA Layer 3 — Temporal pattern analysis
  0H  EDA Layer 4 — Aggregation EDA
  0I  Naive seasonal baseline + WRMSSE floor score

Run environment: Kaggle CPU notebook (no GPU required for EDA).
All figures are saved to OUTPUT_DIR and displayed inline.
"""

# ==============================================================================
# 0 — IMPORTS
# ==============================================================================
import gc
import os
import random
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import kurtosis, skew
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")

# Print library versions for reproducibility
import matplotlib
import scipy
import statsmodels

print("=" * 55)
print("Library versions")
print("-" * 55)
print(f"  pandas       {pd.__version__}")
print(f"  numpy        {np.__version__}")
print(f"  matplotlib   {matplotlib.__version__}")
print(f"  seaborn      {sns.__version__}")
print(f"  scipy        {scipy.__version__}")
print(f"  statsmodels  {statsmodels.__version__}")
print("=" * 55)


# ==============================================================================
# 1 — CONFIGURATION
# ==============================================================================
CONFIG: Dict = {
    # ---- File paths ----
    "DATASET_DIR":   Path("/kaggle/input/datasets/ldatdzs1tg/hbaac-quantity-forecasting"),
    "OUTPUT_DIR":    Path("/kaggle/working/eda_outputs"),

    # ---- Inclusive date boundaries ----
    "TRAIN_START":        pd.Timestamp("2020-11-17"),
    "TRAIN_END":          pd.Timestamp("2025-09-05"),
    "PUBLIC_TEST_START":  pd.Timestamp("2025-09-06"),
    "PUBLIC_TEST_END":    pd.Timestamp("2025-10-03"),
    "PRIVATE_TEST_START": pd.Timestamp("2025-10-04"),
    "PRIVATE_TEST_END":   pd.Timestamp("2025-10-31"),

    # ---- Data quality ----
    # User-flagged: rows before this date appear anomalously sparse;
    # EDA will quantify this and flag affected SKUs.
    "EARLY_DATA_CUTOFF": pd.Timestamp("2022-01-01"),

    # SKU whose last observed sale is this many days before TRAIN_END
    # is flagged as potentially discontinued.
    "DISCONTINUED_DAYS": 90,

    # ---- SKU demand-density taxonomy ----
    "ZERO_RATE_SPARSE":        0.90,   # >90% zero-days  → Sparse
    "ZERO_RATE_INTERMITTENT":  0.55,   # 55–90%          → Intermittent
    #                                   <55%              → Dense

    # ---- SKU profit-tier taxonomy ----
    # Cumulative-weight cut-offs (fraction of total WRMSSE weight)
    "PROFIT_TIER_A_CUM": 0.50,   # top SKUs → first 50 % of weight
    "PROFIT_TIER_B_CUM": 0.80,   # next     → up to 80 %
    "PROFIT_TIER_C_CUM": 0.95,   # next     → up to 95 %
    #                              rest      → D-tier (near-zero weight)

    # ---- CV ----
    "HORIZON_DAYS": 56,          # each fold's validation window length
    "N_FOLDS":       4,          # 3 recent consecutive + 1 early anchor

    # ---- Visualisation ----
    "SEED":           42,
    "PLOT_DPI":       130,
    "TOP_SKU_PLOTS":  20,        # individual time-series plots for top-N SKUs
    "FIGSIZE_WIDE":  (18, 5),
    "FIGSIZE_TALL":  (14, 9),
    "FIGSIZE_SQ":    (12, 10),
}

# Reproducibility
random.seed(CONFIG["SEED"])
np.random.seed(CONFIG["SEED"])

CONFIG["OUTPUT_DIR"].mkdir(parents=True, exist_ok=True)


# Tết (Lunar New Year): first day in Gregorian calendar
_TET_DATES: List[str] = [
    "2021-02-12", "2022-02-01", "2023-01-22",
    "2024-02-10", "2025-01-29",
]


# ==============================================================================
# 2 — UTILITY FUNCTIONS
# ==============================================================================

def set_plot_style() -> None:
    """Apply a consistent, clean Matplotlib style for all EDA figures."""
    plt.rcParams.update({
        "figure.dpi":       CONFIG["PLOT_DPI"],
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "axes.grid":        True,
        "grid.alpha":       0.35,
        "font.size":        11,
        "axes.titlesize":   13,
        "axes.labelsize":   11,
    })
    sns.set_palette("tab10")


def save_fig(fig: plt.Figure, name: str) -> None:
    """Save a figure to OUTPUT_DIR with a standardised filename."""
    path = CONFIG["OUTPUT_DIR"] / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=CONFIG["PLOT_DPI"])
    print(f"  [saved] {path}")


def parse_vnd_price(series: pd.Series) -> pd.Series:
    """
    Parse Vietnamese VND price/cost strings to float64.

    Vietnamese number formatting:
      Period (.)  = thousands separator  →  1.500.000  =  1,500,000
      Comma  (,)  = decimal separator    →  1.500,50   =  1,500.50

    Handles edge cases: empty strings, '-', NaN, plain integers.

    Parameters
    ----------
    series : pd.Series of raw string values from UnitPrice / Unit Cost columns.

    Returns
    -------
    pd.Series of float64.
    """
    def _parse(val: object) -> float:
        if pd.isna(val):
            return np.nan
        s = str(val).strip().replace(" ", "")
        if s in ("", "-", "N/A", "n/a"):
            return np.nan
        if "," in s and "." in s:
            # European/Vietnamese mixed format: "1.500,50"
            # → remove thousands sep (.), swap decimal (,) to (.)
            s = s.replace(".", "").replace(",", ".")
        elif "," in s:
            # Comma only: decide between decimal vs thousands separator.
            # Heuristic: if digits after the LAST comma ≤ 2 → decimal
            parts = s.split(",")
            if len(parts[-1]) <= 2:
                s = s.replace(",", ".")   # treat as decimal
            else:
                s = s.replace(",", "")    # treat as thousands
        elif "." in s:
            # Period only: if ALL right-side groups after splitting are 3 digits
            # it is a thousands separator, otherwise decimal.
            parts = s.split(".")
            if len(parts) > 1 and all(len(p) == 3 for p in parts[1:]):
                s = s.replace(".", "")    # thousands separator
            # else: leave period as decimal point
        try:
            return float(s)
        except ValueError:
            return np.nan

    return series.apply(_parse)


def build_vn_holiday_series(
    date_range: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Build a per-day holiday-feature DataFrame for a given date range.

    Columns
    -------
    days_to_tet   : int    — signed distance to nearest Tết first-day
                             (negative = before Tết, positive = after)
    tet_window    : bool   — True for ±3 days around Tết first-day
    """
    tet_ts = [pd.Timestamp(d) for d in _TET_DATES]
    tet_window   = pd.Series(False,  index=date_range, dtype=bool)

    # Tết: 1 day before + first day + 3 days after (5-day window)
    for t in tet_ts:
        for offset in range(-1, 4):
            d = t + timedelta(days=offset)
            if d in date_range:
                tet_window[d]   = True

    # Signed distance to nearest Tết
    def _dist_to_tet(d: pd.Timestamp) -> int:
        diffs = [(d - t).days for t in tet_ts]
        return min(diffs, key=abs)

    days_to_tet = pd.Series(
        {d: _dist_to_tet(d) for d in date_range},
        dtype=int,
    )

    return pd.DataFrame({
        "days_to_tet":  days_to_tet.values,
        "tet_window":   tet_window.values,
    }, index=date_range)


def compute_cv_fold_dates(
    train_start: pd.Timestamp,
    train_end:   pd.Timestamp,
    horizon:     int,
    n_folds:     int = 5,
    anchor_gap_days: int = 365,
) -> List[Dict[str, pd.Timestamp]]:
    """
    Derive CV fold boundary dates from actual training data dates.

    Structure
    ---------
    Folds n_folds-1 down to 1 (recent): consecutive non-overlapping
      28-day validation windows counting back from train_end.
    Fold 0 (early anchor): validation window ending ~anchor_gap_days
      before the second fold's validation start.

    Parameters
    ----------
    train_start      : first day of training data
    train_end        : last day of training data
    horizon          : validation window length in days (28)
    n_folds          : total number of folds (5)
    anchor_gap_days  : days gap between anchor fold and fold 1 (≈ 1 year)

    Returns
    -------
    List of dicts, each with keys:
        fold_id, train_start, train_end, val_start, val_end
    Ordered from earliest to latest validation window.
    """
    folds = []

    # --- Recent folds (folds 1 to n_folds-1, counting backwards from train_end)
    recent_n = n_folds - 1   # = 4
    val_end = train_end
    for i in range(recent_n):
        val_start = val_end - timedelta(days=horizon - 1)
        fold_train_end = val_start - timedelta(days=1)
        folds.append({
            "fold_id":     recent_n - i,      # labels: 4, 3, 2, 1
            "train_start": train_start,
            "train_end":   fold_train_end,
            "val_start":   val_start,
            "val_end":     val_end,
        })
        val_end = fold_train_end  # walk backward

    folds.reverse()   # now ordered fold 1 → 4 (oldest val first)
    # Re-label so fold 1 is the earliest recent fold
    for i, f in enumerate(folds):
        f["fold_id"] = i + 1

    # --- Early anchor fold (fold 0)
    # Place its validation window ~anchor_gap_days before fold 1 val_start
    anchor_val_end   = folds[0]["val_start"] - timedelta(days=anchor_gap_days)
    anchor_val_start = anchor_val_end - timedelta(days=horizon - 1)
    anchor_train_end = anchor_val_start - timedelta(days=1)

    # Safety: anchor must start after training start
    if anchor_train_end > train_start + timedelta(days=horizon * 4):
        folds.insert(0, {
            "fold_id":     0,
            "train_start": train_start,
            "train_end":   anchor_train_end,
            "val_start":   anchor_val_start,
            "val_end":     anchor_val_end,
        })
    else:
        print("  [WARNING] Early anchor fold skipped: insufficient history.")

    return folds


def compute_naive_rmsse_denominator(
    series: np.ndarray,
) -> float:
    """
    Compute the RMSSE naive denominator for one SKU's training series.

    Denominator = mean squared error of the naive lag-1 forecast on training.
    Formula: (1 / (n-1)) * sum_{t=2}^{n} (Y_t - Y_{t-1})^2

    Returns 0.0 if the series has < 2 non-NaN points (edge case: constant or
    all-zero series). Callers must handle the zero-denominator case.
    """
    s = series[~np.isnan(series)]
    if len(s) < 2:
        return 0.0
    diffs_sq = np.diff(s) ** 2
    return float(np.mean(diffs_sq))


def compute_wrmsse(
    actuals:    pd.DataFrame,   # shape (h, n_skus), columns = ItemCode
    forecasts:  pd.DataFrame,   # same shape and columns
    train_qtys: pd.DataFrame,   # shape (n_train_days, n_skus)
    weights:    pd.Series,      # index = ItemCode, values = profit weight
) -> Tuple[float, pd.Series]:
    """
    Compute WRMSSE and per-SKU RMSSE.

    Parameters
    ----------
    actuals    : actual net quantities over the forecast horizon
    forecasts  : predicted quantities (non-negative not enforced here)
    train_qtys : historical daily net quantities (for denominator)
    weights    : profit weights per SKU (sum to 1, negatives already zeroed)

    Returns
    -------
    wrmsse  : float scalar
    rmsse   : pd.Series of per-SKU RMSSE (index = ItemCode)
    """
    skus = actuals.columns
    h    = len(actuals)
    rmsse_vals = {}

    for sku in skus:
        y_true  = actuals[sku].values.astype(float)
        y_pred  = forecasts[sku].values.astype(float)
        y_train = train_qtys[sku].values.astype(float) if sku in train_qtys else np.array([])

        # Numerator: mean squared forecast error over horizon
        mse_forecast = np.mean((y_true - y_pred) ** 2)

        # Denominator: naive lag-1 MSE on training data
        denom = compute_naive_rmsse_denominator(y_train)

        if denom == 0.0:
            # Series is constant or too short; RMSSE is undefined.
            # Weight should be 0 for such SKUs (zero profit), so RMSSE
            # value doesn't affect WRMSSE — assign 0 to avoid inf.
            rmsse_vals[sku] = 0.0
        else:
            rmsse_vals[sku] = float(np.sqrt(mse_forecast / denom))

    rmsse  = pd.Series(rmsse_vals)
    w      = weights.reindex(skus).fillna(0.0)
    wrmsse = float((rmsse * w).sum())
    return wrmsse, rmsse


# ==============================================================================
# 3 — DATA LOADING & PARSING  (Stage 0A)
# ==============================================================================

def load_raw_data(dataset_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train.csv and sample_submission.csv from dataset_dir.

    Handles two common Kaggle layouts:
      1. Files directly in dataset_dir
      2. Files inside a single sub-folder of dataset_dir

    Returns
    -------
    train_raw        : raw train DataFrame (no type casting yet)
    submission_raw   : sample submission DataFrame
    """
    def _find_file(root: Path, name: str) -> Path:
        direct = root / name
        if direct.exists():
            return direct
        # Search one level deeper
        for child in root.iterdir():
            if child.is_dir():
                candidate = child / name
                if candidate.exists():
                    return candidate
        raise FileNotFoundError(
            f"Cannot find '{name}' in {root} or its immediate subdirectories."
        )

    train_path = _find_file(dataset_dir, "train.csv")
    sub_path   = _find_file(dataset_dir, "sample_submission.csv")

    print(f"Loading train from : {train_path}")
    train_raw = pd.read_csv(train_path, dtype=str, low_memory=False)

    print(f"Loading submission : {sub_path}")
    sub_raw = pd.read_csv(sub_path, low_memory=False)

    return train_raw, sub_raw


def parse_train(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Parse and type-cast the raw train DataFrame.

    Operations
    ----------
    1. Parse 'Date' column to datetime.
    2. Cast 'Stt' to int (row sequence, not used in modelling).
    3. Cast 'Quantity' and 'SalesAmount' and 'Cost Amount' to int64.
    4. Parse 'UnitPrice' and 'Unit Cost' from VND string format to float64.
    5. Derive 'Profit' = SalesAmount - Cost Amount per row.
    6. Flag rows with negative Quantity as return transactions.

    Returns
    -------
    Parsed DataFrame with added columns:
        profit        : float64  per-row profit (SalesAmount - Cost Amount)
        is_return     : bool     True when Quantity < 0
    """
    df = df_raw.copy()

    # --- Date ---
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    n_bad_dates = df["Date"].isna().sum()
    if n_bad_dates > 0:
        print(f"  [WARNING] {n_bad_dates} rows with unparseable Date — dropped.")
    df = df.dropna(subset=["Date"])

    # --- Numeric columns ---
    df["Stt"]         = pd.to_numeric(df["Stt"],         errors="coerce")
    df["Quantity"]    = pd.to_numeric(df["Quantity"],    errors="coerce").astype("Int64")
    df["SalesAmount"] = pd.to_numeric(df["SalesAmount"], errors="coerce").astype("Int64")
    df["Cost Amount"] = pd.to_numeric(df["Cost Amount"], errors="coerce").astype("Int64")

    # --- VND price strings ---
    df["UnitPrice"] = parse_vnd_price(df["UnitPrice"])
    df["Unit Cost"] = parse_vnd_price(df["Unit Cost"])

    # --- Derived columns ---
    df["profit"]    = (df["SalesAmount"].astype(float)
                       - df["Cost Amount"].astype(float))
    df["is_return"] = df["Quantity"].astype(float) < 0

    print(
        f"  Parsed {len(df):,} rows | "
        f"date range: {df['Date'].min().date()} → {df['Date'].max().date()} | "
        f"unique SKUs: {df['ItemCode'].nunique():,}"
    )
    return df


# ==============================================================================
# 4 — DAILY PANEL CONSTRUCTION  (Stage 0B)
# ==============================================================================

def build_daily_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transaction-level data to a daily (Date × ItemCode) panel.

    Aggregation rules
    -----------------
    - net_qty          : sum(Quantity)  — returns net out automatically.
                         Can be negative for a given day if returns > sales.
                         *** Do NOT clip here. Clip only at submission. ***
    - gross_qty        : sum(Quantity[Quantity > 0])  — gross sales quantity
    - return_qty       : abs(sum(Quantity[Quantity < 0]))  — returned quantity
    - daily_sales      : sum(SalesAmount)
    - daily_cost       : sum(Cost Amount)
    - daily_profit     : sum(profit)
    - mean_unit_price  : daily_sales / gross_qty  (NaN if no gross sales)
    - n_transactions   : number of transaction lines

    Missing (Date, ItemCode) combinations are filled with 0 for quantities
    and NaN for price (no transaction = no observable price that day).

    Returns
    -------
    pd.DataFrame indexed by (Date, ItemCode) with the aggregated columns.
    """
    # Separate positive and negative rows before groupby for efficiency
    pos = df[df["Quantity"].astype(float) > 0]
    neg = df[df["Quantity"].astype(float) < 0]

    agg = df.groupby(["Date", "ItemCode"], observed=True).agg(
        net_qty       =("Quantity",    "sum"),
        daily_sales   =("SalesAmount", "sum"),
        daily_cost    =("Cost Amount", "sum"),
        daily_profit  =("profit",      "sum"),
        n_transactions=("Stt",         "count"),
    ).reset_index()

    gross = (
        pos.groupby(["Date", "ItemCode"], observed=True)["Quantity"]
        .sum().reset_index().rename(columns={"Quantity": "gross_qty"})
    )
    returns_ = (
        neg.groupby(["Date", "ItemCode"], observed=True)["Quantity"]
        .sum().abs().reset_index().rename(columns={"Quantity": "return_qty"})
    )

    agg = agg.merge(gross,    on=["Date", "ItemCode"], how="left")
    agg = agg.merge(returns_, on=["Date", "ItemCode"], how="left")
    agg["gross_qty"]  = agg["gross_qty"].fillna(0)
    agg["return_qty"] = agg["return_qty"].fillna(0)

    # Mean unit price from gross sales only (avoid division by return rows)
    agg["mean_unit_price"] = np.where(
        agg["gross_qty"] > 0,
        agg["daily_sales"].astype(float) / agg["gross_qty"].astype(float),
        np.nan,
    )

    # Cast tidy types
    int_cols = ["net_qty", "gross_qty", "return_qty",
                "daily_sales", "daily_cost", "daily_profit", "n_transactions"]
    for c in int_cols:
        agg[c] = agg[c].astype(float)   # keep float for downstream math

    agg = agg.sort_values(["ItemCode", "Date"]).reset_index(drop=True)

    # ---- Create a COMPLETE (Date, ItemCode) calendar grid ----
    # This ensures every SKU has an entry for every calendar day, even if
    # it was inactive (fill qty=0, price=NaN).
    all_dates = pd.date_range(
        CONFIG["TRAIN_START"], CONFIG["TRAIN_END"], freq="D"
    )
    all_skus  = agg["ItemCode"].unique()
    idx       = pd.MultiIndex.from_product(
        [all_dates, all_skus], names=["Date", "ItemCode"]
    )
    panel = (
        agg.set_index(["Date", "ItemCode"])
           .reindex(idx, fill_value=0)
           .reset_index()
    )
    # Restore NaN for price on zero-transaction days (was filled with 0 by reindex)
    panel.loc[panel["n_transactions"] == 0, "mean_unit_price"] = np.nan

    print(
        f"  Daily panel: {len(panel):,} rows | "
        f"{len(all_dates)} dates × {len(all_skus):,} SKUs"
    )
    return panel


# ==============================================================================
# 5 — PROFIT WEIGHTS & SKU TAXONOMY  (Stage 0C)
# ==============================================================================

def compute_sku_stats(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-SKU statistics used for profit weights and taxonomy.

    Columns in output
    -----------------
    ItemCode
    cumulative_profit     : total profit over entire training set (can be negative)
    profit_weight         : max(0, profit) / sum_j(max(0, profit_j))
    cum_weight_rank       : rank by profit_weight descending (1 = highest weight)
    profit_tier           : A / B / C / D  based on cumulative weight cut-offs
    zero_rate             : fraction of calendar days with net_qty <= 0
    demand_density        : Dense / Intermittent / Sparse
    historical_mean_qty   : mean daily net_qty over training period
    historical_std_qty    : std  daily net_qty
    historical_max_qty    : max  daily net_qty
    qty_skewness          : skewness of daily net_qty distribution
    qty_kurtosis          : excess kurtosis
    first_sale_date       : first date with net_qty > 0
    last_sale_date        : last  date with net_qty > 0
    active_days           : number of days with net_qty > 0
    is_discontinued       : True if last_sale_date < TRAIN_END - DISCONTINUED_DAYS
    days_since_first_sale : calendar days from first_sale_date to TRAIN_END
    mean_unit_price_global: time-averaged unit price (weighted by gross_qty)
    mean_margin_rate      : mean (profit / sales) where sales > 0
    has_pre_cutoff_only   : True if ALL sales occurred before EARLY_DATA_CUTOFF
    """
    g = panel.groupby("ItemCode", observed=True)

    stats_df = pd.DataFrame()
    stats_df["cumulative_profit"]  = g["daily_profit"].sum()
    stats_df["active_days"]        = (panel[panel["net_qty"] > 0]
                                       .groupby("ItemCode", observed=True)
                                       .size())
    stats_df["active_days"]        = stats_df["active_days"].fillna(0)
    stats_df["historical_mean_qty"]= g["net_qty"].mean()
    stats_df["historical_std_qty"] = g["net_qty"].std()
    stats_df["historical_max_qty"] = g["net_qty"].max()

    # Skewness/kurtosis on non-zero days to capture demand spikes
    def _skew(x):  return float(skew(x[x > 0])) if (x > 0).sum() > 3 else np.nan
    def _kurt(x):  return float(kurtosis(x[x > 0])) if (x > 0).sum() > 3 else np.nan
    stats_df["qty_skewness"] = g["net_qty"].apply(_skew)
    stats_df["qty_kurtosis"] = g["net_qty"].apply(_kurt)

    # Zero rate: fraction of ALL calendar days with net_qty <= 0
    n_days = panel["Date"].nunique()
    stats_df["zero_rate"] = 1.0 - stats_df["active_days"] / n_days

    # First / last sale dates
    pos_panel = panel[panel["net_qty"] > 0]
    stats_df["first_sale_date"] = pos_panel.groupby("ItemCode", observed=True)["Date"].min()
    stats_df["last_sale_date"]  = pos_panel.groupby("ItemCode", observed=True)["Date"].max()

    stats_df["is_discontinued"] = (
        stats_df["last_sale_date"]
        < CONFIG["TRAIN_END"] - timedelta(days=CONFIG["DISCONTINUED_DAYS"])
    )
    stats_df["days_since_first_sale"] = (
        CONFIG["TRAIN_END"] - stats_df["first_sale_date"]
    ).dt.days

    # Mean unit price (weighted average)
    def _wavg_price(grp):
        w = grp["gross_qty"]
        p = grp["mean_unit_price"]
        mask = w > 0
        return (p[mask] * w[mask]).sum() / w[mask].sum() if mask.sum() > 0 else np.nan
    stats_df["mean_unit_price_global"] = panel.groupby(
        "ItemCode", observed=True
    ).apply(_wavg_price)

    # Mean margin rate
    def _margin(grp):
        mask = grp["daily_sales"] > 0
        if mask.sum() == 0:
            return np.nan
        return (grp.loc[mask, "daily_profit"] / grp.loc[mask, "daily_sales"]).mean()
    stats_df["mean_margin_rate"] = panel.groupby(
        "ItemCode", observed=True
    ).apply(_margin)

    # Flag SKUs with ALL activity before the early-data cutoff
    post_cutoff_active = (
        panel[panel["Date"] >= CONFIG["EARLY_DATA_CUTOFF"]]
        .groupby("ItemCode", observed=True)["net_qty"]
        .sum()
    )
    stats_df["has_pre_cutoff_only"] = ~stats_df.index.isin(
        post_cutoff_active[post_cutoff_active > 0].index
    )

    stats_df = stats_df.reset_index()

    # ---- Profit weights ----
    clipped_profit = stats_df["cumulative_profit"].clip(lower=0)
    total_weight   = clipped_profit.sum()
    stats_df["profit_weight"] = (
        clipped_profit / total_weight if total_weight > 0 else 0.0
    )

    # ---- Profit tier via cumulative weight ----
    stats_df = stats_df.sort_values("profit_weight", ascending=False).reset_index(drop=True)
    stats_df["cum_weight_rank"] = stats_df.index + 1
    cum_w = stats_df["profit_weight"].cumsum()
    conditions = [
        cum_w.shift(1, fill_value=0) < CONFIG["PROFIT_TIER_A_CUM"],
        cum_w.shift(1, fill_value=0) < CONFIG["PROFIT_TIER_B_CUM"],
        cum_w.shift(1, fill_value=0) < CONFIG["PROFIT_TIER_C_CUM"],
    ]
    choices = ["A", "B", "C"]
    stats_df["profit_tier"] = np.select(conditions, choices, default="D")

    # ---- Demand density ----
    stats_df["demand_density"] = pd.cut(
        stats_df["zero_rate"],
        bins=[-0.001,
              CONFIG["ZERO_RATE_INTERMITTENT"],
              CONFIG["ZERO_RATE_SPARSE"],
              1.001],
        labels=["Dense", "Intermittent", "Sparse"],
    )

    print(
        f"  SKU stats computed for {len(stats_df):,} SKUs | "
        f"total WRMSSE weight (sanity sum): {stats_df['profit_weight'].sum():.4f}"
    )
    return stats_df


# ==============================================================================
# 6 — EDA LAYER 1: DATASET-LEVEL DIAGNOSTICS  (Stage 0E)
# ==============================================================================

def eda_layer1_dataset(df: pd.DataFrame, panel: pd.DataFrame) -> None:
    """
    Layer 1 EDA: transaction-level and daily-panel-level diagnostics.

    Plots
    -----
    1A  Quantity distribution (log-scale histogram)
    1B  Daily transaction count over time (with EARLY_DATA_CUTOFF marked)
    1C  Return transaction fraction over time
    1D  Early-data sparsity investigation — pre vs post cutoff monthly volume
    """
    set_plot_style()
    print("\n=== EDA Layer 1: Dataset-Level ===")

    # ---- 1A: Quantity distribution ----
    fig, axes = plt.subplots(1, 2, figsize=CONFIG["FIGSIZE_WIDE"])
    fig.suptitle("Fig 1A — Transaction Quantity Distribution", fontweight="bold")

    qty = df["Quantity"].astype(float)

    ax = axes[0]
    ax.hist(qty[qty >= 0], bins=100, log=True, color="steelblue", edgecolor="none", alpha=0.8)
    ax.set_title("Gross Sale Quantities (log Y)")
    ax.set_xlabel("Quantity per transaction line")
    ax.set_ylabel("Count (log scale)")

    ax = axes[1]
    ax.hist(qty[qty < 0].abs(), bins=60, log=True, color="tomato", edgecolor="none", alpha=0.8)
    ax.set_title("Return Quantities — absolute value (log Y)")
    ax.set_xlabel("Absolute return quantity")

    plt.tight_layout()
    save_fig(fig, "1A_qty_distribution")
    plt.show()

    # Summary stats
    print(f"\n  Gross sales rows  : {(qty > 0).sum():,}")
    print(f"  Return rows       : {(qty < 0).sum():,}")
    print(f"  Zero-qty rows     : {(qty == 0).sum():,}")
    print(f"  Return row %      : {100*(qty<0).mean():.2f}%")
    print(f"  Gross qty — mean={qty[qty>0].mean():.1f}, "
          f"median={qty[qty>0].median():.0f}, "
          f"max={qty[qty>0].max():.0f}, "
          f"skew={skew(qty[qty>0].dropna()):.2f}")

    # ---- 1B: Daily transaction count over time ----
    daily_tx = df.groupby("Date").size().reset_index(name="n_transactions")
    daily_tx["Date"] = pd.to_datetime(daily_tx["Date"])

    fig, ax = plt.subplots(figsize=CONFIG["FIGSIZE_WIDE"])
    ax.plot(daily_tx["Date"], daily_tx["n_transactions"],
            lw=0.7, color="steelblue", alpha=0.6, label="Daily tx count")
    ax.plot(daily_tx["Date"],
            daily_tx["n_transactions"].rolling(30).mean(),
            lw=2, color="navy", label="30-day rolling mean")

    # Mark early-data cutoff
    ax.axvline(CONFIG["EARLY_DATA_CUTOFF"], color="red", lw=2, ls="--",
               label=f"Early-data cutoff ({CONFIG['EARLY_DATA_CUTOFF'].date()})")
    ax.set_title("Fig 1B — Daily Transaction Count Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of transaction rows")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    save_fig(fig, "1B_daily_tx_count")
    plt.show()

    # ---- 1C: Monthly return fraction ----
    df_tmp = df.copy()
    df_tmp["month"] = df_tmp["Date"].dt.to_period("M")
    monthly = df_tmp.groupby("month").agg(
        total_rows   =("Quantity", "count"),
        return_rows  =("is_return", "sum"),
    ).reset_index()
    monthly["return_frac"] = monthly["return_rows"] / monthly["total_rows"]
    monthly["month_dt"]    = monthly["month"].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=CONFIG["FIGSIZE_WIDE"])
    ax.bar(monthly["month_dt"], monthly["return_frac"],
           width=20, color="tomato", alpha=0.75, label="Return fraction")
    ax.axvline(CONFIG["EARLY_DATA_CUTOFF"], color="red", lw=2, ls="--",
               label="Early-data cutoff")
    ax.set_title("Fig 1C — Monthly Return Transaction Fraction")
    ax.set_xlabel("Month")
    ax.set_ylabel("Return rows / total rows")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend()
    plt.tight_layout()
    save_fig(fig, "1C_return_fraction")
    plt.show()

    # ---- 1D: Early-data sparsity investigation ----
    # Compare monthly total net_qty and unique active SKUs pre vs post cutoff
    panel_tmp = panel.copy()
    panel_tmp["month"] = panel_tmp["Date"].dt.to_period("M")
    monthly_panel = panel_tmp.groupby("month").agg(
        total_net_qty  =("net_qty",    "sum"),
        active_skus    =("ItemCode",   lambda x: (panel_tmp.loc[x.index, "net_qty"] > 0).sum()),
    ).reset_index()
    monthly_panel["month_dt"] = monthly_panel["month"].dt.to_timestamp()

    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
    fig.suptitle("Fig 1D — Early Data Sparsity Investigation", fontweight="bold")

    cutoff_line_kw = dict(color="red", lw=2, ls="--",
                          label=f"Cutoff {CONFIG['EARLY_DATA_CUTOFF'].date()}")

    ax = axes[0]
    ax.bar(monthly_panel["month_dt"], monthly_panel["total_net_qty"],
           width=20, color="steelblue", alpha=0.75)
    ax.axvline(CONFIG["EARLY_DATA_CUTOFF"], **cutoff_line_kw)
    ax.set_ylabel("Total net qty (all SKUs)")
    ax.set_title("Monthly Aggregate Demand Volume")
    ax.legend()

    ax = axes[1]
    ax.bar(monthly_panel["month_dt"], monthly_panel["active_skus"],
           width=20, color="darkorange", alpha=0.75)
    ax.axvline(CONFIG["EARLY_DATA_CUTOFF"], **cutoff_line_kw)
    ax.set_ylabel("Active SKUs per month")
    ax.set_title("Monthly Active SKU Count")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=30)
    ax.legend()

    plt.tight_layout()
    save_fig(fig, "1D_early_data_sparsity")
    plt.show()

    # Print quantitative comparison
    pre  = panel_tmp[panel_tmp["Date"] < CONFIG["EARLY_DATA_CUTOFF"]]
    post = panel_tmp[panel_tmp["Date"] >= CONFIG["EARLY_DATA_CUTOFF"]]
    print(f"\n  Pre-cutoff period  ({CONFIG['TRAIN_START'].date()} → "
          f"{CONFIG['EARLY_DATA_CUTOFF'].date() }): "
          f"avg {pre.groupby('Date')['n_transactions'].sum().mean():.0f} tx/day")
    print(f"  Post-cutoff period ({CONFIG['EARLY_DATA_CUTOFF'].date()} → "
          f"{CONFIG['TRAIN_END'].date()}): "
          f"avg {post.groupby('Date')['n_transactions'].sum().mean():.0f} tx/day")
    del df_tmp, panel_tmp
    gc.collect()


# ==============================================================================
# 7 — EDA LAYER 2: SKU-LEVEL DIAGNOSTICS  (Stage 0F)
# ==============================================================================

def eda_layer2_sku(sku_stats: pd.DataFrame) -> None:
    """
    Layer 2 EDA: per-SKU distributions, profit concentration, taxonomy.

    Plots
    -----
    2A  Lorenz curve of profit weight concentration
    2B  Top-50 SKUs by profit weight (bar chart)
    2C  Zero-rate (demand sparsity) distribution across all SKUs
    2D  SKU taxonomy heatmap: profit_tier × demand_density cell counts
    2E  SKU lifespan scatter: first_sale_date vs last_sale_date
    2F  Mean unit price distribution (log scale) — price tier overview
    """
    set_plot_style()
    print("\n=== EDA Layer 2: SKU-Level ===")

    # Print taxonomy summary
    tier_summary = sku_stats.groupby(
        ["profit_tier", "demand_density"], observed=True
    ).agg(
        n_skus        =("ItemCode",       "count"),
        total_weight  =("profit_weight",  "sum"),
    ).reset_index()
    print("\n  Profit Tier × Demand Density cell summary:")
    print(tier_summary.to_string(index=False))

    n_discontinued = sku_stats["is_discontinued"].sum()
    n_neg_profit   = (sku_stats["cumulative_profit"] < 0).sum()
    print(f"\n  SKUs with negative cumulative profit (weight=0): {n_neg_profit:,}")
    print(f"  SKUs flagged as discontinued (no sales in last "
          f"{CONFIG['DISCONTINUED_DAYS']} days): {n_discontinued:,}")

    # ---- 2A: Lorenz curve ----
    sorted_w  = np.sort(sku_stats["profit_weight"].values)
    cum_w     = np.cumsum(sorted_w)
    cum_w     = cum_w / cum_w[-1]
    cum_skus  = np.arange(1, len(cum_w) + 1) / len(cum_w)

    gini = 1 - 2 * np.trapz(cum_w, cum_skus)  # Gini coefficient

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(cum_skus * 100, cum_w * 100, color="steelblue", lw=2.5,
            label=f"Profit weight (Gini={gini:.3f})")
    ax.plot([0, 100], [0, 100], "k--", lw=1, alpha=0.5, label="Perfect equality")

    # Mark 50/80/95% weight thresholds
    for pct, color, tier in [(50, "orange", "A"), (80, "red", "B"), (95, "purple", "C")]:
        idx = np.searchsorted(cum_w, pct / 100)
        sku_pct = cum_skus[min(idx, len(cum_skus)-1)] * 100
        ax.axhline(pct, color=color, ls=":", lw=1.2, alpha=0.7)
        ax.axvline(sku_pct, color=color, ls=":", lw=1.2, alpha=0.7,
                   label=f"Tier {tier} cutoff: {sku_pct:.1f}% of SKUs → {pct}% weight")

    ax.set_xlabel("Cumulative % of SKUs (sorted by weight ascending)")
    ax.set_ylabel("Cumulative % of WRMSSE weight")
    ax.set_title("Fig 2A — Lorenz Curve of Profit Weight Concentration")
    ax.legend(fontsize=9)
    plt.tight_layout()
    save_fig(fig, "2A_lorenz_curve")
    plt.show()

    # ---- 2B: Top 50 SKUs ----
    top50 = sku_stats.nlargest(50, "profit_weight")
    fig, ax = plt.subplots(figsize=(18, 6))
    colors = {"A": "steelblue", "B": "darkorange", "C": "green", "D": "gray"}
    bar_colors = [colors[t] for t in top50["profit_tier"]]
    ax.bar(range(50), top50["profit_weight"] * 100, color=bar_colors, edgecolor="none")
    ax.set_xticks(range(50))
    ax.set_xticklabels(top50["ItemCode"], rotation=90, fontsize=7)
    ax.set_ylabel("WRMSSE Weight (%)")
    ax.set_title("Fig 2B — Top 50 SKUs by Profit Weight")
    from matplotlib.patches import Patch
    legend_elements = [Patch(fc=v, label=f"Tier {k}") for k, v in colors.items()]
    ax.legend(handles=legend_elements)
    plt.tight_layout()
    save_fig(fig, "2B_top50_weights")
    plt.show()

    # ---- 2C: Zero-rate distribution ----
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(sku_stats["zero_rate"], bins=50, color="steelblue",
            edgecolor="none", alpha=0.8)
    ax.axvline(CONFIG["ZERO_RATE_INTERMITTENT"], color="orange", lw=2,
               label=f"Dense|Intermittent ({int(CONFIG['ZERO_RATE_INTERMITTENT']*100)}%)")
    ax.axvline(CONFIG["ZERO_RATE_SPARSE"], color="red", lw=2,
               label=f"Intermittent|Sparse ({int(CONFIG['ZERO_RATE_SPARSE']*100)}%)")
    ax.set_xlabel("Zero-sales day fraction")
    ax.set_ylabel("Number of SKUs")
    ax.set_title("Fig 2C — Demand Density Distribution (Zero-Rate)")
    ax.legend()
    plt.tight_layout()
    save_fig(fig, "2C_zero_rate_distribution")
    plt.show()
    print(f"  Dense SKUs        : "
          f"{(sku_stats['demand_density']=='Dense').sum():,}")
    print(f"  Intermittent SKUs : "
          f"{(sku_stats['demand_density']=='Intermittent').sum():,}")
    print(f"  Sparse SKUs       : "
          f"{(sku_stats['demand_density']=='Sparse').sum():,}")

    # ---- 2D: Taxonomy heatmap ----
    heatmap_data = sku_stats.pivot_table(
        index="profit_tier",
        columns="demand_density",
        values="ItemCode",
        aggfunc="count",
        fill_value=0,
        observed=True,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        heatmap_data,
        annot=True, fmt="d", cmap="Blues",
        linewidths=0.5, ax=ax, cbar_kws={"label": "SKU count"},
    )
    ax.set_title("Fig 2D — SKU Taxonomy: Profit Tier × Demand Density (counts)")
    plt.tight_layout()
    save_fig(fig, "2D_taxonomy_heatmap")
    plt.show()

    # Weight version of heatmap
    heatmap_wt = sku_stats.pivot_table(
        index="profit_tier",
        columns="demand_density",
        values="profit_weight",
        aggfunc="sum",
        fill_value=0.0,
        observed=True,
    ) * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        heatmap_wt,
        annot=True, fmt=".2f", cmap="Oranges",
        linewidths=0.5, ax=ax, cbar_kws={"label": "Cumulative WRMSSE weight (%)"},
    )
    ax.set_title("Fig 2D-W — SKU Taxonomy: Profit Tier × Demand Density (WRMSSE weight %)")
    plt.tight_layout()
    save_fig(fig, "2D_taxonomy_heatmap_weight")
    plt.show()

    # ---- 2E: SKU lifespan scatter ----
    sample = sku_stats.dropna(subset=["first_sale_date", "last_sale_date"]).copy()
    sample = sample.sample(min(2000, len(sample)), random_state=CONFIG["SEED"])
    tier_colors = {"A": "steelblue", "B": "orange", "C": "green", "D": "lightgray"}

    fig, ax = plt.subplots(figsize=(12, 7))
    for tier, grp in sample.groupby("profit_tier", observed=True):
        ax.scatter(grp["first_sale_date"], grp["last_sale_date"],
                   c=tier_colors[tier], alpha=0.5, s=15, label=f"Tier {tier}")
    ax.plot([CONFIG["TRAIN_START"], CONFIG["TRAIN_END"]],
            [CONFIG["TRAIN_START"], CONFIG["TRAIN_END"]],
            "k--", lw=1, alpha=0.5, label="first=last (single-sale SKU)")
    ax.axhline(CONFIG["TRAIN_END"] - timedelta(days=CONFIG["DISCONTINUED_DAYS"]),
               color="red", lw=1.5, ls=":",
               label=f"Discontinued threshold ({CONFIG['DISCONTINUED_DAYS']}d ago)")
    ax.set_xlabel("First sale date")
    ax.set_ylabel("Last sale date")
    ax.set_title("Fig 2E — SKU Lifespan: First vs Last Sale Date (sample)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    save_fig(fig, "2E_sku_lifespan")
    plt.show()

    # ---- 2F: Price distribution ----
    prices = sku_stats["mean_unit_price_global"].dropna()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(np.log10(prices.clip(lower=1)), bins=60,
            color="steelblue", edgecolor="none", alpha=0.8)
    ax.set_xlabel("log₁₀(Mean unit price, VND)")
    ax.set_ylabel("Number of SKUs")
    ax.set_title("Fig 2F — Mean Unit Price Distribution (log₁₀ scale)")
    # Add reference tick labels
    ticks = [3, 4, 5, 6, 7]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"10^{t}\n({10**t:,} VND)" for t in ticks], fontsize=9)
    plt.tight_layout()
    save_fig(fig, "2F_price_distribution")
    plt.show()


# ==============================================================================
# 8 — EDA LAYER 3: TEMPORAL PATTERN ANALYSIS  (Stage 0G)
# ==============================================================================

def eda_layer3_temporal(
    panel: pd.DataFrame,
    sku_stats: pd.DataFrame,
    holiday_df: pd.DataFrame,
) -> None:
    """
    Layer 3 EDA: seasonality, trend, autocorrelation, holiday effects.

    Plots
    -----
    3A  Aggregate daily demand + rolling averages (all SKUs)
    3B  Weekly seasonality — demand by day-of-week (box plot)
    3C  Annual seasonality — month × year heat map
    3D  Holiday impact analysis — ±14-day windows around key holidays
    3E  STL decomposition of aggregate daily demand
    3F  ACF + PACF of aggregate daily demand (stationarity assessment)
    3G  ADF / KPSS stationarity test results (printed)
    3H  Top-5 A-tier Dense SKU individual time series
    """
    set_plot_style()
    print("\n=== EDA Layer 3: Temporal Patterns ===")

    # Aggregate daily demand (all SKUs combined)
    daily_agg = (
        panel.groupby("Date")["net_qty"]
        .sum()
        .reset_index()
        .rename(columns={"net_qty": "total_qty"})
    )
    daily_agg = daily_agg.set_index("Date").sort_index()

    # ---- 3A: Aggregate demand with rolling means ----
    fig, ax = plt.subplots(figsize=CONFIG["FIGSIZE_WIDE"])
    ax.plot(daily_agg.index, daily_agg["total_qty"],
            lw=0.6, color="steelblue", alpha=0.45, label="Daily total qty")
    ax.plot(daily_agg.index, daily_agg["total_qty"].rolling(7).mean(),
            lw=1.5, color="darkorange", label="7-day rolling mean")
    ax.plot(daily_agg.index, daily_agg["total_qty"].rolling(28).mean(),
            lw=2.5, color="navy", label="28-day rolling mean")
    ax.axvline(CONFIG["EARLY_DATA_CUTOFF"], color="red", lw=1.5, ls="--",
               label=f"Early-data cutoff", alpha=0.8)

    # Mark Tết dates
    for t in _TET_DATES:
        ax.axvline(pd.Timestamp(t), color="gold", lw=1, ls=":", alpha=0.7)
    ax.axvline(pd.Timestamp(_TET_DATES[0]), color="gold", lw=1, ls=":",
               label="Tết first day")

    ax.set_title("Fig 3A — Aggregate Daily Demand (All SKUs)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total net quantity")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    save_fig(fig, "3A_aggregate_demand")
    plt.show()

    # ---- 3B: Weekly seasonality box plot ----
    daily_agg_post = daily_agg[daily_agg.index >= CONFIG["EARLY_DATA_CUTOFF"]].copy()
    daily_agg_post["dow"] = daily_agg_post.index.dayofweek
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig, ax = plt.subplots(figsize=(10, 6))
    daily_agg_post.boxplot(
        column="total_qty", by="dow",
        ax=ax, showfliers=False, patch_artist=True,
    )
    ax.set_xticklabels(day_names)
    ax.set_title("Fig 3B — Weekly Seasonality: Total Daily Demand by Day-of-Week")
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Total net qty")
    plt.suptitle("")
    plt.tight_layout()
    save_fig(fig, "3B_weekly_seasonality")
    plt.show()

    # ---- 3C: Month × Year heatmap ----
    daily_agg_post2 = daily_agg[daily_agg.index >= CONFIG["EARLY_DATA_CUTOFF"]].copy()
    daily_agg_post2["year"]  = daily_agg_post2.index.year
    daily_agg_post2["month"] = daily_agg_post2.index.month
    monthly_pivot = daily_agg_post2.groupby(
        ["year", "month"]
    )["total_qty"].mean().unstack("month")
    monthly_pivot.columns = [
        "Jan","Feb","Mar","Apr","May","Jun",
        "Jul","Aug","Sep","Oct","Nov","Dec"
    ][:len(monthly_pivot.columns)]

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(monthly_pivot, annot=True, fmt=".0f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Mean daily net qty"})
    ax.set_title("Fig 3C — Month × Year Heatmap of Mean Daily Demand")
    ax.set_ylabel("Year")
    plt.tight_layout()
    save_fig(fig, "3C_month_year_heatmap")
    plt.show()

    # ---- 3D: Holiday impact ----
    holiday_df_subset = holiday_df[holiday_df.index >= CONFIG["EARLY_DATA_CUTOFF"]]
    daily_agg_h = daily_agg.join(holiday_df_subset, how="left")
    daily_agg_h["days_to_tet"] = daily_agg_h["days_to_tet"].fillna(999)

    # Tết window: ±14 days
    tet_window = daily_agg_h[daily_agg_h["days_to_tet"].abs() <= 14].copy()
    baseline_mean = daily_agg_h.loc[
        daily_agg_h["days_to_tet"].abs() > 14, "total_qty"
    ].mean()

    if len(tet_window) > 0:
        tet_agg = tet_window.groupby("days_to_tet")["total_qty"].mean()
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(tet_agg.index, tet_agg.values, color="gold", edgecolor="none", alpha=0.8,
               label="Mean demand by days-to-Tết")
        ax.axhline(baseline_mean, color="navy", lw=2, ls="--",
                   label=f"Baseline mean = {baseline_mean:,.0f}")
        ax.axvline(0, color="red", lw=2, label="Tết day 1")
        ax.set_title("Fig 3D — Tết Holiday Impact (±14 days)")
        ax.set_xlabel("Days relative to Tết first day (negative = before)")
        ax.set_ylabel("Mean total net qty")
        ax.legend()
        plt.tight_layout()
        save_fig(fig, "3D_tet_impact")
        plt.show()

    # ---- 3E: STL decomposition ----
    # Use post-cutoff data only for reliable decomposition
    series_stl = (
        daily_agg[daily_agg.index >= CONFIG["EARLY_DATA_CUTOFF"]]["total_qty"]
        .fillna(method="ffill")
    )
    try:
        stl = STL(series_stl, period=7, robust=True)
        result = stl.fit()
        fig, axes = plt.subplots(4, 1, figsize=(18, 12), sharex=True)
        fig.suptitle("Fig 3E — STL Decomposition of Aggregate Daily Demand "
                     "(post-cutoff, period=7)", fontweight="bold")
        components = {
            "Observed":  series_stl,
            "Trend":     result.trend,
            "Seasonal":  result.seasonal,
            "Residual":  result.resid,
        }
        colors_stl = ["steelblue", "navy", "darkorange", "gray"]
        for ax, (name, comp), color in zip(axes, components.items(), colors_stl):
            ax.plot(comp.index, comp.values, lw=0.8, color=color)
            ax.set_ylabel(name, fontsize=10)
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=30)
        plt.tight_layout()
        save_fig(fig, "3E_stl_decomposition")
        plt.show()
    except Exception as e:
        print(f"  [WARNING] STL decomposition failed: {e}")

    # ---- 3F: ACF + PACF ----
    series_acf = (
        daily_agg[daily_agg.index >= CONFIG["EARLY_DATA_CUTOFF"]]["total_qty"]
        .fillna(0)
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    plot_acf(series_acf, lags=56, ax=axes[0],
             title="Fig 3F-ACF — Aggregate Daily Demand (56 lags)")
    plot_pacf(series_acf, lags=56, ax=axes[1], method="ywm",
              title="Fig 3F-PACF — Aggregate Daily Demand (56 lags)")
    for ax in axes:
        ax.axvline(28, color="red", lw=1.5, ls="--", alpha=0.7, label="lag=28")
        ax.axvline(7,  color="orange", lw=1.5, ls="--", alpha=0.7, label="lag=7")
        ax.legend(fontsize=8)
    plt.tight_layout()
    save_fig(fig, "3F_acf_pacf")
    plt.show()

    # ---- 3G: Stationarity tests (printed) ----
    print("\n  Stationarity Tests on Aggregate Daily Demand (post-cutoff):")
    # ADF
    adf_result = adfuller(series_acf.dropna(), autolag="AIC")
    print(f"  ADF  test statistic={adf_result[0]:.4f}, "
          f"p-value={adf_result[1]:.4f} "
          f"{'→ STATIONARY (p<0.05)' if adf_result[1] < 0.05 else '→ NON-STATIONARY (p≥0.05)'}")
    # KPSS
    try:
        kpss_result = kpss(series_acf.dropna(), regression="ct", nlags="auto")
        print(f"  KPSS test statistic={kpss_result[0]:.4f}, "
              f"p-value={kpss_result[1]:.4f} "
              f"{'→ NON-STATIONARY (p<0.05)' if kpss_result[1] < 0.05 else '→ STATIONARY (p≥0.05)'}")
    except Exception as e:
        print(f"  KPSS test error: {e}")

    # ---- 3H: Top-5 A-tier Dense individual traces ----
    top_dense_a = (
        sku_stats[
            (sku_stats["profit_tier"] == "A") &
            (sku_stats["demand_density"] == "Dense")
        ]
        .nlargest(5, "profit_weight")["ItemCode"]
        .tolist()
    )

    if top_dense_a:
        fig, axes = plt.subplots(
            len(top_dense_a), 1,
            figsize=(18, 3.5 * len(top_dense_a)),
            sharex=True,
        )
        if len(top_dense_a) == 1:
            axes = [axes]
        fig.suptitle("Fig 3H — Top-5 A-tier Dense SKU Time Series",
                     fontweight="bold")
        for ax, sku in zip(axes, top_dense_a):
            sku_data = (
                panel[panel["ItemCode"] == sku]
                .set_index("Date")["net_qty"]
                .sort_index()
            )
            ax.plot(sku_data.index, sku_data.values,
                    lw=0.8, color="steelblue", alpha=0.7)
            ax.plot(sku_data.index, sku_data.rolling(28).mean(),
                    lw=2, color="navy", label="28-day MA")
            w = sku_stats.loc[sku_stats["ItemCode"] == sku, "profit_weight"].values[0]
            ax.set_title(f"{sku}  |  weight={w*100:.3f}%", fontsize=10)
            ax.set_ylabel("Net qty")
            ax.legend(fontsize=8)
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=30)
        plt.tight_layout()
        save_fig(fig, "3H_top_sku_traces")
        plt.show()


# ==============================================================================
# 9 — EDA LAYER 4: AGGREGATION EDA  (Stage 0H)
# ==============================================================================

def eda_layer4_aggregation(
    panel:     pd.DataFrame,
    sku_stats: pd.DataFrame,
) -> None:
    """
    Layer 4 EDA: cross-SKU patterns, tier-level aggregations,
    return rate dynamics, price-tier clustering.

    Plots
    -----
    4A  Aggregate demand by profit tier over time
    4B  Return rate over time (aggregate + by tier)
    4C  SKU-level demand correlation (top-30 A-tier Dense)
    4D  Price tier × profit tier relationship
    """
    set_plot_style()
    print("\n=== EDA Layer 4: Aggregation EDA ===")

    panel_tier = panel.merge(
        sku_stats[["ItemCode", "profit_tier", "demand_density"]],
        on="ItemCode", how="left",
    )

    # ---- 4A: Demand by profit tier ----
    tier_daily = (
        panel_tier[panel_tier["Date"] >= CONFIG["EARLY_DATA_CUTOFF"]]
        .groupby(["Date", "profit_tier"], observed=True)["net_qty"]
        .sum().reset_index()
    )

    fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=True)
    fig.suptitle("Fig 4A — Aggregate Demand by Profit Tier (post-cutoff)",
                 fontweight="bold")
    tier_colors = {"A": "steelblue", "B": "darkorange", "C": "green", "D": "gray"}
    for ax, tier in zip(axes, ["A", "B", "C", "D"]):
        sub = tier_daily[tier_daily["profit_tier"] == tier].set_index("Date")
        if len(sub) == 0:
            ax.set_title(f"Tier {tier} — no data")
            continue
        ax.plot(sub.index, sub["net_qty"],
                lw=0.7, color=tier_colors[tier], alpha=0.5)
        ax.plot(sub.index, sub["net_qty"].rolling(28).mean(),
                lw=2, color=tier_colors[tier], label="28-day MA")
        w_tot = sku_stats[sku_stats["profit_tier"] == tier]["profit_weight"].sum()
        ax.set_title(f"Tier {tier}  (WRMSSE weight: {w_tot*100:.1f}%)")
        ax.set_ylabel("Net qty")
        ax.legend(fontsize=8)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    save_fig(fig, "4A_demand_by_tier")
    plt.show()

    # ---- 4B: Return rate over time ----
    panel_post = panel[panel["Date"] >= CONFIG["EARLY_DATA_CUTOFF"]].copy()
    daily_returns = panel_post.groupby("Date").agg(
        total_gross  =("gross_qty",  "sum"),
        total_returns=("return_qty", "sum"),
    )
    daily_returns["return_rate"] = (
        daily_returns["total_returns"] /
        (daily_returns["total_gross"] + 1e-9)
    )

    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
    fig.suptitle("Fig 4B — Return Rate Dynamics", fontweight="bold")
    axes[0].plot(daily_returns.index, daily_returns["return_rate"],
                 lw=0.6, color="tomato", alpha=0.5)
    axes[0].plot(daily_returns.index, daily_returns["return_rate"].rolling(28).mean(),
                 lw=2, color="darkred", label="28-day MA")
    axes[0].set_ylabel("Return rate (returns/gross)")
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    axes[0].set_ylim(0, 1)  # clip y-axis to 0%–100%
    axes[0].legend()

    axes[1].plot(daily_returns.index, daily_returns["total_gross"].rolling(7).mean(),
                 lw=1.5, color="steelblue", label="Gross qty (7d MA)")
    axes[1].plot(daily_returns.index, daily_returns["total_returns"].rolling(7).mean(),
                 lw=1.5, color="tomato", label="Returns qty (7d MA)")
    axes[1].set_ylabel("Quantity (7d rolling mean)")
    axes[1].legend()
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    save_fig(fig, "4B_return_rate")
    plt.show()

    # ---- 4C: SKU demand correlation heatmap (top-30 A-tier Dense) ----
    top_a_dense = (
        sku_stats[
            (sku_stats["profit_tier"] == "A") &
            (sku_stats["demand_density"] == "Dense")
        ]
        .nlargest(30, "profit_weight")["ItemCode"]
        .tolist()
    )

    if len(top_a_dense) >= 2:
        corr_panel = (
            panel[
                (panel["ItemCode"].isin(top_a_dense)) &
                (panel["Date"] >= CONFIG["EARLY_DATA_CUTOFF"])
            ]
            .pivot_table(index="Date", columns="ItemCode",
                         values="net_qty", fill_value=0)
        )
        corr_matrix = corr_panel.corr()

        fig, ax = plt.subplots(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(
            corr_matrix, mask=mask, cmap="RdBu_r",
            vmin=-1, vmax=1, annot=len(top_a_dense) <= 20,
            fmt=".2f", linewidths=0.3,
            ax=ax, cbar_kws={"label": "Pearson correlation"},
        )
        ax.set_title("Fig 4C — Demand Correlation: Top-30 A-tier Dense SKUs")
        plt.tight_layout()
        save_fig(fig, "4C_sku_correlation")
        plt.show()

    # ---- 4D: Price tier vs profit tier ----
    sku_stats_copy = sku_stats.copy()
    sku_stats_copy["log10_price"] = np.log10(
        sku_stats_copy["mean_unit_price_global"].clip(lower=1)
    )
    price_tier_labels = ["<10k", "10k-100k", "100k-1M", "1M-10M", ">10M"]
    price_bins = [0, 4, 5, 6, 7, np.inf]
    sku_stats_copy["price_tier"] = pd.cut(
        sku_stats_copy["log10_price"],
        bins=price_bins,
        labels=price_tier_labels,
    )

    cross = pd.crosstab(
        sku_stats_copy["price_tier"],
        sku_stats_copy["profit_tier"],
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    cross.plot(kind="bar", ax=ax, colormap="tab10", edgecolor="none")
    ax.set_title("Fig 4D — Price Tier vs Profit Tier Distribution")
    ax.set_xlabel("Price tier (VND)")
    ax.set_ylabel("Number of SKUs")
    ax.legend(title="Profit tier")
    plt.xticks(rotation=0)
    plt.tight_layout()
    save_fig(fig, "4D_price_vs_profit_tier")
    plt.show()


# ==============================================================================
# 10 — NAIVE SEASONAL BASELINE + WRMSSE FLOOR  (Stage 0I)
# ==============================================================================

def compute_naive_baseline(
    panel:     pd.DataFrame,
    sku_stats: pd.DataFrame,
    folds:     List[Dict],
) -> Dict:
    """
    Compute the naive seasonal baseline and evaluate WRMSSE across CV folds.

    Naive forecast rule
    -------------------
    For each (SKU, forecast_day), predict the mean of the same day-of-week
    observed over the 4 weeks immediately preceding the fold's train_end.
    If no same-DOW observations exist in that window, fall back to the
    mean of the last 28 training days. If still zero, predict 0.

    Returns
    -------
    Dict with keys:
        fold_wrmsse : List[float]   per-fold WRMSSE
        mean_wrmsse : float
        std_wrmsse  : float
        oof_preds   : pd.DataFrame  all OOF predictions (for ensemble use)
    """
    print("\n=== Stage 0I: Naive Seasonal Baseline ===")

    weights = sku_stats.set_index("ItemCode")["profit_weight"]

    # Build wide panel: rows=Date, columns=ItemCode (faster slicing)
    panel_wide = panel.pivot_table(
        index="Date", columns="ItemCode", values="net_qty", fill_value=0
    )
    all_skus = panel_wide.columns.tolist()

    fold_wrmsse_list = []
    all_oof_records  = []

    for fold in folds:
        fold_id   = fold["fold_id"]
        train_end = fold["train_end"]
        val_start = fold["val_start"]
        val_end   = fold["val_end"]

        train_wide = panel_wide.loc[:train_end]
        val_wide   = panel_wide.loc[val_start:val_end]
        h          = len(val_wide)

        if h == 0:
            print(f"  Fold {fold_id}: empty validation window — skipped.")
            continue

        forecasts = {}
        for sku in all_skus:
            train_series = train_wide[sku].values
            # Last 28 days of training for this fold
            recent_28 = train_series[-28:]
            # Corresponding dates for DOW lookup
            recent_dates = train_wide.index[-28:]

            preds = []
            for val_date in val_wide.index:
                target_dow = val_date.dayofweek
                same_dow_vals = recent_28[
                    [d.dayofweek == target_dow for d in recent_dates]
                ]
                if len(same_dow_vals) > 0 and same_dow_vals.mean() > 0:
                    preds.append(max(0.0, float(same_dow_vals.mean())))
                else:
                    # Fallback: mean of last 28 days
                    fallback = max(0.0, float(recent_28.mean()))
                    preds.append(fallback)
            forecasts[sku] = preds

        forecast_df = pd.DataFrame(
            forecasts,
            index=val_wide.index,
        )
        actuals_df = val_wide

        # WRMSSE
        wrmsse_val, rmsse_per_sku = compute_wrmsse(
            actuals    = actuals_df,
            forecasts  = forecast_df,
            train_qtys = train_wide,
            weights    = weights,
        )
        fold_wrmsse_list.append(wrmsse_val)
        print(f"  Fold {fold_id}  "
              f"[{val_start.date()} → {val_end.date()}]  "
              f"WRMSSE = {wrmsse_val:.4f}")

        # Collect OOF records
        for sku in all_skus:
            for i, val_date in enumerate(val_wide.index):
                all_oof_records.append({
                    "fold_id":   fold_id,
                    "ItemCode":  sku,
                    "Date":      val_date,
                    "actual":    float(actuals_df.at[val_date, sku]),
                    "naive_pred":float(forecast_df.at[val_date, sku]),
                })

    mean_w = float(np.mean(fold_wrmsse_list))
    std_w  = float(np.std(fold_wrmsse_list))
    print(f"\n  Naive baseline CV WRMSSE: {mean_w:.4f} ± {std_w:.4f}")
    print(f"  (This is the floor score that all models must beat.)")

    oof_df = pd.DataFrame(all_oof_records)

    return {
        "fold_wrmsse": fold_wrmsse_list,
        "mean_wrmsse": mean_w,
        "std_wrmsse":  std_w,
        "oof_preds":   oof_df,
    }


def plot_naive_baseline_diagnostics(naive_results: Dict) -> None:
    """
    Plot CV fold WRMSSE scores for the naive baseline.

    Plots
    -----
    N1  Bar chart of per-fold WRMSSE with mean ± std band
    """
    set_plot_style()
    fold_scores = naive_results["fold_wrmsse"]
    mean_w      = naive_results["mean_wrmsse"]
    std_w       = naive_results["std_wrmsse"]

    fig, ax = plt.subplots(figsize=(9, 5))
    fold_ids = [f"Fold {i}" for i in range(len(fold_scores))]
    bars = ax.bar(fold_ids, fold_scores, color="steelblue",
                  edgecolor="none", alpha=0.8, width=0.5)
    ax.axhline(mean_w, color="navy", lw=2, ls="--",
               label=f"Mean = {mean_w:.4f}")
    ax.axhspan(mean_w - std_w, mean_w + std_w,
               alpha=0.15, color="navy", label=f"±1 std ({std_w:.4f})")
    for bar, score in zip(bars, fold_scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{score:.4f}", ha="center", va="bottom", fontsize=10)
    ax.set_title("Fig N1 — Naive Seasonal Baseline: CV WRMSSE per Fold")
    ax.set_ylabel("WRMSSE")
    ax.legend()
    plt.tight_layout()
    save_fig(fig, "N1_naive_baseline_cv")
    plt.show()

# ==============================================================================
# 11 — MAIN EXECUTION BLOCK
# ==============================================================================

def run_eda() -> None:
    """
    Execute the full Stage 0 pipeline in order.
    Each stage prints a section header for easy notebook navigation.
    """
    set_plot_style()

    print("\n" + "=" * 60)
    print(" HBAAC 2026 — Stage 0: EDA & Data Foundation")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 0A: Load raw data
    # ------------------------------------------------------------------
    print("\n--- Stage 0A: Loading raw data ---")
    train_raw, sub_raw = load_raw_data(CONFIG["DATASET_DIR"])
    print(f"  train_raw shape   : {train_raw.shape}")
    print(f"  submission shape  : {sub_raw.shape}")
    print(f"  Submission SKUs   : {sub_raw['id'].str.extract(r'^(.+)_(validation|evaluation)$')[0].nunique():,}")

    # ------------------------------------------------------------------
    # 0A continued: Parse
    # ------------------------------------------------------------------
    print("\n--- Stage 0A: Parsing ---")
    df = parse_train(train_raw)
    del train_raw
    gc.collect()

    # ------------------------------------------------------------------
    # 0B: Build daily panel
    # ------------------------------------------------------------------
    print("\n--- Stage 0B: Building daily panel ---")
    panel = build_daily_panel(df)

    # ------------------------------------------------------------------
    # 0C: Profit weights & SKU taxonomy
    # ------------------------------------------------------------------
    print("\n--- Stage 0C: Computing SKU stats, profit weights, taxonomy ---")
    sku_stats = compute_sku_stats(panel)
    sku_stats.to_csv(CONFIG["OUTPUT_DIR"] / "sku_stats.csv", index=False)
    print(f"  Saved sku_stats.csv to {CONFIG['OUTPUT_DIR']}")

    # ------------------------------------------------------------------
    # 0D: CV fold dates
    # ------------------------------------------------------------------
    print("\n--- Stage 0D: Computing CV fold dates ---")
    folds = compute_cv_fold_dates(
        train_start      = CONFIG["TRAIN_START"],
        train_end        = CONFIG["TRAIN_END"],
        horizon          = CONFIG["HORIZON_DAYS"],
        n_folds          = CONFIG["N_FOLDS"],
        anchor_gap_days  = 365,
    )
    print(f"\n  {'Fold':>5}  {'Train start':>12}  {'Train end':>12}  "
          f"{'Val start':>12}  {'Val end':>12}")
    print("  " + "-" * 62)
    for f in folds:
        print(f"  {f['fold_id']:>5}  {str(f['train_start'].date()):>12}  "
              f"{str(f['train_end'].date()):>12}  "
              f"{str(f['val_start'].date()):>12}  "
              f"{str(f['val_end'].date()):>12}")

    # ------------------------------------------------------------------
    # Build holiday series for full training + test range
    # ------------------------------------------------------------------
    full_date_range = pd.date_range(
        CONFIG["TRAIN_START"],
        CONFIG["PRIVATE_TEST_END"],
        freq="D",
    )

    # ------------------------------------------------------------------
    # 0E: EDA Layer 1
    # ------------------------------------------------------------------
    eda_layer1_dataset(df, panel)

    # ------------------------------------------------------------------
    # 0F: EDA Layer 2
    # ------------------------------------------------------------------
    eda_layer2_sku(sku_stats)

    # ------------------------------------------------------------------
    # 0G: EDA Layer 3
    # ------------------------------------------------------------------
    holiday_df = build_vn_holiday_series(full_date_range)
    
    eda_layer3_temporal(panel, sku_stats, holiday_df)

    # ------------------------------------------------------------------
    # 0H: EDA Layer 4
    # ------------------------------------------------------------------
    eda_layer4_aggregation(panel, sku_stats)

    # ------------------------------------------------------------------
    # 0I: Naive baseline + WRMSSE floor
    # ------------------------------------------------------------------
    naive_results = compute_naive_baseline(panel, sku_stats, folds)
    plot_naive_baseline_diagnostics(naive_results)

    # Save naive OOF predictions for ensemble baseline comparison
    naive_results["oof_preds"].to_parquet(
        CONFIG["OUTPUT_DIR"] / "naive_oof_predictions.parquet",
        index=False,
    )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" Stage 0 Complete — Summary")
    print("=" * 60)
    print(f"  Total SKUs              : {sku_stats['ItemCode'].nunique():,}")
    print(f"  Training calendar days  : {panel['Date'].nunique():,}")
    print(f"  A-tier SKUs             : "
          f"{(sku_stats['profit_tier']=='A').sum():,}  "
          f"({sku_stats.loc[sku_stats['profit_tier']=='A','profit_weight'].sum()*100:.1f}% weight)")
    print(f"  B-tier SKUs             : "
          f"{(sku_stats['profit_tier']=='B').sum():,}  "
          f"({sku_stats.loc[sku_stats['profit_tier']=='B','profit_weight'].sum()*100:.1f}% weight)")
    print(f"  C-tier SKUs             : "
          f"{(sku_stats['profit_tier']=='C').sum():,}  "
          f"({sku_stats.loc[sku_stats['profit_tier']=='C','profit_weight'].sum()*100:.1f}% weight)")
    print(f"  D-tier SKUs             : "
          f"{(sku_stats['profit_tier']=='D').sum():,}  "
          f"({sku_stats.loc[sku_stats['profit_tier']=='D','profit_weight'].sum()*100:.1f}% weight)")
    print(f"  Naive baseline WRMSSE   : "
          f"{naive_results['mean_wrmsse']:.4f} ± {naive_results['std_wrmsse']:.4f}")
    print(f"  All outputs saved to    : {CONFIG['OUTPUT_DIR']}")
    print("=" * 60)

    # Return key artifacts for interactive use in notebook
    return {
        "df":            df,
        "panel":         panel,
        "sku_stats":     sku_stats,
        "folds":         folds,
        "naive_results": naive_results,
        "sub_raw":       sub_raw
    }
    
# Entry point EDA
if __name__ == "__main__":
    artifacts = run_eda()
    panel = artifacts["panel"]
    sku_stats = artifacts["sku_stats"]
    folds = artifacts["folds"]
    naive_results = artifacts["naive_results"]
    sub_raw = artifacts["sub_raw"]





