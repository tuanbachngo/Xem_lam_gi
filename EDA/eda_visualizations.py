"""
HBAAC Retail Time-Series Exploratory Data Analysis (EDA) Pipeline.

Author: Data Scientist & Visualization Expert
Description: PEP-8 compliant, production-grade exploratory data analysis script.
             Generates seven business-critical and econometric visualizations:
             1. Pareto Distribution (Long-tail SKU analysis)
             2. Macro Seasonality Trends (Weekly aggregated sales timeline)
             3. Micro Demand Structure (Day-of-week and Month-of-year boxplots)
             4. Sparsity Histogram (Demand intermittency across SKUs)
             5. Price Elasticity (UnitPrice vs Quantity for Top 5 revenue SKUs)
             6. Autocorrelation Profiles (ACF and PACF with 60-day lag)
             7. Return Transactions Tracker (Timeline comparing gross sales vs returns)
"""

import os
import time
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants & Configurations
# Make paths robust so the script runs successfully from both the project root and the EDA/ folder
if os.path.exists(os.path.join("hbaac-round2", "train.csv")):
    TRAIN_PATH = os.path.join("hbaac-round2", "train.csv")
    OUTPUT_DIR = os.path.join("EDA", "eda_plots")
else:
    TRAIN_PATH = os.path.join("..", "hbaac-round2", "train.csv")
    OUTPUT_DIR = "eda_plots"

START_DATE = "2020-11-17"
END_DATE = "2025-09-05"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set premium styling aesthetics globally
sns.set_theme(style="white", palette="muted")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16,
    "font.family": "sans-serif"
})


def parse_monetary_column(series: pd.Series) -> pd.Series:
    """
    Standardizes dirty monetary string columns into floats for audit validation.
    Handles varied locales, decimal commas, and multiple thousand separators.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float).fillna(0.0)
    
    cleaned = series.astype(str).str.strip().str.replace('"', '').str.replace("'", "")
    cleaned = cleaned.replace({'nan': '0', '': '0'})
    
    unique_vals = cleaned.unique()
    parsed_map = {}
    
    for val in unique_vals:
        if not val or val == '0':
            parsed_map[val] = 0.0
            continue
            
        if ',' in val:
            parts = val.split(',')
            if len(parts) == 2:
                if len(parts[1]) == 3:
                    parsed_map[val] = float("".join(parts))
                else:
                    parsed_map[val] = float(f"{parts[0]}.{parts[1]}")
            else:
                if len(parts[-1]) == 3:
                    parsed_map[val] = float("".join(parts))
                else:
                    parsed_map[val] = float("".join(parts[:-1]) + "." + parts[-1])
        else:
            try:
                parsed_map[val] = float(val)
            except ValueError:
                parsed_map[val] = 0.0
                
    return cleaned.map(parsed_map).fillna(0.0)


def load_and_preprocess_data(train_path: str) -> tuple:
    """
    Loads raw transaction log, parses dates, parses monetary columns, and returns
    both raw transaction-level and daily aggregated datasets.
    """
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing training dataset file: {train_path}")
        
    logger.info("Loading transaction logs for EDA from %s...", train_path)
    df_raw = pd.read_csv(train_path, low_memory=False)
    
    logger.info("Casting date column to datetime...")
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])
    
    logger.info("Parsing and casting monetary columns...")
    monetary_cols = ['UnitPrice', 'SalesAmount', 'Unit Cost', 'Cost Amount']
    for col in monetary_cols:
        if col in df_raw.columns:
            df_raw[col] = parse_monetary_column(df_raw[col])
            
    # Clean Quantity for raw operations (handle negative returns cleanly in parsing copy)
    df_raw['Quantity'] = pd.to_numeric(df_raw['Quantity'], errors='coerce').fillna(0.0)
            
    # Aggregate to daily levels by ItemCode
    logger.info("Aggregating sales quantities daily by ItemCode...")
    df_agg = df_raw.groupby(['Date', 'ItemCode'], as_index=False).agg({
        'Quantity': 'sum',
        'SalesAmount': 'sum'
    })
    
    # Non-negative constraint enforcement (handling return anomalies) for aggregated data
    df_agg['Quantity'] = df_agg['Quantity'].clip(lower=0)
    df_agg['SalesAmount'] = df_agg['SalesAmount'].clip(lower=0)
    
    return df_raw, df_agg


def build_timeseries_matrix(df_agg: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Pivots daily aggregated quantities and reindexes to a continuous date range.
    Lays the foundation for sparsity evaluation.
    """
    logger.info("Building pivoted continuous daily time-series matrix...")
    df_pivot = df_agg.pivot(index='ItemCode', columns='Date', values='Quantity')
    continuous_dates = pd.date_range(start=start_date, end=end_date)
    df_matrix = df_pivot.reindex(columns=continuous_dates, fill_value=0)
    return df_matrix


# -----------------------------------------------------------------
# 1. Pareto Long-Tail Distribution
# -----------------------------------------------------------------
def plot_pareto_distribution(df_agg: pd.DataFrame, output_dir: str) -> None:
    logger.info("Generating Pareto Long-Tail Distribution plot...")
    sku_sales = df_agg.groupby('ItemCode')['Quantity'].sum().reset_index()
    sku_sales = sku_sales.sort_values(by='Quantity', ascending=False).reset_index(drop=True)
    sku_sales['Cumulative_Quantity'] = sku_sales['Quantity'].cumsum()
    total_quantity = sku_sales['Quantity'].sum()
    sku_sales['Cumulative_Percentage'] = (sku_sales['Cumulative_Quantity'] / total_quantity) * 100
    
    cross_80_idx = np.where(sku_sales['Cumulative_Percentage'] >= 80)[0][0]
    sku_pct_at_80 = ((cross_80_idx + 1) / len(sku_sales)) * 100
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    top_n = min(100, len(sku_sales))
    
    ax1.bar(
        range(top_n), 
        sku_sales['Quantity'].head(top_n), 
        color="#1f77b4", 
        alpha=0.85, 
        edgecolor='none', 
        width=0.8,
        label="Individual SKU Demand"
    )
    ax1.set_ylabel("Total Demand Quantity Sold", color="#1f77b4")
    ax1.set_xlabel("SKUs (ItemCode) Sorted by Sales Volume (Top 100 displayed)")
    ax1.tick_params(axis='y', labelcolor="#1f77b4")
    ax1.set_title("Pareto Long-Tail Demand Distribution across SKUs", pad=15)
    
    ax2 = ax1.twinx()
    ax2.plot(
        range(top_n), 
        sku_sales['Cumulative_Percentage'].head(top_n), 
        color="#e377c2", 
        linewidth=2.5, 
        label="Cumulative % of Total Demand"
    )
    ax2.set_ylabel("Cumulative Percentage of Total Demand (%)", color="#e377c2")
    ax2.tick_params(axis='y', labelcolor="#e377c2")
    
    ax2.axhline(80, color="#d62728", linestyle="--", alpha=0.7, linewidth=1.5)
    ax2.text(
        x=top_n * 0.4, 
        y=82, 
        s=f"Top {sku_pct_at_80:.1f}% of SKUs account for 80% of total demand volume", 
        color="#d62728", 
        weight='semibold'
    )
    
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "pareto_sku_distribution.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved Pareto plot to %s.", plot_path)


# -----------------------------------------------------------------
# 2. Macro Seasonality Trends
# -----------------------------------------------------------------
def plot_macro_trends(df_agg: pd.DataFrame, output_dir: str) -> None:
    logger.info("Generating Macro Seasonality Trends timeline...")
    daily_sales = df_agg.groupby('Date')['Quantity'].sum().reset_index()
    daily_sales = daily_sales.set_index('Date')
    weekly_sales = daily_sales['Quantity'].resample('W').sum().reset_index()
    
    plt.figure(figsize=(14, 5.5))
    plt.plot(
        weekly_sales['Date'], 
        weekly_sales['Quantity'], 
        color="#2ca02c", 
        linewidth=2.0, 
        alpha=0.9,
        label="Weekly Demand Sum"
    )
    
    weekly_sales['Trend_12W'] = weekly_sales['Quantity'].rolling(window=12, center=True).mean()
    plt.plot(
        weekly_sales['Date'], 
        weekly_sales['Trend_12W'], 
        color="#ff7f0e", 
        linestyle="-", 
        linewidth=2.5,
        label="12-Week Central Trend"
    )
    
    plt.title("Global Retail Demand Timeline: Macro Trends and Peak Seasonality", pad=15)
    plt.xlabel("Timeline (Years)")
    plt.ylabel("Total Quantity Sold (Weekly)")
    plt.grid(axis='y', linestyle=":", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", edgecolor="none")
    
    sns.despine(left=True, bottom=True)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "macro_sales_trends.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved Macro trends plot to %s.", plot_path)


# -----------------------------------------------------------------
# 3. Micro Demand Structure (Calendar Seasonality)
# -----------------------------------------------------------------
def plot_micro_seasonality(df_agg: pd.DataFrame, output_dir: str) -> None:
    logger.info("Generating Micro Demand calendar boxplots...")
    daily_sales = df_agg.groupby('Date')['Quantity'].sum().reset_index()
    
    daily_sales['DayOfWeek'] = daily_sales['Date'].dt.day_name()
    daily_sales['Month'] = daily_sales['Date'].dt.strftime('%b')
    
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    daily_sales['DayOfWeek'] = pd.Categorical(daily_sales['DayOfWeek'], categories=day_order, ordered=True)
    daily_sales['Month'] = pd.Categorical(daily_sales['Month'], categories=month_order, ordered=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
    
    sns.boxplot(
        x='DayOfWeek', 
        y='Quantity', 
        data=daily_sales, 
        ax=axes[0], 
        palette="Blues",
        showfliers=False,
        linewidth=1.2
    )
    axes[0].set_title("Demand Distribution by Day of Week", pad=12)
    axes[0].set_xlabel("Day of the Week")
    axes[0].set_ylabel("Daily Quantity Sold (Global Sum)")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30)
    
    sns.boxplot(
        x='Month', 
        y='Quantity', 
        data=daily_sales, 
        ax=axes[1], 
        palette="Purples",
        showfliers=False,
        linewidth=1.2
    )
    axes[1].set_title("Demand Distribution by Month of Year", pad=12)
    axes[1].set_xlabel("Month of the Year")
    axes[1].set_ylabel("")
    
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.0)
        ax.grid(axis='y', linestyle=":", alpha=0.4)
        
    plt.suptitle("Micro Demand Patterns: Calendar-Driven Cyclic Seasonality", y=0.98)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "micro_calendar_seasonality.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved Micro seasonality boxplot to %s.", plot_path)


# -----------------------------------------------------------------
# 4. Sparsity Histogram (Advanced Chart 1)
# -----------------------------------------------------------------
def plot_sparsity_distribution(df_matrix: pd.DataFrame, output_dir: str) -> None:
    """
    Calculates the exact demand sparsity per ItemCode (percentage of days with 0 quantity)
    and plots its distribution.
    """
    logger.info("Generating Sparsity (Intermittent Demand) Distribution plot...")
    
    # Calculate zero days percentage for each row (ItemCode)
    zero_pct = (df_matrix == 0).sum(axis=1) / df_matrix.shape[1] * 100
    
    plt.figure(figsize=(10, 6))
    sns.histplot(
        zero_pct, 
        bins=30, 
        kde=True, 
        color="#4a90e2", 
        edgecolor="white", 
        alpha=0.85
    )
    
    # Mathematical and Econometric Insights Polish
    plt.title("Distribution of Demand Sparsity (Zero-Sales Days %) across SKUs", pad=15)
    plt.xlabel("Percentage of Days with Zero Sales (%)")
    plt.ylabel("Count of SKUs")
    plt.grid(axis='y', linestyle=":", alpha=0.5)
    
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "sparsity_histogram.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved Sparsity plot to %s.", plot_path)


# -----------------------------------------------------------------
# 5. Price Elasticity Scatter Plot (Advanced Chart 2)
# -----------------------------------------------------------------
def plot_price_elasticity(df_raw: pd.DataFrame, output_dir: str) -> None:
    """
    Identifies the Top 5 ItemCodes with the highest sales revenue (SalesAmount)
    and draws scatter subplots of UnitPrice vs Quantity on active sale days.
    """
    logger.info("Generating Price Elasticity Scatter Plot for Top 5 SKUs by SalesAmount...")
    
    # Find top 5 ItemCodes by total SalesAmount
    top_5_skus = df_raw.groupby('ItemCode')['SalesAmount'].sum().nlargest(5).index.tolist()
    
    fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=False)
    
    for i, sku in enumerate(top_5_skus):
        # Extract active sales where both Quantity and UnitPrice are positive
        sku_data = df_raw[
            (df_raw['ItemCode'] == sku) & 
            (df_raw['Quantity'] > 0) & 
            (df_raw['UnitPrice'] > 0)
        ]
        
        # Draw Scatter Plot with a regression line to display elasticity
        sns.regplot(
            x='UnitPrice', 
            y='Quantity', 
            data=sku_data, 
            ax=axes[i],
            scatter_kws={'alpha': 0.5, 'color': '#2ca02c', 'edgecolor': 'none', 's': 20},
            line_kws={'color': '#d62728', 'linewidth': 2}
        )
        
        axes[i].set_title(f"SKU: {sku}", fontsize=12, pad=10)
        axes[i].set_xlabel("UnitPrice")
        
        if i == 0:
            axes[i].set_ylabel("Quantity (Active Days)")
        else:
            axes[i].set_ylabel("")
            
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].grid(linestyle=":", alpha=0.4)
        
    plt.suptitle("Price Elasticity of Demand: UnitPrice vs Quantity (Top 5 Revenue SKUs)", y=1.02)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "price_elasticity_scatter.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved Price Elasticity plot to %s.", plot_path)


# -----------------------------------------------------------------
# 6. Autocorrelation (ACF / PACF Plots) (Advanced Chart 3)
# -----------------------------------------------------------------
def plot_acf_pacf(df_agg: pd.DataFrame, output_dir: str) -> None:
    """
    Aggregates overall daily sales quantity and plots ACF/PACF side-by-side
    with up to a 60-day lag profile.
    """
    logger.info("Generating ACF and PACF plots for global daily demand...")
    
    # Aggregate demand globally by date
    daily_global = df_agg.groupby('Date')['Quantity'].sum().sort_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Autocorrelation Function (ACF)
    plot_acf(
        daily_global, 
        lags=60, 
        ax=axes[0]
    )
    axes[0].set_title("Autocorrelation (ACF) - Lags up to 60 Days", pad=10)
    axes[0].set_xlabel("Lag (Days)")
    axes[0].set_ylabel("Correlation Coefficient")
    axes[0].grid(linestyle=":", alpha=0.4)
    
    # Partial Autocorrelation Function (PACF)
    plot_pacf(
        daily_global, 
        lags=60, 
        ax=axes[1]
    )
    axes[1].set_title("Partial Autocorrelation (PACF) - Lags up to 60 Days", pad=10)
    axes[1].set_xlabel("Lag (Days)")
    axes[1].set_ylabel("")
    axes[1].grid(linestyle=":", alpha=0.4)
    
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    plt.suptitle("Global Demand Autocorrelation Profile", y=1.02)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "acf_pacf_plots.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved ACF/PACF plots to %s.", plot_path)


# -----------------------------------------------------------------
# 7. Return Transactions Tracker (Advanced Chart 4)
# -----------------------------------------------------------------
def plot_return_tracker(df_raw: pd.DataFrame, output_dir: str) -> None:
    """
    Compares gross positive sales quantities against absolute negative quantities (returns)
    on a synchronized weekly timeline to understand return noise.
    """
    logger.info("Generating Return Transactions Tracker timeline...")
    
    # Calculate daily positive sales quantity
    daily_sales = df_raw[df_raw['Quantity'] > 0].groupby('Date')['Quantity'].sum().reset_index()
    
    # Calculate daily returns (Quantity < 0) absolute quantity
    daily_returns = df_raw[df_raw['Quantity'] < 0].copy()
    daily_returns['Quantity'] = daily_returns['Quantity'].abs()
    daily_returns = daily_returns.groupby('Date')['Quantity'].sum().reset_index()
    
    # Merge timelines and fill zero days
    merged = pd.merge(
        daily_sales, 
        daily_returns, 
        on='Date', 
        how='outer', 
        suffixes=('_Sales', '_Returns')
    ).fillna(0.0)
    
    merged = merged.sort_values(by='Date').set_index('Date')
    
    # Resample weekly to smooth high frequency noise and present a clean timeline
    weekly_timeline = merged.resample('W').sum().reset_index()
    
    plt.figure(figsize=(14, 5.5))
    
    plt.plot(
        weekly_timeline['Date'], 
        weekly_timeline['Quantity_Sales'], 
        color="#2ca02c", 
        linewidth=2.0, 
        alpha=0.85, 
        label="Gross Outward Sales (Quantity > 0)"
    )
    plt.plot(
        weekly_timeline['Date'], 
        weekly_timeline['Quantity_Returns'], 
        color="#d62728", 
        linewidth=1.8, 
        alpha=0.85, 
        linestyle="--", 
        label="Customer Returns (Absolute Quantity < 0)"
    )
    
    plt.title("Comparative Retail Timeline: Sales Volume vs Customer Returns (Weekly Aggregated)", pad=15)
    plt.xlabel("Timeline")
    plt.ylabel("Aggregated Quantity")
    plt.grid(axis='y', linestyle=":", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", edgecolor="none")
    
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "return_transactions_tracker.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved Return Tracker plot to %s.", plot_path)


# -----------------------------------------------------------------
# Main Execution Orchestrator
# -----------------------------------------------------------------
def main():
    logger.info("=== Starting Exploratory Data Analysis (EDA) Visualization Pipeline ===")
    start_time = time.time()
    
    try:
        # Step 1: Load and clean data (reusing robust transactional parsing)
        df_raw, df_agg = load_and_preprocess_data(TRAIN_PATH)
        
        # Step 2: Build pivoted zero-filled continuous matrix
        df_matrix = build_timeseries_matrix(df_agg, START_DATE, END_DATE)
        
        # Step 3: Run original plotting scripts
        plot_pareto_distribution(df_agg, OUTPUT_DIR)
        plot_macro_trends(df_agg, OUTPUT_DIR)
        plot_micro_seasonality(df_agg, OUTPUT_DIR)
        
        # Step 4: Run advanced plotting scripts
        plot_sparsity_distribution(df_matrix, OUTPUT_DIR)
        plot_price_elasticity(df_raw, OUTPUT_DIR)
        plot_acf_pacf(df_agg, OUTPUT_DIR)
        plot_return_tracker(df_raw, OUTPUT_DIR)
        
        logger.info("=== EDA Pipeline Completed Successfully in %.2f seconds! ===", time.time() - start_time)
        logger.info("All plots are saved in directory: %s/", OUTPUT_DIR)
        
    except Exception as e:
        logger.exception("EDA pipeline failed due to an error: %s", str(e))


if __name__ == "__main__":
    main()
