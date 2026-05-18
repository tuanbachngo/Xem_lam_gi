"""
HBAAC Retail Time-Series Feature Engineering Pipeline.

Author: Senior Data Scientist & Retail Analytics Specialist
Description: PEP-8 compliant feature engineering pipeline. Melts the continuous
             time-series matrix into long format and constructs safe lag and rolling 
             features anchored at Lag_28 to prevent data leakage in 28-day direct
             multi-step forecasting.
"""

import os
import time
import logging
import pandas as pd
import numpy as np

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants and Configuration
TRAIN_PATH = os.path.join("hbaac-round2", "train.csv")
OUTPUT_DIR = "src"
FEATURE_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "train_features.csv")

START_DATE = "2020-11-17"
END_DATE = "2025-09-05"


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


def load_and_preprocess_data(train_path: str) -> pd.DataFrame:
    """
    Loads raw transaction log, parses dates/monetary types, aggregates quantity
    daily, and clips negative quantities from returned transactions.
    """
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing training dataset file: {train_path}")
        
    logger.info("Loading transaction logs from %s...", train_path)
    df = pd.read_csv(train_path, low_memory=False)
    
    logger.info("Casting date column to datetime...")
    df['Date'] = pd.to_datetime(df['Date'])
    
    logger.info("Parsing and type casting monetary columns...")
    monetary_cols = ['UnitPrice', 'SalesAmount', 'Unit Cost', 'Cost Amount']
    for col in monetary_cols:
        if col in df.columns:
            df[col] = parse_monetary_column(df[col])
            
    # Daily aggregation by ItemCode
    logger.info("Aggregating sales quantities daily by ItemCode...")
    df_agg = df.groupby(['Date', 'ItemCode'], as_index=False)['Quantity'].sum()
    
    # Enforce positive quantities (handling return transactions)
    df_agg['Quantity'] = df_agg['Quantity'].clip(lower=0)
    
    return df_agg


def build_timeseries_matrix(df_agg: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Pivots daily aggregated quantities and reindexes to a continuous date range.
    Uses fillna(0.0) to ensure zero-sales days inside active dates are properly filled.
    """
    logger.info("Pivoting aggregated DataFrame into broad time-series matrix...")
    df_pivot = df_agg.pivot(index='ItemCode', columns='Date', values='Quantity')
    
    logger.info("Reindexing matrix to continuous range: %s to %s...", start_date, end_date)
    continuous_dates = pd.date_range(start=start_date, end=end_date)
    # Reindex handles completely missing columns/dates; fillna(0.0) handles existing sparse cell NaNs.
    df_matrix = df_pivot.reindex(columns=continuous_dates, fill_value=0).fillna(0.0)
    
    return df_matrix


def melt_matrix(df_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    1. Melting Time-Series Matrix
    Converts wide continuous time-series matrix into a long format.
    Guarantees structural sorting by ItemCode and Date before lag engineering.
    """
    logger.info("Melting time-series matrix from wide to long format...")
    df_long = df_matrix.reset_index().melt(
        id_vars=['ItemCode'], 
        var_name='Date', 
        value_name='Target_Quantity'
    )
    
    # Ensure proper datatypes and chronological sort order
    df_long['Date'] = pd.to_datetime(df_long['Date'])
    df_long = df_long.sort_values(by=['ItemCode', 'Date']).reset_index(drop=True)
    
    return df_long


def extract_temporal_features(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    2. Temporal Features Extraction
    Extracts time-based predictors from the Date column. Uses low-precision
    integers to optimize memory usage over large datasets.
    """
    logger.info("Extracting temporal features (Day_of_week, Month, Is_Weekend)...")
    # Day_of_week: Monday=0, Sunday=6
    df_long['Day_of_week'] = df_long['Date'].dt.dayofweek.astype('int8')
    # Month: 1 to 12
    df_long['Month'] = df_long['Date'].dt.month.astype('int8')
    # Is_Weekend: Saturday (5) and Sunday (6) are flagged as 1, else 0
    df_long['Is_Weekend'] = df_long['Date'].dt.dayofweek.isin([5, 6]).astype('int8')
    
    return df_long


def generate_lag_features(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    3. Lag Features Generation
    Constructs safe history-based lags anchored at t-28 to avoid data leakage
    during 28-day forecasting horizons.
    """
    logger.info("Generating leakage-free lag features (Lag_28, Lag_35, Lag_42, Lag_56)...")
    grouped = df_long.groupby('ItemCode')['Target_Quantity']
    
    # Anchored at 28 days to match forecasting horizon boundary
    df_long['Lag_28'] = grouped.shift(28)
    df_long['Lag_35'] = grouped.shift(35)
    df_long['Lag_42'] = grouped.shift(42)
    df_long['Lag_56'] = grouped.shift(56)
    
    return df_long


def generate_rolling_features(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    4. Rolling Window Features
    Computes rolling averages and volatility statistics strictly on the
    anchored Lag_28 column. This prevents future information leakage.
    """
    logger.info("Generating rolling window features on Lag_28 anchor...")
    grouped = df_long.groupby('ItemCode')['Lag_28']
    
    # Rolling Means
    df_long['Rolling_Mean_7_of_Lag_28'] = grouped.transform(
        lambda x: x.rolling(window=7, min_periods=7).mean()
    )
    df_long['Rolling_Mean_28_of_Lag_28'] = grouped.transform(
        lambda x: x.rolling(window=28, min_periods=28).mean()
    )
    
    # Rolling Volatility (Standard Deviation)
    df_long['Rolling_Std_28_of_Lag_28'] = grouped.transform(
        lambda x: x.rolling(window=28, min_periods=28).std()
    )
    
    return df_long


def clean_and_save(df_long: pd.DataFrame, output_path: str) -> None:
    """
    5. Final Cleaning & Ingestion
    Removes NaN rows caused by historical shift/rolling operations and saves
    the finalized features matrix to a CSV file.
    """
    logger.info("Executing final cleaning (dropping NaNs from shift/rolling)...")
    initial_rows = len(df_long)
    
    # Drop rows containing NaNs
    df_features = df_long.dropna().reset_index(drop=True)
    final_rows = len(df_features)
    dropped_rows = initial_rows - final_rows
    
    logger.info("Dropped %d rows containing NaNs. Remaining records: %d", dropped_rows, final_rows)
    logger.info("Saving training features dataset to %s...", output_path)
    
    df_features.to_csv(output_path, index=False)
    logger.info("✔ Feature Engineering Pipeline completed successfully!")


def main():
    logger.info("=== Starting Feature Engineering Pipeline ===")
    start_time = time.time()
    
    try:
        # Step 1: Preprocess transactional data
        df_agg = load_and_preprocess_data(TRAIN_PATH)
        
        # Step 2: Build continuous time-series matrix
        df_matrix = build_timeseries_matrix(df_agg, START_DATE, END_DATE)
        
        # Step 3: Melt wide matrix to long format (28 million rows)
        df_long = melt_matrix(df_matrix)
        
        # Step 4: Extract calendar/temporal features
        df_long = extract_temporal_features(df_long)
        
        # Step 5: Engineer safe lag features
        df_long = generate_lag_features(df_long)
        
        # Step 6: Engineer safe rolling features on Lag_28
        df_long = generate_rolling_features(df_long)
        
        # Step 7: Clean NaNs and persist training dataset
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        clean_and_save(df_long, FEATURE_OUTPUT_PATH)
        
        logger.info("=== Pipeline Completed Successfully in %.2f seconds! ===", time.time() - start_time)
        
    except Exception as e:
        logger.exception("Feature engineering pipeline failed due to an error: %s", str(e))


if __name__ == "__main__":
    main()
