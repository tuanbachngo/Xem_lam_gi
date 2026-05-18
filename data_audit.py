"""
HBAAC Retail Time-Series Data Quality Audit Tool.

Author: Data Scientist & Retail Analytics Specialist
Description: Production-ready diagnostics script to audit raw transaction data quality.
             Analyzes missing values, exact zero values, and negative quantity returns
             using clean, standardized Pandas transformations.
"""

import os
import time
import pandas as pd
import numpy as np

# Absolute file paths (relative to root directory)
TRAIN_PATH = os.path.join("hbaac-round2", "train.csv")


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


def perform_data_audit(file_path: str) -> None:
    """
    Executes the data quality audit on train.csv and prints a formatted terminal report.
    """
    if not os.path.exists(file_path):
        print(f"[-] Error: Target data file not found at: {file_path}")
        return

    start_time = time.time()
    
    # -------------------------------------------------------------
    # 1. Dataset Ingestion
    # -------------------------------------------------------------
    print("=" * 85)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] INITIALIZING RETAIL DATA QUALITY AUDIT")
    print(f"Loading raw dataset: {file_path}")
    print("=" * 85)
    
    # Read CSV with low_memory=False to prevent mixed dtype warnings
    df_raw = pd.read_csv(file_path, low_memory=False)
    load_duration = time.time() - start_time
    total_rows = len(df_raw)
    
    print(f"✔ Dataset loaded successfully in {load_duration:.2f} seconds.")
    print(f"✔ Dimensions: {total_rows:,} rows | {len(df_raw.columns)} columns")
    print("-" * 85)

    # -------------------------------------------------------------
    # 2. Missing Values Analysis (NaN / Null)
    # -------------------------------------------------------------
    print("\n--- 1. MISSING VALUES ANALYSIS (Null / NaN) ---")
    print(f"{'Column Name':<25} | {'Null Count':<15} | {'Percentage (%)':<15}")
    print("-" * 65)
    
    null_counts = df_raw.isnull().sum()
    for col in df_raw.columns:
        cnt = null_counts[col]
        pct = (cnt / total_rows) * 100
        print(f"{col:<25} | {cnt:<15,} | {pct:<15.4f}%")
    print("-" * 65)

    # -------------------------------------------------------------
    # 3. Data Cleaning & Parsing for Quantitative Audit Checks
    # -------------------------------------------------------------
    df_parsed = df_raw.copy()
    
    # Standardize numeric inputs
    if 'Quantity' in df_parsed.columns:
        df_parsed['Quantity'] = pd.to_numeric(df_parsed['Quantity'], errors='coerce').fillna(0.0)
        
    if 'SalesAmount' in df_parsed.columns:
        df_parsed['SalesAmount'] = parse_monetary_column(df_parsed['SalesAmount'])

    # -------------------------------------------------------------
    # 4. Exact Zero Values Analysis
    # -------------------------------------------------------------
    print("\n--- 2. EXACT ZERO VALUES ANALYSIS ---")
    print(f"{'Target Column':<25} | {'Zero Count':<15} | {'Percentage (%)':<15}")
    print("-" * 65)
    
    for col in ['Quantity', 'SalesAmount']:
        if col in df_parsed.columns:
            zero_count = (df_parsed[col] == 0).sum()
            zero_pct = (zero_count / total_rows) * 100
            print(f"{col:<25} | {zero_count:<15,} | {zero_pct:<15.4f}%")
        else:
            print(f"{col:<25} | {'Not Found':<15} | {'N/A':<15}")
    print("-" * 65)

    # -------------------------------------------------------------
    # 5. Negative Values (Return Transactions) Analysis
    # -------------------------------------------------------------
    print("\n--- 3. NEGATIVE VALUES (RETURN TRANSACTIONS) ANALYSIS ---")
    print(f"{'Target Column':<25} | {'Negative Count':<15} | {'Percentage (%)':<15}")
    print("-" * 65)
    
    if 'Quantity' in df_parsed.columns:
        neg_count = (df_parsed['Quantity'] < 0).sum()
        neg_pct = (neg_count / total_rows) * 100
        print(f"{'Quantity':<25} | {neg_count:<15,} | {neg_pct:<15.4f}%")
    else:
        print(f"{'Quantity':<25} | {'Not Found':<15} | {'N/A':<15}")
    print("-" * 65)
    
    # -------------------------------------------------------------
    # 6. Audit Finalization
    # -------------------------------------------------------------
    print("\n" + "=" * 85)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] AUDIT COMPLETED IN {time.time() - start_time:.2f} SECONDS")
    print("=" * 85)


if __name__ == "__main__":
    perform_data_audit(TRAIN_PATH)
