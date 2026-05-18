"""
HBAAC Retail Time-Series Inference & Submission Pipeline.

Author: Senior Data Scientist & Retail Analytics Specialist
Description: PEP-8 compliant model inference script. Imports feature engineering
             functions from src.feature to prevent training-serving skew.
             Loads the trained LightGBM Tweedie regressor, constructs safe future 
             features for the validation window (2025-09-06 to 2025-10-03), and
             generates the final submission.csv mapped to the template.
"""

import os
import sys
import time
import logging
import joblib
import numpy as np
import pandas as pd

# Add current workspace to Python path to ensure module import robustness
sys.path.append('.')

# Import feature pipeline components to maintain perfect consistency (avoid skew)
from src.feature import (
    load_and_preprocess_data,
    build_timeseries_matrix,
    melt_matrix,
    extract_temporal_features,
    generate_lag_features,
    generate_rolling_features
)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants and Configuration
TRAIN_PATH = os.path.join("hbaac-round2", "train.csv")
SAMPLE_SUB_PATH = os.path.join("hbaac-round2", "sample_submission.csv")
MODEL_PATH = os.path.join("models", "lgbm_tweedie.pkl")
OUTPUT_SUB_PATH = "submission.csv"

START_DATE = "2020-11-17"
TEST_START_DATE = "2025-09-06"
TEST_END_DATE = "2025-10-03"  # 28 days validation window


def main():
    logger.info("=== Starting Inference & Submission Generation ===")
    overall_start = time.time()
    
    try:
        # Step 1: Preprocess transactional data
        df_agg = load_and_preprocess_data(TRAIN_PATH)
        
        # Step 2: Build extended timeseries matrix up to 2025-10-03 (future date range)
        # This will automatically initialize the future dates as zero, which is correct
        # since their features only look backward (Lag_28 and older)
        logger.info("Building extended time-series matrix up to %s...", TEST_END_DATE)
        df_matrix = build_timeseries_matrix(df_agg, START_DATE, TEST_END_DATE)
        
        # Step 3: Run DRY feature engineering pipeline
        logger.info("Generating features on extended timeline...")
        df_long = melt_matrix(df_matrix)
        df_long = extract_temporal_features(df_long)
        df_long = generate_lag_features(df_long)
        df_long = generate_rolling_features(df_long)
        
        # Step 4: Filter for the validation window (Test Set)
        logger.info("Filtering for test set (Date range: %s to %s)...", TEST_START_DATE, TEST_END_DATE)
        df_test = df_long[df_long['Date'] >= TEST_START_DATE].copy()
        
        # Check for NaNs to ensure feature alignment and zero leakage
        nan_counts = df_test.isnull().sum().sum()
        logger.info("Validation features verified. Total NaN values in test set: %d", nan_counts)
        if nan_counts > 0:
            logger.warning("Warning: NaN values found in test features! Filling with 0.0.")
            df_test = df_test.fillna(0.0)
            
        # Prepare test features (ensure exact column ordering as X_train)
        X_test = df_test.drop(columns=['Date', 'Target_Quantity'])
        
        # Explicitly cast categorical columns to 'category'
        categorical_cols = ['ItemCode', 'Day_of_week', 'Month', 'Is_Weekend']
        for col in categorical_cols:
            X_test[col] = X_test[col].astype('category')
            
        # Step 5: Load LightGBM Tweedie Model & Run Predict
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Trained LightGBM model not found at: {MODEL_PATH}")
            
        logger.info("Loading LightGBM Tweedie model from %s...", MODEL_PATH)
        model = joblib.load(MODEL_PATH)
        
        logger.info("Running model predictions on %d test records...", len(X_test))
        preds = model.predict(X_test)
        
        # Enforce non-negative constraint
        df_test['Prediction_Quantity'] = np.clip(preds, a_min=0.0, a_max=None)
        
        # Step 6: Pivot predictions back to wide format (ItemCode as index, Date as columns)
        logger.info("Pivoting predictions to wide submission format...")
        df_pivot_pred = df_test.pivot(index='ItemCode', columns='Date', values='Prediction_Quantity')
        
        # Sort and map dates to F1-F28 columns
        dates_sorted = sorted(df_test['Date'].unique())
        date_to_f_map = {date: f"F{i+1}" for i, date in enumerate(dates_sorted)}
        df_pivot_pred.columns = [date_to_f_map[col] for col in df_pivot_pred.columns]
        
        # Step 7: Map validation predictions to sample submission template
        if not os.path.exists(SAMPLE_SUB_PATH):
            raise FileNotFoundError(f"Missing submission template: {SAMPLE_SUB_PATH}")
            
        logger.info("Loading submission template from %s...", SAMPLE_SUB_PATH)
        submission = pd.read_csv(SAMPLE_SUB_PATH, index_col=0)
        
        logger.info("Mapping predicted sales quantities to validation rows...")
        submission_idx = submission.index.to_series()
        parts = submission_idx.str.rsplit('_', n=1)
        item_codes = parts.str[0]
        window_types = parts.str[1]
        
        for col in [f"F{i}" for i in range(1, 29)]:
            # Map predictions from pivot dataframe
            mapped_preds = item_codes.map(df_pivot_pred[col]).fillna(0.0)
            
            # Enforce validation predictions, leaving evaluation rows strictly 0.0
            submission[col] = np.where(window_types == 'validation', mapped_preds, 0.0)
            
        # Step 8: Persist final submission file
        logger.info("Saving final formatted submission to %s...", OUTPUT_SUB_PATH)
        submission.to_csv(OUTPUT_SUB_PATH, index_label='id')
        
        logger.info("=== Inference & Submission Completed Successfully in %.2f seconds! ===", 
                    time.time() - overall_start)
        
    except Exception as e:
        logger.exception("Inference pipeline failed due to an error: %s", str(e))


if __name__ == "__main__":
    main()
