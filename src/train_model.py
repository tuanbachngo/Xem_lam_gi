"""
HBAAC Retail Time-Series Model Training Pipeline.

Author: Senior Data Scientist & Econometric Specialist
Description: PEP-8 compliant LightGBM Tweedie regressor training script.
             Includes high-performance memory downcasting, dtype-optimized 
             CSV ingestion to prevent Out-Of-Memory (OOM) on 12GB RAM, 
             chronological time-based validation splitting, and early stopping.
"""

import os
import time
import logging
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants and Configurations
FEATURE_PATH = os.path.join("src", "train_features.csv")
MODELS_DIR = "models"
MODEL_OUTPUT_PATH = os.path.join(MODELS_DIR, "lgbm_tweedie.pkl")
VALIDATION_WINDOW_DAYS = 28


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Memory Downcasting Optimizer
    Iterates through all numerical columns of a dataframe and modifies their data
    types to the lowest safe precision to minimize RAM footprint.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    logger.info("Starting memory usage optimization. Initial RAM: {:.2f} MB".format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        # Skip Date and pre-casted categorical columns to preserve structures
        if pd.api.types.is_datetime64_any_dtype(df[col]) or col_type.name == 'category':
            continue
            
        if pd.api.types.is_numeric_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()
            
            if pd.api.types.is_integer_dtype(df[col]):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)  
            else:
                # Downcast floats to float32 (avoid float16 to prevent loss of precision)
                df[col] = df[col].astype(np.float32)
                
    end_mem = df.memory_usage().sum() / 1024**2
    logger.info("Memory usage after downcasting: {:.2f} MB".format(end_mem))
    logger.info("✔ RAM footprint reduced by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def load_optimized_dataset(file_path: str) -> pd.DataFrame:
    """
    2. Low-Memory Data Loading
    Loads the dataset by pre-specifying standard dtypes to prevent OOM spikes.
    Automatically parses dates and casts categorical features.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Engineered features file not found: {file_path}")
        
    logger.info("Initializing low-memory CSV ingestion for %s...", file_path)
    
    # Pre-define dtypes to prevent pandas parser memory spikes
    dtypes = {
        'ItemCode': 'category',
        'Target_Quantity': 'float32',
        'Day_of_week': 'category',
        'Month': 'category',
        'Is_Weekend': 'category',
        'Lag_28': 'float32',
        'Lag_35': 'float32',
        'Lag_42': 'float32',
        'Lag_56': 'float32',
        'Rolling_Mean_7_of_Lag_28': 'float32',
        'Rolling_Mean_28_of_Lag_28': 'float32',
        'Rolling_Std_28_of_Lag_28': 'float32'
    }
    
    start_time = time.time()
    df = pd.read_csv(
        file_path, 
        dtype=dtypes, 
        parse_dates=['Date']
    )
    logger.info("Loaded dataset in %.2f seconds.", time.time() - start_time)
    
    # Run structural memory downcasting
    df = reduce_mem_usage(df)
    
    return df


def split_chronological_data(df: pd.DataFrame) -> tuple:
    """
    3. Chronological Time-Based Splitting
    Splits the continuous time-series dataset into training and validation sets.
    The last 28 days are designated as the validation set (evaluating direct F1-F28).
    """
    logger.info("Performing chronological time-based train/validation splitting...")
    max_date = df['Date'].max()
    val_start_date = max_date - pd.Timedelta(days=VALIDATION_WINDOW_DAYS - 1)
    
    logger.info("Information Horizon Max Date: %s", max_date.strftime('%Y-%m-%d'))
    logger.info("Validation Window Start Date: %s", val_start_date.strftime('%Y-%m-%d'))
    
    # Split
    df_train = df[df['Date'] < val_start_date].copy()
    df_val = df[df['Date'] >= val_start_date].copy()
    
    logger.info("Train Set size: %d rows (Dates: %s to %s)", 
                len(df_train), 
                df_train['Date'].min().strftime('%Y-%m-%d'),
                df_train['Date'].max().strftime('%Y-%m-%d'))
    logger.info("Validation Set size: %d rows (Dates: %s to %s)", 
                len(df_val), 
                df_val['Date'].min().strftime('%Y-%m-%d'),
                df_val['Date'].max().strftime('%Y-%m-%d'))
    
    # Features and Targets preparation (drop Date and Target_Quantity from features)
    X_train = df_train.drop(columns=['Date', 'Target_Quantity'])
    y_train = df_train['Target_Quantity']
    
    X_val = df_val.drop(columns=['Date', 'Target_Quantity'])
    y_val = df_val['Target_Quantity']
    
    return X_train, y_train, X_val, y_val


def train_lightgbm_tweedie(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> lgb.LGBMRegressor:
    """
    4. LightGBM Tweedie Training
    Trains the LightGBM regressor using the compound Tweedie distribution loss
    specifically suited for sparse retail sales.
    """
    logger.info("Configuring LightGBM Regressor with Tweedie loss...")
    
    # Initialize the LGBMRegressor
    model = lgb.LGBMRegressor(
        objective='tweedie',
        metric='rmse',
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    categorical_features = ['ItemCode', 'Day_of_week', 'Month', 'Is_Weekend']
    
    logger.info("Starting model training with early stopping...")
    start_time = time.time()
    
    # Fit the model using the recommended callbacks interface
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        categorical_feature=categorical_features,
        callbacks=[
            early_stopping(stopping_rounds=50, verbose=True),
            log_evaluation(period=10)
        ]
    )
    
    logger.info("✔ Training completed in %.2f seconds.", time.time() - start_time)
    return model


def persist_trained_model(model: lgb.LGBMRegressor, output_path: str) -> None:
    """
    5. Save Persisted Model
    Saves the final trained LightGBM regressor using joblib serialization.
    """
    logger.info("Saving trained model to %s...", output_path)
    joblib.dump(model, output_path)
    logger.info("✔ Model successfully persisted and ready for inference!")


def main():
    logger.info("=== Starting Model Training Pipeline ===")
    overall_start = time.time()
    
    try:
        # Step 1: Ingest features with high memory optimization
        df = load_optimized_dataset(FEATURE_PATH)
        
        # Step 2: Perform chronological split
        X_train, y_train, X_val, y_val = split_chronological_data(df)
        
        # Step 3: Train LightGBM Model with Tweedie loss
        model = train_lightgbm_tweedie(X_train, y_train, X_val, y_val)
        
        # Step 4: Persist final model
        os.makedirs(MODELS_DIR, exist_ok=True)
        persist_trained_model(model, MODEL_OUTPUT_PATH)
        
        logger.info("=== Model Training Completed Successfully in %.2f seconds! ===", time.time() - overall_start)
        
    except Exception as e:
        logger.exception("Model training pipeline failed due to an error: %s", str(e))


if __name__ == "__main__":
    main()
