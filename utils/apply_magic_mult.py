import pandas as pd
import numpy as np
from pathlib import Path

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Set your magic multiplier here based on your CV sweep results
MAGIC_MULT = 1.02  

# File paths
INPUT_SUBMISSION = "post_process_submission/submission_experiment_new.csv"  # The path to your original submission
OUTPUT_SUBMISSION = f"post_process_submission/submission_x{MAGIC_MULT}.csv"


def main():
    print("=" * 60)
    print(f" APPLYING MAGIC MULTIPLIER: {MAGIC_MULT}x")
    print("=" * 60)
    
    input_path = Path(INPUT_SUBMISSION)
    if not input_path.exists():
        print(f"❌ Error: Could not find input file '{INPUT_SUBMISSION}'.")
        print("Please check the INPUT_SUBMISSION path.")
        return

    print(f"1. Loading original submission: {INPUT_SUBMISSION}...")
    df = pd.read_csv(input_path)
    
    # Identify prediction columns (usually F1, F2, ..., F28 or similar)
    # We select all columns except the ID column ('id' or 'ItemCode')
    pred_cols = [c for c in df.columns if c not in ["ItemCode", "id"]]
    print(f"   -> Found {len(pred_cols)} prediction columns.")
    
    print(f"2. Multiplying all predictions by {MAGIC_MULT}...")
    # Multiply and ensure we don't accidentally create negative values or NaNs
    df[pred_cols] = (df[pred_cols] * MAGIC_MULT).clip(lower=0.0).fillna(0.0)
    
    # Create parent folder if not exists
    output_path = Path(OUTPUT_SUBMISSION)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"3. Saving boosted submission to: {OUTPUT_SUBMISSION}...")
    df.to_csv(output_path, index=False)
    
    print("✅ Done!")
    print("=" * 60)

if __name__ == "__main__":
    main()
