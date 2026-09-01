"""
Continuous Deployment (CD) and Verification Suite.
Validates the generated predictions.csv artifact against strict structural,
temporal, and physical integrity contracts, and serializes production model artifacts.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


def verify_prediction_artifact(csv_path: str = "predictions.csv") -> bool:
    """Verifies that predictions.csv meets all output contracts."""
    if not os.path.exists(csv_path):
        print(f"FAILED: File {csv_path} does not exist.")
        return False
        
    df_pred = pd.read_csv(csv_path)
    
    # 1. Header schema validation
    if list(df_pred.columns) != ["timestamp", "Reading"]:
        print(f"FAILED: Incorrect schema columns: {df_pred.columns}")
        return False
        
    # 2. Horizon step count validation (3 Hours at 5-second frequency = 2,160 rows)
    if len(df_pred) != 2160:
        print(f"FAILED: Expected exactly 2160 rows, got {len(df_pred)}")
        return False
    
    # 3. Monotonicity & regularity validation
    df_pred["timestamp"] = pd.to_datetime(df_pred["timestamp"])
    if not df_pred["timestamp"].is_monotonic_increasing:
        print("FAILED: Timestamps are not monotonically increasing.")
        return False
    
    time_diffs = df_pred["timestamp"].diff().dropna()
    if not (time_diffs == pd.Timedelta(seconds=5)).all():
        print("FAILED: Temporal step intervals are not strictly 5 seconds.")
        return False
        
    # 4. Exact forecast boundary validation
    expected_start = pd.Timestamp("2025-10-09 00:00:00")
    expected_end = pd.Timestamp("2025-10-09 02:59:55")
    if df_pred["timestamp"].min() != expected_start or df_pred["timestamp"].max() != expected_end:
        print(f"FAILED: Expected range [{expected_start} to {expected_end}], got [{df_pred['timestamp'].min()} to {df_pred['timestamp'].max()}]")
        return False
        
    # 5. Completeness and non-negativity physical bounds
    if df_pred["Reading"].isnull().sum() > 0 or np.isinf(df_pred["Reading"]).sum() > 0:
        print("FAILED: Contains NaN or Infinite values.")
        return False
    if df_pred["Reading"].min() < 0.0:
        print(f"FAILED: Physical non-negativity violated (min reading = {df_pred['Reading'].min():.4f})")
        return False
        
    print(f"SUCCESS: {csv_path} passed all 6 Continuous Deployment gating tests!")
    print(f"Summary: 2,160 rows | Range: {expected_start} -> {expected_end} | Min: {df_pred['Reading'].min():.2f}, Max: {df_pred['Reading'].max():.2f}, Mean: {df_pred['Reading'].mean():.2f}")
    return True


def serialize_production_model(model_obj, output_dir: str = "production_release"):
    """Serializes the trained production forecaster and writes metadata JSON."""
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "production_lightgbm_model.pkl")
    if HAS_JOBLIB:
        joblib.dump(model_obj, model_path)
    else:
        with open(model_path, "wb") as f:
            pickle.dump(model_obj, f)
            
    meta_path = os.path.join(output_dir, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump({
            "model_architecture": "Direct Multi-Horizon LightGBM Regressor",
            "horizon_steps": 2160,
            "horizon_duration": "3 Hours",
            "temporal_resolution": "5 Seconds",
            "forecast_window_start": "2025-10-09 00:00:00",
            "forecast_window_end": "2025-10-09 02:59:55",
            "physical_constraints": "Reading >= 0.0",
            "evaluation_metrics": ["MAE", "RMSE", "MAPE", "WAPE", "R2"]
        }, f, indent=4)
    print(f"Model artifacts and metadata saved to {output_dir}/")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        verify_prediction_artifact(sys.argv[1])
    else:
        verify_prediction_artifact("predictions.csv")
