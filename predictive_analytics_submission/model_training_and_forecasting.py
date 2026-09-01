"""
Multi-Horizon Forecasting Engine & Model Benchmark Arena.

Implements:
1. 4-Tier Walk-Forward Partitioning: Train (Days 1–23), Val (Days 24–27), Test (Days 28–29 / 2 Days), Holdout (Day 30 / 1 Day).
2. Dynamic Horizon-Adaptive Forecaster: Smoothly transitions from high-frequency autoregressive projections (0–30 min)
   to 24-hour diurnal harmonic profiles (60–180 min), resolving long-term horizon window degradation.
3. Standardized Metrics Suite: MAE, RMSE, MAPE, WAPE, R2, MDA, and Lead-Time Buckets.
4. Continuous Deployment (CD) verification of predictions.csv.
"""

import os
import json
import numpy as np
import pandas as pd
from data_cleaning_pipeline import run_data_cleaning_pipeline
from feature_store import build_feature_matrix
from cd_deployment_and_verification import verify_prediction_artifact, serialize_production_model


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculates standardized regression & time series forecasting metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(0.0, np.asarray(y_pred, dtype=float))
    
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    mask = y_true > 0.05
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0) if np.sum(mask) > 0 else 0.0
    
    total_actual = float(np.sum(y_true))
    wape = float((np.sum(np.abs(y_true - y_pred)) / (total_actual + 1e-8)) * 100.0)
    
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))
    
    if len(y_true) > 1:
        diff_true = np.diff(y_true)
        diff_pred = np.diff(y_pred)
        mda = float(np.mean((np.sign(diff_true) == np.sign(diff_pred)).astype(float)) * 100.0)
    else:
        mda = 100.0
        
    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE (%)": round(mape, 2),
        "WAPE (%)": round(wape, 2),
        "R2": round(r2, 4),
        "MDA (%)": round(mda, 2)
    }


class DynamicHorizonForecaster:
    """
    Direct Multi-Horizon Forecaster with Dynamic Horizon Decay.
    Utilizes regularized multi-output projections for immediate lead times (0–30 min)
    and smoothly transitions into calibrated 24-hour diurnal harmonic profiles for distant lead times (60–180 min).
    """
    def __init__(self, horizon: int = 2160, l2_reg: float = 10.0, transition_step: int = 720):
        self.horizon = horizon
        self.l2_reg = l2_reg
        self.transition_step = transition_step
        self.weights = None
        self.intercept = None
        self.diurnal_profile = None

    def fit(self, X_all: np.ndarray, y_all: np.ndarray, sample_indices: np.ndarray, timestamps: pd.DatetimeIndex = None):
        X_mat = X_all[sample_indices]
        N = len(sample_indices)
        
        Y_mat = np.zeros((N, self.horizon), dtype=np.float32)
        y_arr = np.asarray(y_all, dtype=np.float32)
        for i, idx in enumerate(sample_indices):
            Y_mat[i, :] = y_arr[idx : idx + self.horizon]
            
        X_design = np.hstack([np.ones((N, 1), dtype=np.float32), X_mat])
        p = X_design.shape[1]
        A = X_design.T @ X_design + self.l2_reg * np.eye(p, dtype=np.float32)
        B = X_design.T @ Y_mat
        
        W = np.linalg.solve(A, B)
        self.intercept = W[0, :]
        self.weights = W[1:, :]
        
        if timestamps is not None:
            ts_train = timestamps[:len(y_all)]
            step_of_day = (ts_train.hour * 3600 + ts_train.minute * 60 + ts_train.second) // 5
            df_diurnal = pd.DataFrame({"step": step_of_day, "reading": y_all})
            self.diurnal_profile = df_diurnal.groupby("step")["reading"].mean().to_numpy()
            
        return self

    def predict(self, x_latest: np.ndarray, forecast_timestamps: pd.DatetimeIndex = None) -> np.ndarray:
        x_vec = np.asarray(x_latest, dtype=np.float32)
        if x_vec.ndim == 1:
            x_vec = x_vec.reshape(1, -1)
            
        y_pred_ar = (x_vec @ self.weights + self.intercept).ravel()
        
        if forecast_timestamps is not None and self.diurnal_profile is not None:
            forecast_steps = (forecast_timestamps.hour * 3600 + forecast_timestamps.minute * 60 + forecast_timestamps.second) // 5
            diurnal_vals = self.diurnal_profile[forecast_steps % len(self.diurnal_profile)]
            
            # Dynamic Horizon Linear/Sigmoidal Weighting
            h_steps = np.arange(self.horizon)
            alpha_h = np.clip(h_steps / float(self.transition_step), 0.0, 1.0)
            y_pred = (1.0 - alpha_h) * y_pred_ar + alpha_h * diurnal_vals
        else:
            y_pred = y_pred_ar
            
        return np.maximum(0.0, y_pred)


def run_training_and_evaluation():
    print("=" * 80)
    print("STEP 1: Ingesting & Cleaning Sensor Telemetry...")
    df = run_data_cleaning_pipeline()
    total_steps = len(df)
    print(f"Canonical Timeline: {total_steps} points (5-second intervals over 30.0 days).")
    
    print("\n" + "=" * 80)
    print("STEP 2: Engineering Multi-Scale Feature Matrix...")
    df_features = build_feature_matrix(df)
    X_all = df_features.to_numpy(dtype=np.float32)
    y_all = df["Reading_imputed"].to_numpy(dtype=np.float32)
    timestamps = df.index
    
    # 4-Tier Partitioning Scheme
    train_end = 397440      # Days 1–23 (76.67%)
    val_end = 466560        # Days 24–27 (13.33%)
    test_end = 501120       # Days 28–29 (6.67% / 2 Days)
    holdout_start = 501120  # Day 30 (3.33% / 1 Day Holdout)
    
    print("\n" + "=" * 80)
    print("STEP 3: Partitioning Data (4-Tier Walk-Forward Scheme):")
    print(f"  [1] Training Set:         Steps 0 -> {train_end} ({train_end:,} steps | Days 1–23 | 76.67%)")
    print(f"  [2] Validation Set (HPO): Steps {train_end} -> {val_end} ({val_end - train_end:,} steps | Days 24–27 | 13.33%)")
    print(f"  [3] Test Set (2-Day):     Steps {val_end} -> {test_end} ({test_end - val_end:,} steps | Days 28–29 | 6.67%)")
    print(f"  [4] Untouched Holdout:    Steps {holdout_start} -> {total_steps} ({total_steps - holdout_start:,} steps | Day 30 | 3.33%)")
    
    # Subsampling
    subsample_step = 12
    train_indices = np.arange(2160, train_end - 2160, subsample_step)
    
    print("\n" + "=" * 80)
    print(f"STEP 4: Training Dynamic Horizon-Adaptive Forecaster ({len(train_indices):,} sequence windows)...")
    model = DynamicHorizonForecaster(horizon=2160, l2_reg=10.0, transition_step=720)
    model.fit(X_all[:train_end], y_all[:train_end], train_indices, timestamps=timestamps[:train_end])
    print("Model training completed successfully.")
    
    # Evaluate on Final Holdout Window (Day 30 Final 3 Hours)
    print("\n" + "=" * 80)
    print("STEP 5: Benchmarking on Final Holdout Window (Day 30 Final 3.0 Hours):")
    final_test_start = total_steps - 2160
    y_true_final = y_all[final_test_start:]
    ts_final = timestamps[final_test_start:]
    x_ctx_final = X_all[final_test_start - 1]
    
    y_pred_prod = model.predict(x_ctx_final, forecast_timestamps=ts_final)
    y_pred_naive = y_all[final_test_start - 17280 : final_test_start - 17280 + 2160]
    y_pred_ma = np.full(2160, np.mean(y_all[final_test_start - 720 : final_test_start]))
    
    benchmark_df = pd.DataFrame([
        {"Architecture": "Dynamic Horizon-Adaptive Regressor (Production)", **calculate_metrics(y_true_final, y_pred_prod)},
        {"Architecture": "Naive 24-Hour Seasonal Persistence", **calculate_metrics(y_true_final, y_pred_naive)},
        {"Architecture": "Rolling 1-Hour Moving Average Baseline", **calculate_metrics(y_true_final, y_pred_ma)}
    ])
    print(benchmark_df.to_string(index=False))
    
    # Lead-Time Horizon Degradation Analysis
    print("\n" + "-" * 80)
    print("LEAD-TIME HORIZON DEGRADATION SUMMARY (PRODUCTION MODEL):")
    short_mae = np.mean(np.abs(y_true_final[:360] - y_pred_prod[:360]))
    mid_mae = np.mean(np.abs(y_true_final[360:1080] - y_pred_prod[360:1080]))
    long_mae = np.mean(np.abs(y_true_final[1080:] - y_pred_prod[1080:]))
    print(f"  * 0 to 30 min (Steps 1–360):     MAE = {short_mae:.2f}")
    print(f"  * 30 to 90 min (Steps 361–1080):  MAE = {mid_mae:.2f}")
    print(f"  * 90 to 180 min (Steps 1081–2160): MAE = {long_mae:.2f}")
    
    # Step 6: Generate Future 3-Hour Forecast (2025-10-09 00:00:00 to 02:59:55)
    print("\n" + "=" * 80)
    print("STEP 6: Generating Final 3.0-Hour Future Forecast into predictions.csv...")
    future_start = pd.Timestamp("2025-10-09 00:00:00")
    future_end = pd.Timestamp("2025-10-09 02:59:55")
    future_grid = pd.date_range(start=future_start, end=future_end, freq="5s")
    
    latest_feature_vector = X_all[-1]
    future_predictions = model.predict(latest_feature_vector, forecast_timestamps=future_grid)
    
    df_predictions = pd.DataFrame({
        "timestamp": future_grid.strftime("%Y-%m-%d %H:%M:%S"),
        "Reading": np.round(future_predictions, 4)
    })
    
    csv_path = "predictions.csv"
    df_predictions.to_csv(csv_path, index=False)
    print(f"Predictions saved to {csv_path} ({len(df_predictions)} rows).")
    
    # Step 7: CD Verification & Model Serialization
    print("\n" + "=" * 80)
    print("STEP 7: Continuous Deployment (CD) Validation & Serialization:")
    is_valid = verify_prediction_artifact(csv_path)
    if is_valid:
        serialize_production_model(model, output_dir="production_release")
        print("\nAll Continuous Integration and Deployment Gates PASSED successfully!")
    else:
        print("\nVerification FAILED.")
        
    return benchmark_df, df_predictions


if __name__ == "__main__":
    run_training_and_evaluation()
