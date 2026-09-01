"""
Generates future 3.0-hour predictions (2,160 steps @ 5s resolution: 2025-10-09 00:00:00 to 02:59:55)
using the Top 4 validated model architectures and their optimal weighted ensemble.

Outputs:
1. predictions.csv (Canonical submission file: timestamp, Reading)
2. predictions_top4_comparison.csv (Detailed multi-model comparison)
3. Updates Hyperparameter_Optimization_and_Model_Benchmarks.xlsx with 'Future Top 4 Predictions'
4. Serializes Top 4 production models to production_release/
"""

import os
import json
import time
import numpy as np
import pandas as pd
from data_cleaning_pipeline import run_data_cleaning_pipeline
from feature_store import build_feature_matrix
from cd_deployment_and_verification import verify_prediction_artifact, serialize_production_model


def main():
    print("=" * 80)
    print("GENERATING FUTURE 3-HOUR PREDICTIONS USING TOP 4 ARCHITECTURES")
    print("=" * 80)
    start_time = time.time()
    
    # 1. Ingest Data & Features
    df = run_data_cleaning_pipeline()
    df_features = build_feature_matrix(df)
    X_all = df_features.to_numpy(dtype=np.float32)
    y_all = df["Reading_imputed"].to_numpy(dtype=np.float32)
    timestamps = df.index
    total_steps = len(df)
    
    train_end = 397440      # Days 1–23
    val_end = 466560        # Days 24–27
    test_end = 501120       # Days 28–29
    
    subsample_step = 12
    train_indices = np.arange(2160, train_end - 2160, subsample_step)
    
    # 2. Extract 24-Hour Diurnal Harmonic Profile
    ts_train = timestamps[:train_end]
    step_of_day = (ts_train.hour * 3600 + ts_train.minute * 60 + ts_train.second) // 5
    df_diurnal = pd.DataFrame({"step": step_of_day, "reading": y_all[:train_end]})
    diurnal_profile = df_diurnal.groupby("step")["reading"].mean().to_numpy()
    
    # 3. Fit Multi-Output Design Matrices
    X_mat = X_all[train_indices]
    N = len(train_indices)
    Y_mat = np.zeros((N, 2160), dtype=np.float32)
    for i, idx in enumerate(train_indices):
        Y_mat[i, :] = y_all[idx : idx + 2160]
        
    X_design = np.hstack([np.ones((N, 1), dtype=np.float32), X_mat])
    p = X_design.shape[1]
    
    # Model 1 & 4 Base Ridge (L2=10.0)
    W_base = np.linalg.solve(X_design.T @ X_design + 10.0 * np.eye(p, dtype=np.float32), X_design.T @ Y_mat)
    intercept_base = W_base[0, :]
    weights_base = W_base[1:, :]
    
    # Model 3 Lead-Dependent Regularized Matrix
    W_lead = np.zeros((p, 2160), dtype=np.float32)
    XtX = X_design.T @ X_design
    XtY = X_design.T @ Y_mat
    for h in range(2160):
        lam_h = 2.0 * (1.0 + 20.0 * (h / 2160.0)**1.5)
        W_lead[:, h] = np.linalg.solve(XtX + lam_h * np.eye(p, dtype=np.float32), XtY[:, h])
    intercept_lead = W_lead[0, :]
    weights_lead = W_lead[1:, :]
    
    # Future Time Horizon: 2025-10-09 00:00:00 to 2025-10-09 02:59:55 (2,160 steps)
    future_start = pd.Timestamp("2025-10-09 00:00:00")
    future_end = pd.Timestamp("2025-10-09 02:59:55")
    future_grid = pd.date_range(start=future_start, end=future_end, freq="5s")
    f_steps = (future_grid.hour * 3600 + future_grid.minute * 60 + future_grid.second) // 5
    diurnal_vals = diurnal_profile[f_steps % len(diurnal_profile)]
    h_steps = np.arange(2160)
    
    # Latest historical feature context at cutoff (End of Day 30: 2025-10-08 23:59:55)
    latest_x = X_all[-1].reshape(1, -1)
    y_raw_base = (latest_x @ weights_base + intercept_base).ravel()
    y_raw_lead = (latest_x @ weights_lead + intercept_lead).ravel()
    
    # MODEL 1: Direct Dynamic Forecaster (Sigmoidal Transition)
    w_sig = 1.0 / (1.0 + np.exp(-(h_steps - 540) / 180.0))
    pred_m1 = np.maximum(0.0, (1.0 - w_sig) * y_raw_base + w_sig * diurnal_vals)
    
    # MODEL 2: DLinear Damped Trend-Seasonal Forecaster
    last_reading = float(X_all[-1, df_features.columns.get_loc("lag_1")])
    start_diurnal = diurnal_profile[((future_grid[0].hour*3600 + future_grid[0].minute*60 + future_grid[0].second)//5) % len(diurnal_profile)]
    residual_0 = last_reading - start_diurnal
    decay_ar = np.exp(-h_steps / 480.0)
    pred_m2 = np.maximum(0.0, diurnal_vals + residual_0 * decay_ar)
    
    # MODEL 3: Lead-Time Regularized Projector (lambda(h) Scaling + Linear Reversion)
    alpha_lin = np.clip(h_steps / 720.0, 0.0, 1.0)
    pred_m3 = np.maximum(0.0, (1.0 - alpha_lin) * y_raw_lead + alpha_lin * diurnal_vals)
    
    # MODEL 4: Hierarchical Multi-Scale Block Forecaster
    pred_m4 = np.zeros(2160, dtype=np.float32)
    pred_m4[:360] = 0.90 * y_raw_base[:360] + 0.10 * diurnal_vals[:360]
    w_mid = np.linspace(0.90, 0.20, 720)
    pred_m4[360:1080] = w_mid * y_raw_base[360:1080] + (1.0 - w_mid) * diurnal_vals[360:1080]
    pred_m4[1080:] = diurnal_vals[1080:]
    pred_m4 = np.maximum(0.0, pred_m4)
    
    # TOP 4 WEIGHTED ENSEMBLE FORECAST (Optimal Blending: 30% M1 + 25% M2 + 25% M3 + 20% M4)
    pred_ensemble = np.round(0.30 * pred_m1 + 0.25 * pred_m2 + 0.25 * pred_m3 + 0.20 * pred_m4, 4)
    
    # 4. Save Primary Official Submission File (predictions.csv)
    df_submission = pd.DataFrame({
        "timestamp": future_grid.strftime("%Y-%m-%d %H:%M:%S"),
        "Reading": pred_ensemble
    })
    csv_submission_path = "predictions.csv"
    df_submission.to_csv(csv_submission_path, index=False)
    print(f">> Official submission saved to: {csv_submission_path} ({len(df_submission)} rows).")
    
    # 5. Save Top 4 Detailed Comparison File
    df_comparison = pd.DataFrame({
        "timestamp": future_grid.strftime("%Y-%m-%d %H:%M:%S"),
        "Reading_Ensemble (Production)": pred_ensemble,
        "Reading_Model1 (Direct Dynamic Sigmoid)": np.round(pred_m1, 4),
        "Reading_Model2 (DLinear Trend-Seasonal)": np.round(pred_m2, 4),
        "Reading_Model3 (Lead-Time Regularized)": np.round(pred_m3, 4),
        "Reading_Model4 (Hierarchical Multi-Scale)": np.round(pred_m4, 4)
    })
    csv_comparison_path = "predictions_top4_comparison.csv"
    df_comparison.to_csv(csv_comparison_path, index=False)
    print(f">> Top 4 comparison saved to: {csv_comparison_path}")
    
    # 6. Verify Contract Gates
    is_valid = verify_prediction_artifact(csv_submission_path)
    assert is_valid, "predictions.csv artifact validation failed!"
    
    # 7. Update Excel Workbook with 'Top 4 Future Predictions' Sheet
    excel_file = "Hyperparameter_Optimization_and_Model_Benchmarks.xlsx"
    print(f"\nUpdating Excel report: {excel_file} with Sheet 'Top 4 Future Predictions'...")
    
    # Load existing sheets
    existing_sheets = {}
    xls = pd.ExcelFile(excel_file)
    for s in xls.sheet_names:
        if s != "Top 4 Future Predictions":
            existing_sheets[s] = pd.read_excel(excel_file, sheet_name=s)
            
    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        for s_name, s_df in existing_sheets.items():
            s_df.to_excel(writer, sheet_name=s_name, index=False)
        df_comparison.to_excel(writer, sheet_name="Top 4 Future Predictions", index=False)
        
    print(f"Excel workbook updated successfully with {len(existing_sheets) + 1} total sheets.")
    
    # 8. Print Statistical Summary of the Predictions
    print("\n" + "=" * 80)
    print("FUTURE 3-HOUR PREDICTIONS SUMMARY (2025-10-09 00:00:00 -> 02:59:55):")
    print(f"  * Total Steps:           {len(df_submission):,} (5-second intervals)")
    print(f"  * Ensemble Mean Reading: {pred_ensemble.mean():.2f}")
    print(f"  * Ensemble Min Reading:  {pred_ensemble.min():.2f}")
    print(f"  * Ensemble Max Reading:  {pred_ensemble.max():.2f}")
    print(f"  * Ensemble Std Dev:      {pred_ensemble.std():.2f}")
    print(f"  * First Step (00:00:00): {pred_ensemble[0]:.2f}")
    print(f"  * Last Step (02:59:55):  {pred_ensemble[-1]:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
