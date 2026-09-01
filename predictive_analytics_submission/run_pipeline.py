"""
Master Execution Pipeline for Sensor Telemetry Predictive Analytics.

Orchestrates:
1. Data cleaning & canonical 5-second grid reconstruction (data_cleaning_pipeline.py)
2. Continuous Integration unit testing suite (ci_pipeline_tests.py)
3. Multi-scale feature engineering (feature_store.py)
4. Multi-horizon model training & benchmarking (model_training_and_forecasting.py)
5. Continuous Deployment verification of predictions.csv (cd_deployment_and_verification.py)
"""

import sys
import os
import time
from data_cleaning_pipeline import run_data_cleaning_pipeline
from ci_pipeline_tests import run_ci_suite
from model_training_and_forecasting import run_training_and_evaluation
from cd_deployment_and_verification import verify_prediction_artifact


def main():
    print("=" * 80)
    print("      HIGH-FREQUENCY INDUSTRIAL SENSOR TELEMETRY FORECASTING PIPELINE      ")
    print("=" * 80)
    start_time = time.time()

    # Step 1: CI Unit Testing Suite
    print("\n[PHASE 1] Running 6-Gate Continuous Integration (CI) Test Suite...")
    ci_success = run_ci_suite()
    if not ci_success:
        print("ERROR: CI pipeline test suite failed. Aborting execution.")
        sys.exit(1)
    print(">> CI Phase Passed: All temporal, physical, and completeness gates verified.\n")

    # Step 2: Training, Benchmarking & Forecasting
    print("[PHASE 2] Executing Feature Engineering, Model Training & Benchmarking...")
    results_df, df_predictions = run_training_and_evaluation()

    # Step 3: CD Verification
    print("\n[PHASE 3] Running Continuous Deployment (CD) Artifact Verification...")
    cd_success = verify_prediction_artifact("predictions.csv")
    if not cd_success:
        print("ERROR: CD verification failed for predictions.csv.")
        sys.exit(1)
    print(">> CD Phase Passed: Prediction artifact meets all contractual requirements.\n")

    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"PIPELINE EXECUTION COMPLETED SUCCESSFULLY in {elapsed:.2f} seconds.")
    print("Deliverables Generated:")
    print("  1. predictions.csv (2,160 rows, 3-hour forecast @ 5s frequency)")
    print("  2. production_release/production_lightgbm_model.pkl (Serialized model)")
    print("  3. production_release/model_metadata.json (Architecture metadata)")
    print("=" * 80)


if __name__ == "__main__":
    main()
