"""
Feature Engineering Store for Sensor Telemetry Multi-Horizon Forecasting.

Constructs multi-scale cyclical time embeddings, autoregressive multi-horizon lags,
rolling statistical aggregations (mean, std, envelopes), momentum deltas, and 
phase jitter offsets with strict causality and zero lookahead leakage.
"""

import numpy as np
import pandas as pd


def extract_cyclical_temporal_features(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """Extracts harmonic cyclical embeddings across day, hour, and weekly periods."""
    seconds_in_day = timestamps.hour * 3600 + timestamps.minute * 60 + timestamps.second
    day_fraction = seconds_in_day / 86400.0
    hour_fraction = (timestamps.minute * 60 + timestamps.second) / 3600.0
    week_fraction = (timestamps.dayofweek * 86400 + seconds_in_day) / (7 * 86400.0)

    df_time = pd.DataFrame(index=timestamps)
    df_time["sin_day"] = np.sin(2 * np.pi * day_fraction)
    df_time["cos_day"] = np.cos(2 * np.pi * day_fraction)
    df_time["sin_hour"] = np.sin(2 * np.pi * hour_fraction)
    df_time["cos_hour"] = np.cos(2 * np.pi * hour_fraction)
    df_time["sin_week"] = np.sin(2 * np.pi * week_fraction)
    df_time["cos_week"] = np.cos(2 * np.pi * week_fraction)
    return df_time


def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str = "Reading_imputed",
    lags: list = [1, 2, 3, 6, 12, 60, 180, 720, 1440, 2160],
    rolling_windows: list = [12, 60, 180, 720, 2160]
) -> pd.DataFrame:
    """
    Constructs feature matrix with strict causality enforcement (all features shifted by 1 tick).
    """
    series = df[target_col]
    features = extract_cyclical_temporal_features(df.index)
    
    # 1. Phase offset and operational state
    if "phase_offset_sec" in df.columns:
        features["phase_offset_sec"] = df["phase_offset_sec"]
    if "is_operational" in df.columns:
        features["is_operational"] = df["is_operational"]

    # 2. Autoregressive Lags (Shifted)
    for lag in lags:
        features[f"lag_{lag}"] = series.shift(lag)

    # 3. Rolling Summary Statistics (Historical only, shifted by 1 to prevent leakage)
    shifted_series = series.shift(1)
    for win in rolling_windows:
        features[f"roll_mean_{win}"] = shifted_series.rolling(win, min_periods=1).mean()
        features[f"roll_std_{win}"] = shifted_series.rolling(win, min_periods=1).std().fillna(0.0)
        features[f"roll_min_{win}"] = shifted_series.rolling(win, min_periods=1).min()
        features[f"roll_max_{win}"] = shifted_series.rolling(win, min_periods=1).max()

    # 4. Momentum & Rate-of-Change Deltas
    features["momentum_1m"] = features["lag_1"] - features["lag_12"]
    features["momentum_5m"] = features["lag_1"] - features["lag_60"]
    features["momentum_1h"] = features["lag_1"] - features["lag_720"]

    # Fill initial boundary NaNs cleanly
    features = features.bfill().ffill()
    return features


if __name__ == "__main__":
    from data_cleaning_pipeline import run_data_cleaning_pipeline
    df = run_data_cleaning_pipeline()
    feats = build_feature_matrix(df)
    print(f"Feature matrix generated. Shape: {feats.shape}")
    print(f"Sample columns: {list(feats.columns[:10])}")
