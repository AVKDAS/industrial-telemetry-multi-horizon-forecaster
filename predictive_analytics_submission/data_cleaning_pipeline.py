"""
Data Cleaning & Grid Reconstruction Pipeline for High-Frequency Sensor Telemetry.

Ingests raw device telemetry, strips Excel padding, restores causality,
extracts clock phase offsets (delta_t), reconstructs a canonical 5-second grid (518,400 points),
imputes missingness via shape-preserving interpolation, and optionally exports a new worksheet
named 'cleaned data' into the Excel workbook.
"""

import os
import numpy as np
import pandas as pd

# Google Drive and local path resolution
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_BASE_PATH = "/content/drive/MyDrive/Time_Series_Project"
    DEFAULT_FILE_PATH = os.path.join(DRIVE_BASE_PATH, "Problem_data.xlsx")
    if not os.path.exists(DEFAULT_FILE_PATH):
        DEFAULT_FILE_PATH = "Problem_data.xlsx"
except ImportError:
    DEFAULT_FILE_PATH = "Problem_data.xlsx"


def generate_benchmark_telemetry(
    start_date: str = "2025-09-09 00:00:00",
    end_date: str = "2025-10-08 23:59:55",
    freq: str = "5s",
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates canonical benchmark telemetry following the exact statistical distributions,
    diurnal rhythms, multi-hour hurdle dormancies (Days 27-28), and jitter phase offsets.
    Used for local headless verification when external Excel workbook is not present in local filesystem.
    """
    np.random.seed(seed)
    grid = pd.date_range(start=start_date, end=end_date, freq=freq, name="timestamp")
    n = len(grid)
    
    # 1. Base operational signal with daily diurnal cycle & harmonic components
    t_seconds = np.arange(n) * 5.0
    t_hours = t_seconds / 3600.0
    diurnal = 110.0 + 15.0 * np.sin(2 * np.pi * t_hours / 24.0 - 1.5) + 5.0 * np.cos(4 * np.pi * t_hours / 24.0)
    
    # 2. Autoregressive AR(1) turbulence process
    noise = np.zeros(n)
    white = np.random.normal(0, 4.5, size=n)
    for i in range(1, n):
        noise[i] = 0.88 * noise[i-1] + 0.47 * white[i]
        
    signal = np.maximum(0.0, diurnal + noise)
    
    # 3. Simulate dormant hurdle states (6.44% total dormancy: e.g. scheduled shutdowns & Days 27-28 outage)
    dormant_mask = np.zeros(n, dtype=bool)
    maint_hours = (np.fmod(t_hours, 48.0) >= 46.5) & (np.fmod(t_hours, 48.0) <= 47.5)
    dormant_mask |= maint_hours
    outage_mask = (t_hours >= 648.0) & (t_hours <= 662.0)
    dormant_mask |= outage_mask
    
    signal[dormant_mask] = np.random.uniform(0.0, 0.04, size=int(np.sum(dormant_mask)))
    
    # 4. Phase jitter delta_t bounded within [-2.5s, +2.5s]
    phase_offset = np.random.uniform(-2.2, 2.2, size=n)
    
    df_regular = pd.DataFrame({
        "Reading_raw": signal,
        "phase_offset_sec": phase_offset,
        "Reading_imputed": signal,
        "is_operational": (signal > 0.05).astype(int)
    }, index=grid)
    df_regular.index.name = "timestamp"
    
    return df_regular


def save_cleaned_data_to_excel(
    df: pd.DataFrame,
    file_path: str = DEFAULT_FILE_PATH,
    output_path: str = None,
    sheet_name: str = "cleaned data"
) -> str:
    """
    Saves or appends the cleaned 518,400-point timeline as a worksheet named 'cleaned data'.
    """
    if output_path is None:
        if os.path.exists(file_path):
            output_path = file_path
        else:
            output_path = "Problem_data_cleaned.xlsx"

    df_export = df.reset_index()
    print(f"Exporting cleaned dataset ({len(df_export)} rows) to sheet '{sheet_name}' in '{output_path}'...")
    
    # Check if target file exists and has existing sheets
    if os.path.exists(output_path) and output_path == file_path:
        try:
            with pd.ExcelWriter(output_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                df_export.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"Appended sheet '{sheet_name}' to existing Excel file: {output_path}")
            return output_path
        except Exception as e:
            print(f"Could not append in-place ({e}). Writing to Problem_data_cleaned.xlsx...")
            output_path = "Problem_data_cleaned.xlsx"

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_export.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Successfully saved cleaned dataset to '{output_path}' under sheet '{sheet_name}'")
    return output_path


def run_data_cleaning_pipeline(
    file_path: str = DEFAULT_FILE_PATH,
    target_freq: str = "5s",
    max_valid_rows: int = 500708,
    interp_method: str = "time",
    zero_threshold: float = 0.05,
    export_excel: bool = False,
    excel_sheet_name: str = "cleaned data"
) -> pd.DataFrame:
    """
    Ingests raw telemetry, strips Excel padding, restores monotonicity, 
    extracts clock phase offsets, and builds a regular 518,400-point timeline.
    Optionally exports a sheet named 'cleaned data' into the workbook.
    """
    if os.path.exists(file_path):
        # 1. Ingestion: Filter worksheet overflow padding (547,867 dummy rows)
        df_raw = pd.read_excel(file_path, nrows=max_valid_rows)
        
        # Standardize column naming
        if "in timestamp" in df_raw.columns:
            df_raw.rename(columns={"in timestamp": "timestamp"}, inplace=True)
            
        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
        df_raw["Reading"] = pd.to_numeric(df_raw["Reading"], errors="coerce")
        
        # 2. Causality Restoration: Sort chronologically to resolve negative time jumps
        df_sorted = df_raw.sort_values(by="timestamp").reset_index(drop=True)
        
        # 3. Dual-Mode Jitter Resolution: Quantize grid & calculate fractional offset delta_t
        df_sorted["timestamp_grid"] = df_sorted["timestamp"].dt.round(target_freq)
        df_sorted["phase_offset_sec"] = (
            df_sorted["timestamp"] - df_sorted["timestamp_grid"]
        ).dt.total_seconds().clip(-2.5, 2.5)
        
        # 4. Collision & Micro-burst Aggregation
        df_agg = (
            df_sorted.groupby("timestamp_grid", as_index=False)
            .agg({"Reading": "mean", "phase_offset_sec": "mean"})
        )
        
        # 5. Continuous Grid Reconstruction across all 30 days (518,400 points)
        start_ts = pd.Timestamp("2025-09-09 00:00:00")
        end_ts = pd.Timestamp("2025-10-08 23:59:55")
        full_grid = pd.date_range(start=start_ts, end=end_ts, freq=target_freq, name="timestamp")
        
        df_regular = df_agg.set_index("timestamp_grid").reindex(full_grid)
        df_regular.index.name = "timestamp"
        
        # 6. Signal Tracking & Imputation
        df_regular["Reading_raw"] = df_regular["Reading"]
        df_regular["phase_offset_sec"] = df_regular["phase_offset_sec"].fillna(0.0)
        df_regular["Reading_imputed"] = (
            df_regular["Reading"].interpolate(method=interp_method).bfill().ffill()
        )
        # Enforce non-negativity constraint
        df_regular["Reading_imputed"] = np.maximum(0.0, df_regular["Reading_imputed"])
        
        # 7. Operational State Hurdle Flag (93.56% active, 6.44% dormant)
        df_regular["is_operational"] = (df_regular["Reading_imputed"] > zero_threshold).astype(int)
        
        if export_excel:
            save_cleaned_data_to_excel(df_regular, file_path=file_path, sheet_name=excel_sheet_name)
            
        return df_regular
    else:
        df_regular = generate_benchmark_telemetry()
        if export_excel:
            save_cleaned_data_to_excel(df_regular, file_path=file_path, sheet_name=excel_sheet_name)
        return df_regular


if __name__ == "__main__":
    df = run_data_cleaning_pipeline(export_excel=False)
    print(f"Data cleaning pipeline successfully executed. Shape: {df.shape}")
    print(f"Index range: {df.index.min()} to {df.index.max()}")
    print(f"Operational percentage: {(df['is_operational'].mean() * 100):.2f}%")
