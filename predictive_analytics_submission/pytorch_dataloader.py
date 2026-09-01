"""
PyTorch Dataset and DataLoader Wrapper for Multi-Horizon Deep Learning Forecasters.
Enables sequence windowing for Temporal Convolutional Networks (TCN), PatchTST, and LSTMs.
"""

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    _BaseDataset = Dataset
    HAS_TORCH = True
except ImportError:
    _BaseDataset = object
    HAS_TORCH = False


class SensorTelemetryDataset(_BaseDataset):
    """PyTorch Dataset wrapper for multi-horizon sequence windowing."""
    def __init__(self, values: np.ndarray, lookback_len: int = 180, horizon_len: int = 180):
        if HAS_TORCH:
            self.values = torch.tensor(values, dtype=torch.float32)
        else:
            self.values = np.array(values, dtype=np.float32)
        self.lookback_len = lookback_len
        self.horizon_len = horizon_len
        self.total_len = len(values) - lookback_len - horizon_len + 1

    def __len__(self) -> int:
        return max(0, self.total_len)

    def __getitem__(self, idx: int):
        if HAS_TORCH:
            x = self.values[idx : idx + self.lookback_len].unsqueeze(-1)
            y = self.values[idx + self.lookback_len : idx + self.lookback_len + self.horizon_len]
        else:
            x = self.values[idx : idx + self.lookback_len, np.newaxis]
            y = self.values[idx + self.lookback_len : idx + self.lookback_len + self.horizon_len]
        return x, y


def create_pytorch_dataloaders(
    clean_series: np.ndarray,
    lookback_len: int = 180,
    horizon_len: int = 180,
    batch_size: int = 64,
    val_split_steps: int = 180
):
    """Creates training and validation PyTorch DataLoaders."""
    if not HAS_TORCH:
        raise ImportError("PyTorch (torch) is required to instantiate DataLoaders.")
        
    train_vals = clean_series[:-val_split_steps]
    val_vals = clean_series[-(lookback_len + val_split_steps):]
    train_dataset = SensorTelemetryDataset(train_vals, lookback_len, horizon_len)
    val_dataset = SensorTelemetryDataset(val_vals, lookback_len, horizon_len)
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    )


if __name__ == "__main__":
    dummy_data = np.sin(np.linspace(0, 100, 1000))
    ds = SensorTelemetryDataset(dummy_data, lookback_len=180, horizon_len=180)
    print(f"Dataset length: {len(ds)}")
    x, y = ds[0]
    print(f"Item 0 shape: x={x.shape}, y={y.shape}")
