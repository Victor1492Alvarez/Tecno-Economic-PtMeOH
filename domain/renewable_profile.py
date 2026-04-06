from __future__ import annotations
import numpy as np
import pandas as pd

def build_default_hourly_profile(year: int = 2025, peak_power_mw: float = 140.0) -> pd.DataFrame:
    idx = pd.date_range(f"{year}-01-01", periods=8760, freq="h")
    day = idx.dayofyear.to_numpy()
    hour = idx.hour.to_numpy()
    seasonal = 0.65 + 0.24 * np.sin(2 * np.pi * (day - 90) / 365)
    diurnal = np.clip(np.sin(np.pi * (hour - 6) / 14), 0.0, None)
    weather = 0.92 + 0.10 * np.sin(2 * np.pi * day / 11) + 0.05 * np.cos(2 * np.pi * day / 29)
    power = peak_power_mw * np.clip(seasonal * diurnal * weather, 0.0, 1.05)
    return pd.DataFrame({"timestamp": idx, "renewable_power_mw": power})

