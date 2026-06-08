from __future__ import annotations

import numpy as np
import pandas as pd


def build_default_hourly_profile(
    peak_power_mw: float,
    year_start: str = "2024-01-01 00:00:00",
    hours: int = 8760,
) -> pd.DataFrame:
    idx = pd.date_range(start=year_start, periods=hours, freq="H")

    hour_of_day = idx.hour.to_numpy()
    day_of_year = idx.dayofyear.to_numpy()

    daily_shape = np.maximum(0.0, np.sin((hour_of_day - 6.0) / 24.0 * 2.0 * np.pi))
    seasonal_shape = 0.55 + 0.35 * np.sin((day_of_year - 80.0) / 365.0 * 2.0 * np.pi)
    noise = 0.08 * (np.sin(np.arange(hours) * 0.17) + 1.0) / 2.0

    renewable_power_mw = peak_power_mw * np.clip(daily_shape * seasonal_shape + noise, 0.0, 1.0)

    return pd.DataFrame(
        {
            "timestamp": idx,
            "renewable_power_mw": renewable_power_mw,
        }
    )
