"""
Centralized feature configuration for all forecasting components.
Single source of truth for all feature parameters.
"""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class FeatureConfig:  # pylint: disable=R0902
    """Centralized configuration for all feature engineering parameters."""

    # Lag features (in hours)
    lag_hours: List[int] = field(default_factory=list)
    lag_days: List[int] = field(default_factory=list)  # Will be converted to hours
    lag_weeks: List[int] = field(default_factory=list)  # Will be converted to hours

    # Rolling window features (in hours)
    rolling_windows: List[int] = field(default_factory=list)  # 1 week, 5 weeks
    rolling_min_periods_ratio: float = 0.5  # min_periods = window * ratio

    # Static/time features to include
    include_hour: bool = True
    include_day_of_week: bool = True
    include_month: bool = True
    include_weekend: bool = True
    include_week_in_month: bool = True
    include_cyclic_time: bool = True  # sin/cos transforms

    def get_all_lag_periods(self) -> Dict[str, int]:
        """Get all lag periods in hours with their feature names."""
        lag_periods = {}

        # Hour lags
        for lag in self.lag_hours:
            lag_periods[f"lag_{lag}h"] = lag

        # Daily lags (convert to hours)
        for lag_days in self.lag_days:
            lag_periods[f"lag_{lag_days}d"] = lag_days * 24

        # Weekly lags (convert to hours)
        for lag_weeks in self.lag_weeks:
            lag_periods[f"lag_{lag_weeks}w"] = lag_weeks * 7 * 24

        return lag_periods

    def get_rolling_configs(self) -> Dict[str, Dict]:
        """Get rolling window configurations."""
        configs = {}

        for window in self.rolling_windows:
            window_name = f"{window}h"
            min_periods = max(1, int(window * self.rolling_min_periods_ratio))

            configs[f"mean_{window_name}"] = {
                "window": window,
                "min_periods": min_periods,
                "function": "mean",
            }
            configs[f"std_{window_name}"] = {
                "window": window,
                "min_periods": min_periods,
                "function": "std",
            }

        return configs

    def get_static_features(self) -> List[str]:
        """Get list of static feature names to include."""
        features = []

        if self.include_hour:
            features.append("hour")
        if self.include_day_of_week:
            features.append("day_of_week")
        if self.include_month:
            features.append("month")
        if self.include_weekend:
            features.append("weekend")
        if self.include_week_in_month:
            features.append("week_in_month")
        if self.include_cyclic_time:
            features.extend(["hour_sin", "hour_cos"])

        return features


# Global default configuration - can be modified in one place
DEFAULT_FEATURE_CONFIG = FeatureConfig()
