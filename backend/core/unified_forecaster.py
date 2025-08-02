"""
Unified, optimized forecasting engine.
Consolidates all forecasting logic into one fast implementation.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)

from .feature_config import FeatureConfig, DEFAULT_FEATURE_CONFIG  # noqa: E402


@dataclass
class ForecastingConfig:
    """Configuration for forecasting performance and behavior."""

    batch_size: int = 200
    progress_interval: int = 2000
    use_float32: bool = True


class UnifiedFeatureEngine:
    """Ultra-fast unified feature engineering engine."""

    def __init__(self, feature_config: Optional[FeatureConfig] = None) -> None:
        self.config = feature_config or DEFAULT_FEATURE_CONFIG
        self.feature_columns: List[str] = []
        self.is_fitted: bool = False

        # Initialize attributes to avoid pylint warnings
        self._lag_periods: Dict[str, int] = {}
        self._rolling_configs: Dict[str, Any] = {}
        self._static_features: List[str] = []
        self._feature_order: Optional[List[str]] = None

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform data in one optimized pass."""
        self._validate_input(data)
        self._initialize_metadata()

        features = self._create_all_features_vectorized(data)

        self.feature_columns = list(features.columns)
        self._feature_order = self.feature_columns.copy()
        self.is_fitted = True

        return features

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted configuration."""
        if not self.is_fitted:
            raise ValueError("FeatureEngine must be fitted before transform")

        features = self._create_all_features_vectorized(data)
        return features[self.feature_columns].copy()

    def _initialize_metadata(self) -> None:
        """Initialize pre-computed metadata for optimization."""
        self._lag_periods = self.config.get_all_lag_periods()
        self._rolling_configs = self.config.get_rolling_configs()
        self._static_features = self.config.get_static_features()

    def _validate_input(self, data: pd.DataFrame) -> None:
        """Validate input data."""
        if "value" not in data.columns:
            raise ValueError("Data must contain 'value' column")
        if len(data) == 0:
            raise ValueError("Cannot process empty dataset")

    def _create_all_features_vectorized(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create all features in one optimized vectorized pass."""
        features = pd.DataFrame(index=data.index)
        values = data["value"]

        # Static features (vectorized)
        if self.config.include_hour:
            features["hour"] = data.index.hour
        if self.config.include_day_of_week:
            features["day_of_week"] = data.index.day_of_week
        if self.config.include_month:
            features["month"] = data.index.month
        if self.config.include_weekend:
            features["weekend"] = (data.index.day_of_week >= 5).astype(int)
        if self.config.include_week_in_month:
            features["week_in_month"] = (data.index.day - 1) // 7 + 1

        # Cyclic time features
        if self.config.include_cyclic_time:
            hours = data.index.hour
            features["hour_sin"] = np.sin(2 * np.pi * hours / 24)
            features["hour_cos"] = np.cos(2 * np.pi * hours / 24)

        # Lag features (vectorized)
        for feature_name, lag_periods in self._lag_periods.items():
            features[feature_name] = values.shift(lag_periods)

        # Rolling features (vectorized)
        for feature_name, config in self._rolling_configs.items():
            if config["function"] == "mean":
                features[feature_name] = values.rolling(
                    window=config["window"], min_periods=config["min_periods"]
                ).mean()
            elif config["function"] == "std":
                features[feature_name] = values.rolling(
                    window=config["window"], min_periods=config["min_periods"]
                ).std()

        return features.ffill().bfill().fillna(0)

    def get_feature_metadata(self) -> Dict[str, Any]:
        """Get metadata about features for optimization."""
        return {
            "lag_periods": self._lag_periods,
            "rolling_configs": self._rolling_configs,
            "static_features": self._static_features,
            "feature_order": self._feature_order,
            "total_features": len(self.feature_columns) if self.is_fitted else 0,
        }


class UnifiedIterativeForecaster:
    """Ultra-fast unified iterative forecasting engine."""

    def __init__(
        self,
        model: Any,
        feature_engine: UnifiedFeatureEngine,
        config: Optional[ForecastingConfig] = None,
    ) -> None:
        self.model = model
        self.feature_engine = feature_engine
        self.config = config or ForecastingConfig()

        # Initialize optimization attributes
        self._optimization_ready: bool = False
        self._metadata: Optional[Dict[str, Any]] = None
        self._feature_positions: Optional[Dict[str, int]] = None
        self._n_static: int = 0
        self._n_lag: int = 0
        self._n_rolling: int = 0
        self._n_total: int = 0
        self._lag_periods: Dict[str, int] = {}
        self._rolling_configs: Dict[str, Any] = {}
        self._static_features: List[str] = []
        self._feature_order: Optional[List[str]] = None

    def _setup_optimization_structures(self) -> None:
        """Set up optimization structures after feature engine is fitted."""
        if self._optimization_ready:
            return

        if not self.feature_engine.is_fitted:
            raise ValueError(
                "Feature engine must be fitted before setting up optimization structure"
            )

        self._metadata = self.feature_engine.get_feature_metadata()
        self._lag_periods = self._metadata["lag_periods"]
        self._rolling_configs = self._metadata["rolling_configs"]
        self._static_features = self._metadata["static_features"]
        self._feature_order = self._metadata["feature_order"]

        if self._feature_order:
            self._feature_positions = {
                name: idx for idx, name in enumerate(self._feature_order)
            }

        self._n_static = len(self._static_features)
        self._n_lag = len(self._lag_periods)
        self._n_rolling = len(self._rolling_configs)
        self._n_total = len(self._feature_order) if self._feature_order else 0

        self._optimization_ready = True

    def forecast(
        self, historical_data: pd.DataFrame, forecast_steps: int
    ) -> Dict[str, Any]:
        """Ultra-fast iterative forecasting."""
        if not self.feature_engine.is_fitted:
            self.feature_engine.fit_transform(historical_data)

        self._setup_optimization_structures()

        start_time = datetime.now()

        dtype = np.float32 if self.config.use_float32 else np.float64
        predictions = np.zeros(forecast_steps, dtype=dtype)
        timestamps = self._generate_timestamps(historical_data, forecast_steps)

        static_features_matrix = self._precompute_static_features(timestamps, dtype)

        lag_buffer = self._initialize_lag_buffer(historical_data, dtype)
        rolling_buffers = self._initialize_rolling_buffers(historical_data, dtype)

        batch_size = min(self.config.batch_size, forecast_steps)
        n_batches = (forecast_steps + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, forecast_steps)
            batch_size_actual = batch_end - batch_start

            batch_features = self._prepare_batch_features_optimized(
                batch_start,
                batch_size_actual,
                static_features_matrix,
                lag_buffer,
                rolling_buffers,
                predictions,
            )

            if hasattr(self.model, "predict_processed"):
                batch_predictions = self.model.predict_processed(batch_features)
            else:
                if hasattr(self.model, "model") and hasattr(
                    self.model.model, "predict"
                ):
                    batch_predictions = self.model.model.predict(batch_features)
                else:
                    raise ValueError(
                        "Model does not support direct prediction with processed features."
                    )

            predictions[batch_start:batch_end] = batch_predictions.astype(dtype)

            for i in range(batch_size_actual):
                step_idx = batch_start + i
                self._update_buffers_fast(
                    lag_buffer, rolling_buffers, predictions[step_idx]
                )

            if (
                batch_end % self.config.progress_interval == 0
                or batch_end == forecast_steps
            ):
                self._report_progress(batch_end, forecast_steps, start_time)

        total_time = (datetime.now() - start_time).total_seconds()

        return {
            "predictions": predictions,
            "timestamps": timestamps,
            "total_time_seconds": total_time,
            "predictions_per_second": (
                forecast_steps / total_time if total_time > 0 else 0
            ),
        }

    def _generate_timestamps(
        self, historical_data: pd.DataFrame, steps: int
    ) -> List[pd.Timestamp]:
        """Generate forecast timestamps efficiently."""
        last_ts = historical_data.index[-1]
        return [last_ts + pd.Timedelta(hours=i + 1) for i in range(steps)]

    def _precompute_static_features(
        self, timestamps: List[pd.Timestamp], dtype: np.dtype
    ) -> np.ndarray:
        """Pre-compute all static features using vectorization."""
        n_steps = len(timestamps)
        static_matrix = np.zeros((n_steps, self._n_static), dtype=dtype)

        if self._n_static == 0:
            return static_matrix

        ts_array = pd.to_datetime(timestamps)

        col_idx = 0
        for feature_name in self._static_features:
            if feature_name == "hour":
                static_matrix[:, col_idx] = ts_array.hour.values
            elif feature_name == "day_of_week":
                static_matrix[:, col_idx] = ts_array.day_of_week.values
            elif feature_name == "month":
                static_matrix[:, col_idx] = ts_array.month.values
            elif feature_name == "weekend":
                static_matrix[:, col_idx] = (ts_array.day_of_week >= 5).astype(dtype)
            elif feature_name == "week_in_month":
                static_matrix[:, col_idx] = ((ts_array.day - 1) // 7 + 1).astype(dtype)
            elif feature_name == "hour_sin":
                static_matrix[:, col_idx] = np.sin(
                    2 * np.pi * ts_array.hour.values / 24
                )
            elif feature_name == "hour_cos":
                static_matrix[:, col_idx] = np.cos(
                    2 * np.pi * ts_array.hour.values / 24
                )

            col_idx += 1

        return static_matrix

    def _initialize_lag_buffer(
        self, historical_data: pd.DataFrame, dtype: np.dtype
    ) -> Dict[str, Any]:
        """Initialize optimized lag buffer."""
        max_lag = max(self._lag_periods.values()) if self._lag_periods else 0

        if max_lag == 0:
            return {"values": np.array([], dtype=dtype), "position": 0, "size": 0}

        buffer_size = max_lag + 50
        values = historical_data["value"].values[-buffer_size:].astype(dtype)

        if len(values) < buffer_size:
            padding = np.full(buffer_size - len(values), values[0], dtype=dtype)
            values = np.concatenate([padding, values])

        return {"values": values, "position": 0, "size": buffer_size}

    def _initialize_rolling_buffers(
        self, historical_data: pd.DataFrame, dtype: np.dtype
    ) -> Dict[str, Any]:
        """Initialize optimized rolling statistics buffers."""
        buffers = {}
        historical_values = historical_data["value"].values.astype(dtype)

        for feature_name, config in self._rolling_configs.items():
            window = config["window"]

            buffer_values = (
                historical_values[-window * 2 :]  # noqa: E203
                if len(historical_values) >= window * 2
                else historical_values
            )

            if len(buffer_values) < window:
                padding = np.full(
                    window - len(buffer_values),
                    buffer_values[0] if len(buffer_values) > 0 else 0,
                    dtype=dtype,
                )
                buffer_values = np.concatenate([padding, buffer_values])

            recent_window = buffer_values[-window:]

            buffers[feature_name] = {
                "values": buffer_values.astype(dtype),
                "window": window,
                "position": 0,
                "current_stat": (
                    np.mean(recent_window)
                    if config["function"] == "mean"
                    else np.std(recent_window)
                ),
            }

        return buffers

    def _prepare_batch_features_optimized(
        self,
        batch_start: int,
        batch_size: int,
        static_features: np.ndarray,
        lag_buffer: Dict[str, Any],
        rolling_buffers: Dict[str, Any],
        predictions: np.ndarray,
    ) -> np.ndarray:
        """Prepare feature matrix for batch prediction with maximum optimization."""
        dtype = np.float32 if self.config.use_float32 else np.float64
        batch_features = np.zeros((batch_size, self._n_total), dtype=dtype)

        if not self._feature_positions:
            return batch_features

        for i in range(batch_size):
            step_idx = batch_start + i

            static_col = 0
            for feature_name in self._static_features:
                pos = self._feature_positions[feature_name]
                batch_features[i, pos] = static_features[step_idx, static_col]
                static_col += 1

            for feature_name, lag_periods in self._lag_periods.items():
                pos = self._feature_positions[feature_name]

                if step_idx >= lag_periods:
                    batch_features[i, pos] = predictions[step_idx - lag_periods]
                else:
                    if lag_buffer["size"] > 0:
                        buffer_idx = (
                            lag_buffer["position"] - lag_periods + step_idx
                        ) % lag_buffer["size"]
                        batch_features[i, pos] = lag_buffer["values"][buffer_idx]

            for feature_name, buffer_data in rolling_buffers.items():
                pos = self._feature_positions[feature_name]
                batch_features[i, pos] = buffer_data["current_stat"]

        return batch_features

    def _update_buffers_fast(
        self,
        lag_buffer: Dict[str, Any],
        rolling_buffers: Dict[str, Any],
        new_value: float,
    ) -> None:
        """Update all buffers incrementally for O(1) performance."""
        if lag_buffer["size"] > 0:
            lag_buffer["values"][lag_buffer["position"]] = new_value
            lag_buffer["position"] = (lag_buffer["position"] + 1) % lag_buffer["size"]

        for feature_name, buffer_data in rolling_buffers.items():
            window = buffer_data["window"]

            buffer_data["values"] = np.roll(buffer_data["values"], -1)
            buffer_data["values"][-1] = new_value

            window_data = buffer_data["values"][-window:]
            config = self._rolling_configs[feature_name]

            if config["function"] == "mean":
                buffer_data["current_stat"] = np.mean(window_data)
            elif config["function"] == "std":
                buffer_data["current_stat"] = np.std(window_data)

    def _report_progress(self, current: int, total: int, start_time: datetime) -> None:
        """Report progress efficiently."""
        elapsed = (datetime.now() - start_time).total_seconds()
        rate = current / elapsed if elapsed > 0 else 0
        eta = (total - current) / rate if rate > 0 else 0
        logger.info(
            "Progress: %d/%d | %.1f pred/sec | ETA: %.0fs", current, total, rate, eta
        )


def create_unified_forecaster(
    model: Any, feature_config: Optional[FeatureConfig] = None
) -> UnifiedIterativeForecaster:
    """Factory function to create a unified forecaster with optimized configuration."""
    feature_engine = UnifiedFeatureEngine(feature_config)
    return UnifiedIterativeForecaster(model, feature_engine)
