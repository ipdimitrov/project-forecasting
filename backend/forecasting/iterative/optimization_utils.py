"""
Optimization utilities for faster iterative forecasting.
"""

import numpy as np
import pandas as pd
from typing import Dict
import hashlib


class FeatureCache:
    """Cache for computed features to avoid redundant calculations."""

    def __init__(self, max_cache_size: int = 1000):
        self.cache = {}
        self.max_cache_size = max_cache_size
        self.cache_hits = 0
        self.cache_misses = 0

    def _get_cache_key(self, data_row: pd.DataFrame) -> str:
        """Generate cache key from data row."""
        # Use timestamp and value to create cache key
        timestamp = data_row.index[0]
        value = data_row.iloc[0, 0] if len(data_row.columns) > 0 else 0

        # Create hash from timestamp and value
        key_string = f"{timestamp}_{value:.6f}"
        return hashlib.md5(key_string.encode()).hexdigest()[:16]

    def can_use_cache(self, data_row: pd.DataFrame) -> bool:
        """Check if cached features are available."""
        cache_key = self._get_cache_key(data_row)
        return cache_key in self.cache

    def get_cached_features(self, data_row: pd.DataFrame) -> pd.DataFrame:
        """Retrieve cached features."""
        cache_key = self._get_cache_key(data_row)
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key].copy()
        else:
            self.cache_misses += 1
            return None

    def cache_features(self, data_row: pd.DataFrame, features: pd.DataFrame):
        """Cache computed features."""
        cache_key = self._get_cache_key(data_row)

        # Manage cache size
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[cache_key] = features.copy()

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
        }


class BatchProcessor:
    """Batch processing for multiple time steps to reduce overhead."""

    def __init__(self, batch_size: int = 50):
        self.batch_size = batch_size
        self.feature_buffer = []
        self.prediction_buffer = []

    def can_process_batch(self, current_step: int) -> bool:
        """Check if we can process a batch."""
        return (current_step + 1) % self.batch_size == 0

    def add_to_batch(self, features: pd.DataFrame):
        """Add features to current batch."""
        self.feature_buffer.append(features)

    def process_batch(self, model) -> np.ndarray:
        """Process accumulated batch."""
        if not self.feature_buffer:
            return np.array([])

        # Combine all features in batch
        batch_features = pd.concat(self.feature_buffer, ignore_index=True)

        # Batch prediction
        batch_predictions = model.predict(batch_features)

        # Clear buffer
        self.feature_buffer = []

        return batch_predictions

    def get_pending_features(self) -> int:
        """Get number of pending features in buffer."""
        return len(self.feature_buffer)
