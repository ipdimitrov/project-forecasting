"""
Simplified preprocessor that delegates to the unified feature engine.
All feature engineering logic is now centralized in the unified forecaster.
"""

from typing import List
import pandas as pd

from ..core.unified_forecaster import UnifiedFeatureEngine
from ..core.feature_config import DEFAULT_FEATURE_CONFIG


class FastDataPreprocessor:
    """Simplified preprocessor using unified feature engine."""

    def __init__(self, feature_config=None):
        self.feature_engine = UnifiedFeatureEngine(
            feature_config or DEFAULT_FEATURE_CONFIG
        )
        self.feature_columns: List[str] = []
        self.is_fitted: bool = False

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform using unified engine."""
        result = self.feature_engine.fit_transform(data)
        self.feature_columns = self.feature_engine.feature_columns
        self.is_fitted = True
        return result

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform using unified engine."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        return self.feature_engine.transform(data)
