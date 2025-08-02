"""Configuration classes for forecasting models."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Tuple


@dataclass
class CrossValidationConfig:
    """Configuration for cross-validation parameters."""

    min_train_weeks: int = 26
    eval_period_weeks: int = 6
    cv_folds: int = 4

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.min_train_weeks <= 0:
            raise ValueError("min_train_weeks must be positive")
        if self.eval_period_weeks <= 0:
            raise ValueError("eval_period_weeks must be positive")
        if self.cv_folds <= 0:
            raise ValueError("cv_folds must be positive")


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""

    min_train_weeks: int = 26
    eval_period_weeks: int = 6
    cv_folds: int = 4

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.min_train_weeks <= 0:
            raise ValueError("min_train_weeks must be positive")
        if self.eval_period_weeks <= 0:
            raise ValueError("eval_period_weeks must be positive")
        if self.cv_folds <= 0:
            raise ValueError("cv_folds must be positive")


@dataclass
class TrainingResult:  # pylint: disable=R0902
    """Container for comprehensive training results with both evaluation types."""

    # Required fields
    model_id: str
    model_name: str
    parameters: Dict[str, Any]
    cv_scores: List[float]
    avg_wape: float
    metrics_all_folds: List[Dict[str, float]]
    baseline_cv_scores: List[float]
    baseline_avg_wape: float
    baseline_metrics_all_folds: List[Dict[str, float]]
    train_periods: List[Tuple[str, str]]
    eval_periods: List[Tuple[str, str]]
    features_used: List[str]

    # Optional fields (with defaults)
    iterative_cv_scores: List[float] = field(default_factory=list)
    iterative_avg_wape: float = 0.0
    iterative_metrics_all_folds: List[Dict[str, float]] = field(default_factory=list)
    baseline_iterative_cv_scores: List[float] = field(default_factory=list)
    baseline_iterative_avg_wape: float = 0.0
    baseline_iterative_metrics_all_folds: List[Dict[str, float]] = field(
        default_factory=list
    )
    test_metrics: Dict[str, float] = field(default_factory=dict)
    baseline_test_metrics: Dict[str, float] = field(default_factory=dict)
    iterative_test_metrics: Dict[str, float] = field(default_factory=dict)
    baseline_iterative_test_metrics: Dict[str, float] = field(default_factory=dict)
    test_period_start: str = ""
    test_period_end: str = ""

    def __post_init__(self) -> None:
        """Validate training result data."""
        if not self.model_id:
            raise ValueError("model_id cannot be empty")
        if not self.model_name:
            raise ValueError("model_name cannot be empty")
        if len(self.cv_scores) != len(self.baseline_cv_scores):
            raise ValueError("CV scores length mismatch")

    def to_dict(self) -> Dict[str, Any]:
        """Convert TrainingResult to dictionary."""
        return asdict(self)
