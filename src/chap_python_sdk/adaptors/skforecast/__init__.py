"""Adaptor for using skforecast models with chapkit."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chap_python_sdk.testing.types import PredictFunction, TrainFunction

try:
    from skforecast.recursive import ForecasterRecursiveMultiSeries  # type: ignore[import-untyped]

    SKFORECAST_AVAILABLE = True
except ImportError:
    SKFORECAST_AVAILABLE = False
    ForecasterRecursiveMultiSeries = None

from .config import SkforecastConfig

__all__ = ["SkforecastConfig", "create_skforecast_functions", "SKFORECAST_AVAILABLE"]


def create_skforecast_functions(
    config: SkforecastConfig | None = None,
) -> tuple["TrainFunction", "PredictFunction"]:
    """Create train/predict functions using skforecast.

    Args:
        config: Optional SkforecastConfig for customization

    Returns:
        Tuple of (train_function, predict_function) compatible with chapkit validation

    Raises:
        ImportError: If skforecast is not installed
    """
    if not SKFORECAST_AVAILABLE:
        raise ImportError("skforecast is not installed. Install with: uv add chap-python-sdk[skforecast]")

    from .adaptor import SkforecastAdaptor

    if config is None:
        config = SkforecastConfig()

    adaptor = SkforecastAdaptor(config)
    return adaptor.train, adaptor.predict  # type: ignore[return-value]
