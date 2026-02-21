"""Adaptor for using MultistepModel with chapkit."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chap_python_sdk.testing.types import PredictFunction, TrainFunction

try:
    from sklearn.base import BaseEstimator  # type: ignore[import-untyped]  # noqa: F401

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False  # pyright: ignore[reportConstantRedefinition]

from .cli_model import create_multistep_cli_app
from .config import MultistepConfig

__all__ = ["MultistepConfig", "create_multistep_cli_app", "create_multistep_functions", "SKLEARN_AVAILABLE"]


def create_multistep_functions(
    config: MultistepConfig | None = None,
) -> tuple["TrainFunction", "PredictFunction"]:
    """Create train/predict functions using MultistepModel.

    Args:
        config: Optional MultistepConfig for customization.

    Returns:
        Tuple of (train_function, predict_function) compatible with chapkit validation.

    Raises:
        ImportError: If scikit-learn is not installed.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is not installed. Install with: uv add chap-python-sdk[multistep]")

    from .adaptor import MultistepAdaptor

    if config is None:
        config = MultistepConfig()

    adaptor = MultistepAdaptor(config)
    return adaptor.train, adaptor.predict
