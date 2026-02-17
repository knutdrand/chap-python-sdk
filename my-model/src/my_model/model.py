"""Model implementation for my-model."""

from typing import Any

from chapkit.config.schemas import BaseConfig
from chapkit.data import DataFrame
from geojson_pydantic import FeatureCollection
from pydantic import Field

from chap_python_sdk.testing import RunInfo


class MyModelConfig(BaseConfig):
    """Configuration for my-model."""

    n_samples: int = Field(default=10, description="Number of prediction samples")


async def train(
    config: BaseConfig,
    data: DataFrame,
    run_info: RunInfo | None = None,
    geo: FeatureCollection | None = None,
) -> dict[str, float]:
    """Train the model.

    Args:
        config: Model configuration.
        data: Training data with time_period, location, and covariate columns.
        run_info: Runtime information including prediction_length.
        geo: Optional GeoJSON feature collection for geographic data.

    Returns:
        Trained model object (must be pickleable).
    """
    # Example: compute mean of disease_cases
    mean_value = sum(data["disease_cases"]) / len(data["disease_cases"])
    return {"mean": mean_value}


async def predict(
    config: BaseConfig,
    model: dict[str, float],
    historic: DataFrame,
    future: DataFrame,
    run_info: RunInfo | None = None,
    geo: FeatureCollection | None = None,
) -> DataFrame:
    """Generate predictions.

    Args:
        config: Model configuration.
        model: Trained model object from train().
        historic: Recent historical data.
        future: Future periods to predict (time_period, location columns).
        run_info: Runtime information.
        geo: Optional GeoJSON feature collection.

    Returns:
        DataFrame with time_period, location, and samples columns.
    """
    n_samples = config.n_samples if hasattr(config, "n_samples") else 10

    # Generate samples around the mean
    samples_list = [[model["mean"]] * n_samples for _ in range(len(future))]

    return DataFrame.from_dict({
        "time_period": list(future["time_period"]),
        "location": list(future["location"]),
        "samples": samples_list,
    })
