"""Configuration for skforecast adaptor."""

from typing import Any

from chapkit.config.schemas import BaseConfig
from pydantic import Field


class SkforecastConfig(BaseConfig):
    """Configuration for skforecast adaptor."""

    lags: int | list[int] = 12
    n_samples: int = 100
    use_bootstrapping: bool = True
    exogenous_variables: list[str] | None = None
    model_class: str = "sklearn.ensemble.RandomForestRegressor"
    model_params: dict[str, Any] = Field(default_factory=dict)
    encoding: str = "onehot"
