"""Configuration for skforecast adaptor."""

from typing import Any

from chapkit.config.schemas import BaseConfig
from pydantic import Field


class SkforecastConfig(BaseConfig):
    """Configuration for skforecast adaptor."""

    lags: int | list[int] = 12
    n_samples: int = 200
    use_bootstrapping: bool = True
    exogenous_variables: list[str] | None = None
    model_class: str = "sklearn.ensemble.GradientBoostingRegressor"
    model_params: dict[str, Any] = Field(
        default_factory=lambda: {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.1,
            "min_samples_leaf": 3,
            "random_state": 42,
        }
    )
    encoding: str = "onehot"
    differentiation: int | None = 1
    transformer_series: str | None = "StandardScaler"
    refit_on_predict: bool = True
    n_prediction_steps: int = 3
