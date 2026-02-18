"""Configuration for multistep adaptor."""

from typing import Any

from chapkit.config.schemas import BaseConfig
from pydantic import Field


class MultistepConfig(BaseConfig):
    """Configuration for multistep recursive trajectory sampler."""

    n_target_lags: int = 12
    n_samples: int = 200
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
    exogenous_variables: list[str] | None = None
    target_variable: str = "disease_cases"
    log_transform_target: bool = False
    standardize_target: bool = False
    standardize_covariates: bool = False
