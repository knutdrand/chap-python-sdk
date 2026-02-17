"""Main adaptor for using MultistepModel with chapkit."""

import asyncio
from typing import Any

from chapkit.config.schemas import BaseConfig
from chapkit.data import DataFrame

from chap_python_sdk.testing.types import GeoFeatureCollection, RunInfo

from .config import MultistepConfig
from .data_transformer import chapkit_future_to_xarray, chapkit_to_xarray, xarray_predictions_to_chapkit
from .one_step_model import ResidualBootstrapModel


class MultistepAdaptor:
    """Adaptor for using MultistepModel with chapkit."""

    def __init__(self, config: MultistepConfig):
        """Initialize the adaptor with configuration."""
        self.config = config

    async def train(
        self,
        config: BaseConfig,
        data: DataFrame,
        run_info: RunInfo,
        geo: GeoFeatureCollection | None = None,
    ) -> dict[str, Any]:
        """Train multistep model (async wrapper)."""
        return await asyncio.to_thread(self._train_sync, config, data, run_info, geo)

    def _train_sync(
        self,
        config: BaseConfig,
        data: DataFrame,
        run_info: RunInfo,
        geo: GeoFeatureCollection | None,
    ) -> dict[str, Any]:
        """Synchronous training logic."""
        from chap_python_sdk.adaptors.multistep_model import MultistepModel

        y, X = chapkit_to_xarray(
            data,
            target_variable=self.config.target_variable,
            exogenous_variables=self.config.exogenous_variables,
        )

        one_step = ResidualBootstrapModel(self.config.model_class, self.config.model_params)
        model = MultistepModel(one_step, n_target_lags=self.config.n_target_lags)
        model.fit_multi(y, X)

        return {
            "multistep_model": model,
            "locations": y.coords["location"].values.tolist(),
            "config": self.config.model_dump(),
        }

    async def predict(
        self,
        config: BaseConfig,
        model: dict[str, Any],
        historic: DataFrame,
        future: DataFrame,
        run_info: RunInfo,
        geo: GeoFeatureCollection | None = None,
    ) -> DataFrame:
        """Generate predictions (async wrapper)."""
        return await asyncio.to_thread(self._predict_sync, config, model, historic, future, run_info, geo)

    def _predict_sync(
        self,
        config: BaseConfig,
        model: dict[str, Any],
        historic: DataFrame,
        future: DataFrame,
        run_info: RunInfo,
        geo: GeoFeatureCollection | None,
    ) -> DataFrame:
        """Synchronous prediction logic."""
        restored_config = MultistepConfig(**model["config"])
        multistep_model = model["multistep_model"]

        # Get historic target for lag window
        y_historic, _ = chapkit_to_xarray(
            historic,
            target_variable=restored_config.target_variable,
            exogenous_variables=restored_config.exogenous_variables,
        )
        previous_y = y_historic.isel(time=slice(-restored_config.n_target_lags, None))

        # Get future exogenous features and determine steps
        _, time_periods, X_future = chapkit_future_to_xarray(
            future,
            exogenous_variables=restored_config.exogenous_variables,
        )
        n_steps = len(time_periods)

        # Generate predictions
        predictions = multistep_model.predict_multi(
            previous_y,
            n_steps=n_steps,
            n_samples=restored_config.n_samples,
            X=X_future,
        )

        return xarray_predictions_to_chapkit(predictions, future)
