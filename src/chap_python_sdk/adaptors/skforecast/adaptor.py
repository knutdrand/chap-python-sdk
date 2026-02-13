"""Main adaptor for using skforecast models with chapkit."""

import asyncio
from typing import Any

from chapkit.config.schemas import BaseConfig
from chapkit.data import DataFrame

from chap_python_sdk.testing.types import GeoFeatureCollection, RunInfo

from .config import SkforecastConfig
from .data_transformer import chapkit_to_wide, exog_to_wide, wide_to_chapkit
from .forecaster import SkforecastWrapper


class SkforecastAdaptor:
    """Adaptor for using skforecast models with chapkit."""

    def __init__(self, config: SkforecastConfig):
        """Initialize the adaptor with configuration."""
        self.config = config
        self.forecaster_wrapper: SkforecastWrapper | None = None

    async def train(
        self,
        config: BaseConfig,
        data: DataFrame,
        run_info: RunInfo,
        geo: GeoFeatureCollection | None = None,
    ) -> dict[str, Any]:
        """Train skforecast model (async wrapper)."""
        return await asyncio.to_thread(self._train_sync, config, data, run_info, geo)

    def _train_sync(
        self,
        config: BaseConfig,
        data: DataFrame,
        run_info: RunInfo,
        geo: GeoFeatureCollection | None,
    ) -> dict[str, Any]:
        """Synchronous training logic."""
        # Transform to wide format
        target_wide, exog_wide = chapkit_to_wide(
            data,
            target_variable="disease_cases",
            exogenous_variables=self.config.exogenous_variables,
        )

        # Fit forecaster
        self.forecaster_wrapper = SkforecastWrapper(self.config)
        self.forecaster_wrapper.fit(target_wide, exog_wide)

        # Return trained model dict (must be pickleable)
        return {
            "forecaster": self.forecaster_wrapper.forecaster,
            "residuals": self.forecaster_wrapper.residuals_by_step,
            "locations": list(target_wide.columns),
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
        restored_config = SkforecastConfig(**model["config"])
        forecaster_wrapper = SkforecastWrapper(restored_config)
        forecaster_wrapper.forecaster = model["forecaster"]
        forecaster_wrapper.residuals_by_step = model["residuals"]

        # Refit on historic data if configured and target column is available
        historic_pd = historic.to_pandas()
        if restored_config.refit_on_predict and "disease_cases" in historic_pd.columns:
            target_wide, exog_wide = chapkit_to_wide(
                historic,
                target_variable="disease_cases",
                exogenous_variables=restored_config.exogenous_variables,
            )
            forecaster_wrapper.refit(target_wide, exog_wide)

        # Prepare exogenous data from future if needed
        exog_future = None
        if self.config.exogenous_variables:
            exog_future = exog_to_wide(future, self.config.exogenous_variables)

        # Determine number of steps from future DataFrame
        future_pd = future.to_pandas()
        n_steps = len(future_pd["time_period"].unique())

        # Generate probabilistic samples
        samples_by_location = forecaster_wrapper.predict_samples(
            steps=n_steps,
            exog_future=exog_future,
            n_samples=self.config.n_samples,
        )

        # Transform back to chapkit long format
        result = wide_to_chapkit(samples_by_location, future)

        return result
