"""Main adaptor for using MultistepModel with chapkit."""

import asyncio
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
from chapkit.config.schemas import BaseConfig
from chapkit.data import DataFrame
from sklearn.preprocessing import FunctionTransformer  # type: ignore[import-untyped]

from chap_python_sdk.testing.types import GeoFeatureCollection, RunInfo

from .config import MultistepConfig
from .data_transformer import xarray_predictions_to_chapkit
from .model import DataFrameMultistepModel
from .one_step_model import ResidualBootstrapModel
from .pipeline import build_feature_transformer, build_target_pipeline


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
        df = data.to_pandas()
        index_cols = ["time_period", "location"]
        feature_cols = list(self.config.exogenous_variables) if self.config.exogenous_variables else []

        y: pd.DataFrame = df[index_cols + [self.config.target_variable]]  # pyright: ignore[reportAssignmentType]

        feature_transformer = build_feature_transformer(feature_cols, self.config)
        scaled = feature_transformer.fit_transform(df[feature_cols])  # pyright: ignore[reportAssignmentType]
        x_features = pd.concat(
            [df[index_cols].reset_index(drop=True), pd.DataFrame(scaled).reset_index(drop=True)], axis=1
        )

        target_pipeline = build_target_pipeline(self.config)
        one_step = ResidualBootstrapModel(self.config.model_class, self.config.model_params)
        model = DataFrameMultistepModel(
            one_step,
            self.config.n_target_lags,
            target_pipeline=target_pipeline,
            target_variable=self.config.target_variable,
        )
        model.fit(x_features, y)

        return {
            "model": model,
            "feature_transformer": feature_transformer,
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
        df_model: DataFrameMultistepModel = model["model"]
        feature_transformer = model.get("feature_transformer") or FunctionTransformer()

        historic_pd = historic.to_pandas()
        future_pd = future.to_pandas()

        index_cols = ["time_period", "location"]
        y_historic: pd.DataFrame = historic_pd[index_cols + [restored_config.target_variable]]  # pyright: ignore[reportAssignmentType]

        feature_cols = list(restored_config.exogenous_variables) if restored_config.exogenous_variables else []
        scaled = feature_transformer.transform(future_pd[feature_cols])  # pyright: ignore[reportAssignmentType]
        x_future = pd.concat(
            [future_pd[index_cols].reset_index(drop=True), pd.DataFrame(scaled).reset_index(drop=True)], axis=1
        )

        n_steps = future_pd.groupby("location").size().iloc[0]
        predictions = df_model.predict(y_historic, x_future, n_steps, restored_config.n_samples)

        return xarray_predictions_to_chapkit(predictions, future)
