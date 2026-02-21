"""DataFrame-level wrapper for MultistepModel with target transforms and xarray conversion."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]
import xarray as xr
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import FunctionTransformer  # type: ignore[import-untyped]

from chap_python_sdk.adaptors.multistep_model import MultistepModel, OneStepModel


class DataFrameMultistepModel:
    """Wraps MultistepModel to accept DataFrames and handle target transforms internally."""

    def __init__(
        self,
        one_step_model: OneStepModel,
        n_target_lags: int,
        target_pipeline: Pipeline | None = None,
        target_variable: str = "disease_cases",
    ) -> None:
        """Initialize with a one-step model, lag count, optional target transforms."""
        self._model = MultistepModel(one_step_model, n_target_lags)
        self._target_pipeline = target_pipeline or Pipeline([("identity", FunctionTransformer())])
        self._target_variable = target_variable

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """Fit on feature and target DataFrames (multi-location).

        X: [time_period, location, feat1, feat2, ...]
        y: [time_period, location, <target_variable>]
        """
        y_values = y[self._target_variable].to_numpy().reshape(-1, 1).astype(np.float64)
        y_values = self._target_pipeline.fit_transform(y_values)

        y_transformed = y.copy()
        y_transformed[self._target_variable] = y_values.ravel()

        y_xr = self._target_to_xarray(y_transformed)
        X_xr = self._features_to_xarray(X)
        self._model.fit_multi(y_xr, X_xr)

    def predict_xarray(
        self,
        y_historic: pd.DataFrame,
        X_future: pd.DataFrame | None,
        n_steps: int,
        n_samples: int,
    ) -> xr.DataArray:
        """Predict from DataFrames. Returns (location, trajectory, step) in original scale.

        y_historic: [time_period, location, <target_variable>]
        X_future: [time_period, location, feat1, ...] or None
        """
        y_values = y_historic[self._target_variable].to_numpy().reshape(-1, 1).astype(np.float64)
        y_values = self._target_pipeline.transform(y_values)

        y_transformed = y_historic.copy()
        y_transformed[self._target_variable] = y_values.ravel()

        y_xr = self._target_to_xarray(y_transformed)
        previous_y = y_xr.isel(time=slice(-self._model.n_target_lags, None))

        X_future_xr = self._future_features_to_xarray(X_future) if X_future is not None else None

        predictions = self._model.predict_multi(
            previous_y,
            n_steps=n_steps,
            n_samples=n_samples,
            X=X_future_xr,
        )

        values = predictions.values
        original_shape = values.shape
        flat: npt.NDArray[np.floating[Any]] = self._target_pipeline.inverse_transform(values.reshape(-1, 1)).ravel()
        values = flat.reshape(original_shape)
        return xr.DataArray(values, dims=predictions.dims, coords=predictions.coords)

    def predict(
        self,
        y_historic: pd.DataFrame,
        X_future: pd.DataFrame | None,
        n_steps: int,
        n_samples: int,
    ) -> pd.DataFrame:
        """Predict and return a wide-format DataFrame.

        Returns a DataFrame with columns: location, time_step, sample_0, sample_1, ...
        One row per (location, time_step), one column per sample trajectory.
        """
        predictions = self.predict_xarray(y_historic, X_future, n_steps, n_samples)
        locations = predictions.coords["location"].values
        records: list[dict[str, object]] = []
        for loc_idx, loc in enumerate(locations):
            for step_idx in range(predictions.sizes["step"]):
                row: dict[str, object] = {"location": loc, "time_step": step_idx}
                for traj_idx in range(predictions.sizes["trajectory"]):
                    row[f"sample_{traj_idx}"] = float(predictions.values[loc_idx, traj_idx, step_idx])
                records.append(row)
        return pd.DataFrame(records)

    def _target_to_xarray(self, y_df: pd.DataFrame) -> xr.DataArray:
        """Pivot target DataFrame to xr.DataArray (location, time)."""
        df = y_df.copy()
        df["time_period"] = pd.to_datetime(df["time_period"])
        target_wide = df.pivot(index="time_period", columns="location", values=self._target_variable)
        target_wide = target_wide.sort_index().ffill().bfill()
        locations = list(target_wide.columns)
        times = list(target_wide.index)
        return xr.DataArray(
            target_wide.values.T,
            dims=["location", "time"],
            coords={"location": locations, "time": times},
        )

    def _features_to_xarray(self, X_df: pd.DataFrame) -> xr.DataArray | None:
        """Pivot features DataFrame to xr.DataArray (location, time, feature) or None."""
        index_cols = ["time_period", "location"]
        feature_cols = [c for c in X_df.columns if c not in index_cols]
        if not feature_cols:
            return None

        df = X_df.copy()
        df["time_period"] = pd.to_datetime(df["time_period"])

        feature_arrays = []
        for var in feature_cols:
            var_wide = df.pivot(index="time_period", columns="location", values=var)
            var_wide = var_wide.sort_index().ffill().bfill()
            feature_arrays.append(var_wide.values.T)

        locations = sorted(df["location"].unique().tolist())
        times = sorted(df["time_period"].unique().tolist())

        return xr.DataArray(
            np.stack(feature_arrays, axis=-1),
            dims=["location", "time", "feature"],
            coords={"location": locations, "time": times},
        )

    def _future_features_to_xarray(self, X_df: pd.DataFrame) -> xr.DataArray | None:
        """Pivot future features DataFrame to xr.DataArray (location, step, feature) or None."""
        index_cols = ["time_period", "location"]
        feature_cols = [c for c in X_df.columns if c not in index_cols]
        if not feature_cols:
            return None

        df = X_df.copy()
        df["time_period"] = pd.to_datetime(df["time_period"])
        locations = sorted(df["location"].unique().tolist())

        feature_arrays = []
        for var in feature_cols:
            var_wide = df.pivot(index="time_period", columns="location", values=var)
            var_wide = var_wide.sort_index().ffill().bfill()
            var_wide = var_wide[locations]
            feature_arrays.append(var_wide.values.T)

        return xr.DataArray(
            np.stack(feature_arrays, axis=-1),
            dims=["location", "step", "feature"],
            coords={"location": locations},
        )
