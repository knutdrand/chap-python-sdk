from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
import xarray as xr


class Distribution(Protocol):
    """Protocol for a probability distribution that supports sampling."""

    def sample(self, n_samples: int) -> np.ndarray:
        """Returns shape (n_samples, n_rows)."""
        ...


class OneStepModel(Protocol):
    """Protocol for a one-step probabilistic regression model."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit on (n_samples, n_features) features and (n_samples,) targets."""
        ...

    def predict_proba(self, X: np.ndarray) -> Distribution:
        """Return a Distribution over next-step values.

        Args:
            X: Feature matrix, shape (n_rows, n_features).

        Returns:
            Distribution where sample(n) returns shape (n, n_rows).
        """
        ...


def _build_lag_matrix_xr(y: xr.DataArray, n_lags: int) -> xr.DataArray:
    """Build lag matrix from DataArray with a time dim.

    Uses xr.shift() which operates across all other dims (e.g. location)
    simultaneously. No iteration over locations.

    Args:
        y: DataArray with at least a 'time' dim. May also have 'location'.
        n_lags: Number of lags.

    Returns:
        DataArray with an added 'lag' dim, time trimmed by n_lags.
        Lag order: oldest to newest [y(t-n_lags), ..., y(t-1)].
    """
    shifted = [y.shift(time=k) for k in range(n_lags, 0, -1)]
    lag_matrix = xr.concat(shifted, dim="lag")
    return lag_matrix.isel(time=slice(n_lags, None))


def _build_lag_matrix(y: np.ndarray, n_lags: int) -> xr.DataArray:
    """Build a lag matrix from a 1-d time series.

    Returns a DataArray with dims (time, lag), shape (len(y) - n_lags, n_lags).
    Columns ordered oldest to newest: [y(t-n_lags), ..., y(t-1)].
    """
    n = len(y) - n_lags
    cols = [y[i : i + n] for i in range(n_lags)]
    return xr.DataArray(np.column_stack(cols), dims=["time", "lag"])


class MultistepModel:
    """Wraps a OneStepModel to produce multi-step recursive forecasts."""

    def __init__(self, one_step_model: OneStepModel, n_target_lags: int):
        """Initialize with a one-step model and number of target lags.

        Args:
            one_step_model: A fitted or unfitted one-step probabilistic model.
            n_target_lags: Number of lagged target values to use as features.
        """
        self.one_step_model = one_step_model
        self.n_target_lags = n_target_lags

    def fit(self, y: np.ndarray, X: np.ndarray | None = None) -> None:
        """Build lag matrix from y, append to X, and train the one-step model.

        Args:
            y: Target time series, shape (n_timepoints,).
            X: Exogenous features, shape (n_timepoints, n_features) or None.
        """
        lags = _build_lag_matrix(y, self.n_target_lags)
        y_target = y[self.n_target_lags :]

        if X is not None:
            exog = xr.DataArray(X[self.n_target_lags :], dims=["time", "feature"])
            features = xr.concat([exog, lags.rename(lag="feature")], dim="feature")
        else:
            features = lags.rename(lag="feature")

        self.one_step_model.fit(features.values, y_target)

    def fit_multi(
        self,
        y: xr.DataArray,
        X: xr.DataArray | None = None,
    ) -> None:
        """Fit on multi-location data, pooling all locations into one training set.

        Args:
            y: Target values, dims (location, time).
            X: Exogenous features, dims (location, time, feature) or None.
        """
        lags = _build_lag_matrix_xr(y, self.n_target_lags)  # (lag, location, time')
        y_target = y.isel(time=slice(self.n_target_lags, None))  # (location, time')

        lags_feat = lags.rename(lag="feature")  # (feature, location, time')
        if X is not None:
            X_trimmed = X.isel(time=slice(self.n_target_lags, None))
            features = xr.concat(
                [
                    X_trimmed.transpose("feature", "location", "time"),
                    lags_feat,
                ],
                dim="feature",
            )
        else:
            features = lags_feat

        features_stacked = features.stack(sample=("location", "time"))
        y_stacked = y_target.stack(sample=("location", "time"))

        self.one_step_model.fit(
            features_stacked.transpose("sample", "feature").values,
            y_stacked.values,
        )

    def predict_multi(
        self,
        previous_y: xr.DataArray,
        n_steps: int,
        n_samples: int,
        X: xr.DataArray | None = None,
    ) -> xr.DataArray:
        """Generate multi-step forecasts for multiple locations.

        Args:
            previous_y: Recent observations, dims (location, time), >= n_target_lags timepoints.
            n_steps: Number of forecast steps.
            n_samples: Number of sampled trajectories per location.
            X: Known future exogenous features, dims (location, step, feature) or None.

        Returns:
            DataArray with dims (location, trajectory, step).
        """
        locations = previous_y.coords["location"].values
        results = []
        for loc in locations:
            prev = previous_y.sel(location=loc).values
            X_loc = X.sel(location=loc).values if X is not None else None
            dist = self.predict_proba(prev, n_steps, X_loc)
            samples = dist.sample(n_samples)  # (n_samples, n_steps)
            results.append(samples)

        return xr.DataArray(
            np.stack(results),
            dims=["location", "trajectory", "step"],
            coords={"location": locations},
        )

    def predict_proba(
        self,
        previous_y: np.ndarray,
        n_steps: int,
        X: np.ndarray | None = None,
    ) -> MultistepDistribution:
        """Return a lazy MultistepDistribution for recursive sampling.

        Args:
            previous_y: Most recent observed values, shape (n_target_lags,), newest last.
            n_steps: Number of steps to forecast.
            X: Known future exogenous features, shape (n_steps, n_features) or None.
        """
        return MultistepDistribution(
            model=self.one_step_model,
            previous_y=previous_y[-self.n_target_lags :],
            n_steps=n_steps,
            n_target_lags=self.n_target_lags,
            X=X,
        )


class DeterministicOneStepModel(Protocol):
    """Protocol for a one-step deterministic regression model (e.g. sklearn regressor)."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model on features X and targets y."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return point predictions for features X."""
        ...


class DeterministicMultistepModel:
    """Recursive multi-step forecaster using point predictions only (no sampling).

    Each step feeds the point prediction forward as input to the next step.
    Supports multi-location pooling via fit_multi/predict_multi.
    """

    def __init__(self, one_step_model: DeterministicOneStepModel, n_target_lags: int):
        """Initialize with a one-step deterministic model and lag count."""
        self.one_step_model = one_step_model
        self.n_target_lags = n_target_lags

    def fit(self, y: np.ndarray, X: np.ndarray | None = None) -> None:
        """Build lag matrix from y, append to X, and train the one-step model."""
        lags = _build_lag_matrix(y, self.n_target_lags)
        y_target = y[self.n_target_lags :]

        if X is not None:
            exog = xr.DataArray(X[self.n_target_lags :], dims=["time", "feature"])
            features = xr.concat([exog, lags.rename(lag="feature")], dim="feature")
        else:
            features = lags.rename(lag="feature")

        self.one_step_model.fit(features.values, y_target)

    def fit_multi(self, y: xr.DataArray, X: xr.DataArray | None = None) -> None:
        """Fit on multi-location data, pooling all locations."""
        lags = _build_lag_matrix_xr(y, self.n_target_lags)
        y_target = y.isel(time=slice(self.n_target_lags, None))

        lags_feat = lags.rename(lag="feature")
        if X is not None:
            X_trimmed = X.isel(time=slice(self.n_target_lags, None))
            features = xr.concat(
                [X_trimmed.transpose("feature", "location", "time"), lags_feat],
                dim="feature",
            )
        else:
            features = lags_feat

        features_stacked = features.stack(sample=("location", "time"))
        y_stacked = y_target.stack(sample=("location", "time"))

        self.one_step_model.fit(
            features_stacked.transpose("sample", "feature").values,
            y_stacked.values,
        )

    def predict(
        self,
        previous_y: np.ndarray,
        n_steps: int,
        X: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate deterministic multi-step forecast.

        Args:
            previous_y: Recent observations, shape (>= n_target_lags,).
            n_steps: Number of forecast steps.
            X: Known future exogenous features, shape (n_steps, n_features) or None.

        Returns:
            Array of shape (n_steps,) with point predictions.
        """
        lag_window = previous_y[-self.n_target_lags :].copy().astype(float)
        results = []
        for step in range(n_steps):
            if X is not None:
                features = np.concatenate([X[step], lag_window]).reshape(1, -1)
            else:
                features = lag_window.reshape(1, -1)
            pred = float(self.one_step_model.predict(features)[0])
            results.append(pred)
            lag_window = np.roll(lag_window, -1)
            lag_window[-1] = pred
        return np.array(results)

    def predict_multi(
        self,
        previous_y: xr.DataArray,
        n_steps: int,
        X: xr.DataArray | None = None,
    ) -> xr.DataArray:
        """Generate deterministic multi-step forecasts for multiple locations.

        Returns:
            DataArray with dims (location, step).
        """
        locations = previous_y.coords["location"].values
        results = []
        for loc in locations:
            prev = previous_y.sel(location=loc).values
            X_loc = X.sel(location=loc).values if X is not None else None
            preds = self.predict(prev, n_steps, X_loc)
            results.append(preds)

        return xr.DataArray(
            np.stack(results),
            dims=["location", "step"],
            coords={"location": locations},
        )


class PerStepMultistepModel:
    """Multi-step model that trains separate models per forecast step.

    At each step k, features where ``get_lag_idx(col) is not None`` and
    ``get_lag_idx(col) < k`` are dropped, since those lagged values would
    not be available at that forecast horizon.
    """

    def __init__(
        self,
        model_factory: Callable[[], DeterministicOneStepModel],
        n_target_lags: int,
        n_steps: int,
        get_lag_idx: Callable[[str], int | None],
        feature_names: list[str] | None = None,
    ) -> None:
        """Initialize with a factory for creating one-step models.

        Args:
            model_factory: Callable that returns a fresh DeterministicOneStepModel.
            n_target_lags: Number of lagged target values used as features.
            n_steps: Number of forecast steps (one model per step).
            get_lag_idx: Callback returning the lag index for a feature column,
                or None if the column is not a lagged feature.
            feature_names: Names of the exogenous feature columns (excluding target lags).
        """
        self.model_factory = model_factory
        self.n_target_lags = n_target_lags
        self.n_steps = n_steps
        self.get_lag_idx = get_lag_idx
        self.feature_names = feature_names or []
        self._models: list[DeterministicOneStepModel] = []
        self._feature_masks: list[list[bool]] = []

    def _build_feature_mask(self, step: int) -> list[bool]:
        """Return a boolean mask over exogenous feature columns for a given step.

        True means the feature is available at that step.
        """
        mask = []
        for col in self.feature_names:
            lag = self.get_lag_idx(col)
            if lag is not None and lag < step:
                mask.append(False)
            else:
                mask.append(True)
        return mask

    def fit(self, y: np.ndarray, X: np.ndarray | None = None) -> None:
        """Train n_steps separate models with step-appropriate feature subsets.

        Args:
            y: Target time series, shape (n_timepoints,).
            X: Exogenous features, shape (n_timepoints, n_features) or None.
        """
        lags = _build_lag_matrix(y, self.n_target_lags)
        y_target = y[self.n_target_lags :]

        self._models = []
        self._feature_masks = []

        for step in range(self.n_steps):
            mask = self._build_feature_mask(step)
            self._feature_masks.append(mask)

            if X is not None:
                X_trimmed = X[self.n_target_lags :]
                X_masked = X_trimmed[:, mask]
                exog = xr.DataArray(X_masked, dims=["time", "feature"])
                features = xr.concat([exog, lags.rename(lag="feature")], dim="feature")
            else:
                features = lags.rename(lag="feature")

            model = self.model_factory()
            model.fit(features.values, y_target)
            self._models.append(model)

    def predict(
        self,
        previous_y: np.ndarray,
        X: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate deterministic multi-step forecast using per-step models.

        Args:
            previous_y: Recent observations, shape (>= n_target_lags,).
            X: Known future exogenous features, shape (n_steps, n_features) or None.

        Returns:
            Array of shape (n_steps,) with point predictions.
        """
        lag_window = previous_y[-self.n_target_lags :].copy().astype(float)
        results = []
        for step in range(self.n_steps):
            mask = self._feature_masks[step]
            if X is not None:
                X_step = X[step][mask]
                features = np.concatenate([X_step, lag_window]).reshape(1, -1)
            else:
                features = lag_window.reshape(1, -1)
            pred = float(self._models[step].predict(features)[0])
            results.append(pred)
            lag_window = np.roll(lag_window, -1)
            lag_window[-1] = pred
        return np.array(results)


class MultistepDistribution:
    """Lazy distribution that runs recursive trajectory sampling on .sample()."""

    def __init__(
        self,
        model: OneStepModel,
        previous_y: np.ndarray,
        n_steps: int,
        n_target_lags: int,
        X: np.ndarray | None,
    ):
        """Store configuration for deferred recursive sampling."""
        self._model = model
        self._previous_y = previous_y
        self._n_steps = n_steps
        self._n_target_lags = n_target_lags
        self._X = X

    def sample(self, n: int) -> np.ndarray:
        """Generate n recursive trajectories.

        Returns shape (n, n_steps). Each row is one sampled trajectory.
        """
        lag_window = xr.DataArray(
            np.tile(self._previous_y, (n, 1)),
            dims=["trajectory", "lag"],
        )

        step_results: list[xr.DataArray] = []
        for step in range(self._n_steps):
            if self._X is not None:
                exog = xr.DataArray(
                    np.tile(self._X[step], (n, 1)),
                    dims=["trajectory", "feature"],
                )
                features = xr.concat([exog, lag_window.rename(lag="feature")], dim="feature")
            else:
                features = lag_window.rename(lag="feature")

            dist = self._model.predict_proba(features.values)
            step_samples = xr.DataArray(dist.sample(1)[0], dims=["trajectory"])
            step_results.append(step_samples)

            # Shift lag window: drop oldest lag, append new sample
            lag_window = lag_window.roll(lag=-1)
            lag_window[{"lag": -1}] = step_samples

        trajectories = xr.concat(step_results, dim="step")
        return trajectories.transpose("trajectory", "step").values
