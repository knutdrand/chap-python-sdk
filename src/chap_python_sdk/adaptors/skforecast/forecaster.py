"""Forecasting logic wrapper for skforecast."""

import importlib
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from chap_python_sdk.adaptors.skforecast.config import SkforecastConfig


def _import_class(class_path: str) -> type:
    """Dynamically import a class from a string path."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)  # type: ignore[no-any-return]


class SkforecastWrapper:
    """Wraps skforecast ForecasterRecursiveMultiSeries."""

    def __init__(self, config: "SkforecastConfig"):
        """Initialize the wrapper with configuration."""
        from skforecast.recursive import ForecasterRecursiveMultiSeries  # type: ignore[import-untyped]

        self.config = config
        self.forecaster: Any = None
        self.residuals_by_location: dict[str, np.ndarray] = {}
        self.ForecasterClass = ForecasterRecursiveMultiSeries

    def fit(self, target_wide: pd.DataFrame, exog_wide: pd.DataFrame | None) -> None:
        """Fit forecaster and collect residuals.

        Args:
            target_wide: Target variable in wide format (DatetimeIndex, columns=locations)
            exog_wide: Exogenous variables in wide format (or None)
        """
        # Create sklearn model from config
        model_class = _import_class(self.config.model_class)
        regressor = model_class(**self.config.model_params)

        # Instantiate ForecasterRecursiveMultiSeries
        self.forecaster = self.ForecasterClass(
            regressor=regressor,
            lags=self.config.lags,
            encoding=self.config.encoding,
        )

        # Fit on wide format data
        self.forecaster.fit(series=target_wide, exog=exog_wide)

        # Compute in-sample residuals for bootstrapping
        # For now we skip this to get basic functionality working
        # TODO: Implement proper residual computation
        if self.config.use_bootstrapping:
            # Use a simple variance-based approach for now
            for location in target_wide.columns:
                self.residuals_by_location[location] = np.random.normal(0, target_wide[location].std(), size=100)

    def _compute_residuals(self, target_wide: pd.DataFrame, exog_wide: pd.DataFrame | None) -> None:
        """Compute in-sample residuals for each location."""
        # Determine the number of lags
        if isinstance(self.config.lags, int):
            max_lag = self.config.lags
        else:
            max_lag = max(self.config.lags)

        # For each location, compute residuals
        for location in target_wide.columns:
            # Get actual values (skip initial lags)
            actual = target_wide[location].iloc[max_lag:].values

            # Get predictions for the training period
            # We need to predict one-step-ahead for each point
            predictions_list = []
            for i in range(max_lag, len(target_wide)):
                # Prepare exog for this step if available
                exog_step = None
                if exog_wide is not None:
                    # Get exog variables for this location at this time step
                    location_exog_cols = [col for col in exog_wide.columns if col.endswith(f"_{location}")]
                    exog_step = exog_wide[location_exog_cols].iloc[[i]]

                # Predict using data up to (but not including) current point
                pred = self.forecaster.predict(steps=1, exog=exog_step, levels=location)
                predictions_list.append(float(pred[location].iloc[0]))

            predictions = np.array(predictions_list)
            residuals = actual - predictions

            self.residuals_by_location[location] = residuals

    def predict_samples(
        self,
        steps: int,
        exog_future: pd.DataFrame | None,
        n_samples: int,
    ) -> dict[str, pd.DataFrame]:
        """Generate probabilistic samples via bootstrap.

        Args:
            steps: Number of steps to predict
            exog_future: Future exogenous variables (or None)
            n_samples: Number of sample trajectories to generate

        Returns:
            Dict mapping location -> DataFrame of shape (n_steps, n_samples)
        """
        if self.forecaster is None:
            raise RuntimeError("Forecaster must be fitted before prediction")

        from chap_python_sdk.adaptors.skforecast.sampling import bootstrap_recursive_samples

        # Get list of locations from forecaster
        locations = self.forecaster.series_names_in_

        return bootstrap_recursive_samples(
            forecaster=self.forecaster,
            residuals_by_location=self.residuals_by_location,
            n_steps=steps,
            n_samples=n_samples,
            exog_future=exog_future,
            locations=locations,
        )
