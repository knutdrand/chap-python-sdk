"""Forecasting logic wrapper for skforecast."""

import importlib
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from chap_python_sdk.adaptors.skforecast.config import SkforecastConfig

logger = logging.getLogger(__name__)


def _import_class(class_path: str) -> type:
    """Dynamically import a class from a string path."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)  # type: ignore[no-any-return]


def _resolve_transformer(name: str | None) -> Any:
    """Convert a transformer string name to an sklearn instance.

    Args:
        name: Short name like "StandardScaler" or None.

    Returns:
        An sklearn transformer instance or None.
    """
    if name is None:
        return None
    transformer_map: dict[str, str] = {
        "StandardScaler": "sklearn.preprocessing.StandardScaler",
        "MinMaxScaler": "sklearn.preprocessing.MinMaxScaler",
        "RobustScaler": "sklearn.preprocessing.RobustScaler",
    }
    class_path = transformer_map.get(name)
    if class_path is None:
        raise ValueError(f"Unknown transformer: {name!r}. Choose from {list(transformer_map)}")
    cls = _import_class(class_path)
    return cls()


class SkforecastWrapper:
    """Wraps skforecast ForecasterRecursiveMultiSeries."""

    def __init__(self, config: "SkforecastConfig"):
        """Initialize the wrapper with configuration."""
        from skforecast.recursive import ForecasterRecursiveMultiSeries  # type: ignore[import-untyped]

        self.config = config
        self.forecaster: Any = None
        self.residuals_by_step: dict[str, dict[int, np.ndarray]] = {}
        self.ForecasterClass = ForecasterRecursiveMultiSeries

    def _create_forecaster_instance(self) -> Any:
        """Create a new ForecasterRecursiveMultiSeries from config."""
        model_class = _import_class(self.config.model_class)
        regressor = model_class(**self.config.model_params)

        kwargs: dict[str, Any] = {
            "estimator": regressor,
            "lags": self.config.lags,
            "encoding": self.config.encoding,
        }
        if self.config.differentiation is not None:
            kwargs["differentiation"] = self.config.differentiation
        transformer = _resolve_transformer(self.config.transformer_series)
        if transformer is not None:
            kwargs["transformer_series"] = transformer

        return self.ForecasterClass(**kwargs)

    def fit(self, target_wide: pd.DataFrame, exog_wide: pd.DataFrame | None) -> None:
        """Fit forecaster and collect residuals.

        Args:
            target_wide: Target variable in wide format (DatetimeIndex, columns=locations)
            exog_wide: Exogenous variables in wide format (or None)
        """
        self.forecaster = self._create_forecaster_instance()
        self.forecaster.fit(series=target_wide, exog=exog_wide)

        if self.config.use_bootstrapping:
            self._compute_multistep_residuals(target_wide, exog_wide)

    def _compute_multistep_residuals(self, target_wide: pd.DataFrame, exog_wide: pd.DataFrame | None) -> None:
        """Compute multi-step residuals via expanding-window backtesting.

        For each cutoff in the training data, fit a temporary forecaster on
        data up to that cutoff, predict n_prediction_steps ahead, and record
        per-location per-step residuals (actual - predicted).
        """
        n_train = len(target_wide)
        locations = list(target_wide.columns)
        n_steps = self.config.n_prediction_steps

        if isinstance(self.config.lags, int):
            max_lag = self.config.lags
        else:
            max_lag = max(self.config.lags)

        # Collect residuals: step -> location -> list of residuals
        collected: dict[int, dict[str, list[float]]] = {step: {loc: [] for loc in locations} for step in range(n_steps)}

        min_train_size = max_lag + 3
        for cutoff in range(min_train_size, n_train - n_steps + 1):
            series_to_cutoff = target_wide.iloc[:cutoff]
            exog_to_cutoff = exog_wide.iloc[:cutoff] if exog_wide is not None else None

            try:
                temp_forecaster = self._create_forecaster_instance()
                temp_forecaster.fit(series=series_to_cutoff, exog=exog_to_cutoff)

                exog_future = exog_wide.iloc[cutoff : cutoff + n_steps] if exog_wide is not None else None
                preds = temp_forecaster.predict(steps=n_steps, levels=locations, exog=exog_future)

                for step in range(n_steps):
                    for location in locations:
                        actual = target_wide[location].iloc[cutoff + step]
                        predicted = preds[location].iloc[step]
                        collected[step][location].append(float(actual - predicted))
            except Exception:
                logger.debug("Skipping cutoff %d during residual computation", cutoff)
                continue

        # Convert to final structure: location -> step -> np.ndarray
        self.residuals_by_step = {}
        for location in locations:
            self.residuals_by_step[location] = {}
            for step in range(n_steps):
                res = collected[step][location]
                self.residuals_by_step[location][step] = np.array(res) if res else np.array([0.0])

    def refit(self, target_wide: pd.DataFrame, exog_wide: pd.DataFrame | None) -> None:
        """Refit the forecaster on new data with same configuration.

        Args:
            target_wide: New target data in wide format.
            exog_wide: New exogenous data in wide format (or None).
        """
        self.forecaster = self._create_forecaster_instance()
        self.forecaster.fit(series=target_wide, exog=exog_wide)

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

        locations = self.forecaster.series_names_in_

        return bootstrap_recursive_samples(
            forecaster=self.forecaster,
            residuals_by_step=self.residuals_by_step,
            n_steps=steps,
            n_samples=n_samples,
            exog_future=exog_future,
            locations=locations,
        )
