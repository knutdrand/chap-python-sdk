"""CLI-compatible model using MultistepModel with plain pandas DataFrames."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import cyclopts

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import xarray as xr

from .config import MultistepConfig
from .one_step_model import ResidualBootstrapModel


def pandas_to_xarray(
    df: pd.DataFrame,
    target_variable: str = "disease_cases",
    exogenous_variables: list[str] | None = None,
) -> tuple[xr.DataArray, xr.DataArray | None]:
    """Convert pandas long-format DataFrame to xarray DataArrays.

    Args:
        df: DataFrame in long format [time_period, location, target, ...].
        target_variable: Name of the target variable column.
        exogenous_variables: List of exogenous variable column names.

    Returns:
        Tuple of (y, X) where:
        - y: DataArray with dims (location, time)
        - X: DataArray with dims (location, time, feature) or None
    """
    df = df.copy()
    df["time_period"] = pd.to_datetime(df["time_period"])

    target_wide = df.pivot(index="time_period", columns="location", values=target_variable)
    target_wide = target_wide.sort_index().ffill().bfill()

    locations = list(target_wide.columns)
    times = list(target_wide.index)

    y = xr.DataArray(
        target_wide.values.T,
        dims=["location", "time"],
        coords={"location": locations, "time": times},
    )

    exog: xr.DataArray | None = None
    if exogenous_variables:
        feature_arrays = []
        for var in exogenous_variables:
            if var not in df.columns:
                continue
            var_wide = df.pivot(index="time_period", columns="location", values=var)
            var_wide = var_wide.sort_index().ffill().bfill()
            feature_arrays.append(var_wide.values.T)

        if feature_arrays:
            exog = xr.DataArray(
                np.stack(feature_arrays, axis=-1),
                dims=["location", "time", "feature"],
                coords={"location": locations, "time": times},
            )

    return y, exog


def pandas_future_to_xarray(
    df: pd.DataFrame,
    exogenous_variables: list[str] | None = None,
) -> tuple[list[str], list[object], xr.DataArray | None]:
    """Convert future pandas DataFrame to xarray components.

    Args:
        df: Future DataFrame (no target column).
        exogenous_variables: List of exogenous variable column names.

    Returns:
        Tuple of (locations, time_periods, X_future) where:
        - locations: List of location identifiers.
        - time_periods: List of unique time periods.
        - X_future: DataArray with dims (location, step, feature) or None.
    """
    df = df.copy()
    df["time_period"] = pd.to_datetime(df["time_period"])

    locations = sorted(df["location"].unique().tolist())
    time_periods = sorted(df["time_period"].unique().tolist())

    X_future = None
    if exogenous_variables:
        feature_arrays = []
        for var in exogenous_variables:
            if var not in df.columns:
                continue
            var_wide = df.pivot(index="time_period", columns="location", values=var)
            var_wide = var_wide.sort_index().ffill().bfill()
            var_wide = var_wide[locations]
            feature_arrays.append(var_wide.values.T)

        if feature_arrays:
            X_future = xr.DataArray(
                np.stack(feature_arrays, axis=-1),
                dims=["location", "step", "feature"],
                coords={"location": locations},
            )

    return locations, time_periods, X_future


def xarray_predictions_to_pandas(
    predictions: xr.DataArray,
    future_df: pd.DataFrame,
) -> pd.DataFrame:
    """Convert xarray predictions to pandas long format with samples column.

    Args:
        predictions: DataArray with dims (location, trajectory, step).
        future_df: DataFrame with time_period and location for alignment.

    Returns:
        DataFrame with [time_period, location, samples] columns.
    """
    future_df = future_df.copy()
    # Preserve original time_period strings for output format compatibility
    original_time_strs = future_df["time_period"].astype(str)
    future_df["_original_time"] = original_time_strs
    future_df["time_period"] = pd.to_datetime(future_df["time_period"])

    results_time: list[str] = []
    results_location: list[str] = []
    results_samples: list[list[float]] = []

    locations = predictions.coords["location"].values
    for loc in locations:
        loc_str = str(loc)
        loc_subset = future_df[future_df["location"] == loc_str].sort_values(by="time_period")  # pyright: ignore[reportCallIssue]
        loc_original_times = loc_subset["_original_time"].values

        loc_preds = predictions.sel(location=loc)
        n_steps = loc_preds.sizes["step"]

        for step_idx in range(n_steps):
            samples = loc_preds.isel(step=step_idx).values.tolist()
            results_time.append(str(loc_original_times[step_idx]))
            results_location.append(loc_str)
            results_samples.append(samples)

    return pd.DataFrame(
        {
            "time_period": results_time,
            "location": results_location,
            "samples": results_samples,
        }
    )


def train(config: MultistepConfig, data: pd.DataFrame) -> dict[str, Any]:
    """Train multistep model from pandas DataFrame.

    Args:
        config: MultistepConfig with model parameters.
        data: Training data in long format [time_period, location, disease_cases, ...].

    Returns:
        Pickleable dict with trained model, locations, and config.
    """
    from chap_python_sdk.adaptors.multistep_model import MultistepModel

    y, X = pandas_to_xarray(
        data,
        target_variable=config.target_variable,
        exogenous_variables=config.exogenous_variables,
    )

    one_step = ResidualBootstrapModel(config.model_class, config.model_params)
    model = MultistepModel(one_step, n_target_lags=config.n_target_lags)
    model.fit_multi(y, X)

    return {
        "multistep_model": model,
        "locations": y.coords["location"].values.tolist(),
        "config": config.model_dump(),
    }


def predict(
    config: MultistepConfig,
    model: dict[str, Any],
    historic: pd.DataFrame,
    future: pd.DataFrame,
) -> pd.DataFrame:
    """Generate predictions from pandas DataFrames.

    Args:
        config: MultistepConfig (unused, restored from model dict).
        model: Trained model dict from train().
        historic: Historic data in long format.
        future: Future periods in long format.

    Returns:
        DataFrame with [time_period, location, samples] columns.
    """
    restored_config = MultistepConfig(**model["config"])
    multistep_model = model["multistep_model"]

    y_historic, _ = pandas_to_xarray(
        historic,
        target_variable=restored_config.target_variable,
        exogenous_variables=restored_config.exogenous_variables,
    )
    previous_y = y_historic.isel(time=slice(-restored_config.n_target_lags, None))

    _, time_periods, X_future = pandas_future_to_xarray(
        future,
        exogenous_variables=restored_config.exogenous_variables,
    )
    n_steps = len(time_periods)

    predictions = multistep_model.predict_multi(
        previous_y,
        n_steps=n_steps,
        n_samples=restored_config.n_samples,
        X=X_future,
    )

    return xarray_predictions_to_pandas(predictions, future)


def create_multistep_cli_app(config: MultistepConfig | None = None) -> "cyclopts.App":
    """Create a CLI app for the multistep model.

    Args:
        config: Optional MultistepConfig override (default config used if None).

    Returns:
        Configured cyclopts.App with train-cmd and predict-cmd subcommands.

    Raises:
        ImportError: If scikit-learn is not installed.
    """
    try:
        from sklearn.base import BaseEstimator  # type: ignore[import-untyped]  # noqa: F401
    except ImportError:
        raise ImportError("scikit-learn is not installed. Install with: uv add chap-python-sdk[multistep]")

    from chap_python_sdk.cli_adapter import create_cli_app

    config_class = MultistepConfig if config is None else type(config)

    return create_cli_app(
        train_func=train,
        predict_func=predict,
        config_class=config_class,
        name="multistep-model",
    )
