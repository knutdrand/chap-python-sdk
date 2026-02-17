"""Data format transformations between chapkit DataFrames and xarray."""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import xarray as xr
from chapkit.data import DataFrame


def chapkit_to_xarray(
    data: DataFrame,
    target_variable: str = "disease_cases",
    exogenous_variables: list[str] | None = None,
) -> tuple[xr.DataArray, xr.DataArray | None]:
    """Convert chapkit long format to xarray DataArrays.

    Args:
        data: chapkit DataFrame in long format [time_period, location, disease_cases, ...]
        target_variable: Name of the target variable column.
        exogenous_variables: List of exogenous variable column names.

    Returns:
        Tuple of (y, X) where:
        - y: DataArray with dims (location, time)
        - X: DataArray with dims (location, time, feature) or None
    """
    df = data.to_pandas()
    df["time_period"] = pd.to_datetime(df["time_period"])

    # Pivot target variable: rows=time, columns=location
    target_wide = df.pivot(index="time_period", columns="location", values=target_variable)
    target_wide = target_wide.sort_index().ffill().bfill()

    locations = list(target_wide.columns)
    times = list(target_wide.index)

    y = xr.DataArray(
        target_wide.values.T,  # (location, time)
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
            feature_arrays.append(var_wide.values.T)  # (location, time)

        if feature_arrays:
            # Stack to (location, time, feature)
            # Note: no feature coordinate — MultistepModel.fit_multi concats
            # exog with lag features and requires consistent coordinates.
            exog = xr.DataArray(
                np.stack(feature_arrays, axis=-1),
                dims=["location", "time", "feature"],
                coords={"location": locations, "time": times},
            )

    return y, exog


def chapkit_future_to_xarray(
    future: DataFrame,
    exogenous_variables: list[str] | None = None,
) -> tuple[list[str], list[object], xr.DataArray | None]:
    """Convert future chapkit DataFrame to xarray components.

    Args:
        future: Future chapkit DataFrame (no target column).
        exogenous_variables: List of exogenous variable column names.

    Returns:
        Tuple of (locations, time_periods, X_future) where:
        - locations: List of location identifiers.
        - time_periods: List of unique time periods.
        - X_future: DataArray with dims (location, step, feature) or None.
    """
    df = future.to_pandas()
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
            # Reindex to ensure consistent location ordering
            var_wide = var_wide[locations]
            feature_arrays.append(var_wide.values.T)  # (location, step)

        if feature_arrays:
            X_future = xr.DataArray(
                np.stack(feature_arrays, axis=-1),
                dims=["location", "step", "feature"],
                coords={"location": locations},
            )

    return locations, time_periods, X_future


def xarray_predictions_to_chapkit(
    predictions: xr.DataArray,
    future: DataFrame,
) -> DataFrame:
    """Convert xarray predictions to chapkit long format.

    Args:
        predictions: DataArray with dims (location, trajectory, step).
        future: chapkit DataFrame with time_period and location for alignment.

    Returns:
        chapkit DataFrame with [time_period, location, samples] in long format.
    """
    future_pd = future.to_pandas()
    future_pd["time_period"] = pd.to_datetime(future_pd["time_period"])

    results_time: list[str] = []
    results_location: list[str] = []
    results_samples: list[list[float]] = []

    locations = predictions.coords["location"].values
    for loc in locations:
        loc_str = str(loc)
        loc_times = future_pd[future_pd["location"] == loc_str]["time_period"].sort_values().values

        loc_preds = predictions.sel(location=loc)  # (trajectory, step)
        n_steps = loc_preds.sizes["step"]

        for step_idx in range(n_steps):
            time_period = loc_times[step_idx]
            samples = loc_preds.isel(step=step_idx).values.tolist()  # trajectory values
            results_time.append(str(pd.Timestamp(time_period).isoformat()))  # pyright: ignore[reportAttributeAccessIssue]
            results_location.append(loc_str)
            results_samples.append(samples)

    return DataFrame.from_dict(
        {
            "time_period": results_time,
            "location": results_location,
            "samples": results_samples,
        }
    )
