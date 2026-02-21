"""Data format transformations between chapkit DataFrames and xarray."""

import pandas as pd  # type: ignore[import-untyped]
import xarray as xr
from chapkit.data import DataFrame


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
