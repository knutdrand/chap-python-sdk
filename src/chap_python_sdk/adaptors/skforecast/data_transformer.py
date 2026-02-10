"""Data format transformations between chapkit and skforecast."""

import pandas as pd  # type: ignore[import-untyped]
from chapkit.data import DataFrame


def chapkit_to_wide(
    data: DataFrame,
    target_variable: str = "disease_cases",
    exogenous_variables: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Convert chapkit long format to pandas wide format for skforecast.

    Args:
        data: chapkit DataFrame in long format [time_period, location, disease_cases, ...]
        target_variable: Name of the target variable column
        exogenous_variables: List of exogenous variable column names

    Returns:
        Tuple of (target_wide, exog_wide) where:
        - target_wide: DataFrame with DatetimeIndex and columns per location
        - exog_wide: DataFrame with exogenous variables in wide format (or None)
    """
    # Convert polars to pandas for transformation
    df = data.to_pandas()

    # Parse time_period to datetime
    df["time_period"] = pd.to_datetime(df["time_period"])

    # Pivot target variable by location
    target_wide = df.pivot(index="time_period", columns="location", values=target_variable)
    target_wide = target_wide.sort_index()

    # Infer frequency for the DatetimeIndex (required by skforecast)
    target_wide.index.freq = pd.infer_freq(target_wide.index)

    # Pivot exogenous variables if specified
    exog_wide = None
    if exogenous_variables:
        exog_dfs = []
        for var in exogenous_variables:
            var_wide = df.pivot(index="time_period", columns="location", values=var)
            # Rename columns to include variable name: var_location
            var_wide.columns = [f"{var}_{col}" for col in var_wide.columns]
            exog_dfs.append(var_wide)

        if exog_dfs:
            exog_wide = pd.concat(exog_dfs, axis=1)
            exog_wide = exog_wide.sort_index()

    return target_wide, exog_wide


def wide_to_chapkit(
    predictions_wide: dict[str, pd.DataFrame],
    future: DataFrame,
) -> DataFrame:
    """Convert wide predictions back to chapkit long format.

    Args:
        predictions_wide: Dict mapping location -> DataFrame with shape (n_steps, n_samples)
        future: chapkit DataFrame with time_period and location for alignment

    Returns:
        chapkit DataFrame with [time_period, location, samples] in long format
    """
    # Extract unique time periods and locations from future
    future_pd = future.to_pandas()
    future_pd["time_period"] = pd.to_datetime(future_pd["time_period"])

    results = []
    for location, pred_df in predictions_wide.items():
        # Get time periods for this location from future
        location_times = future_pd[future_pd["location"] == location]["time_period"].values

        if len(location_times) != len(pred_df):
            raise ValueError(
                f"Mismatch in prediction length for {location}: expected {len(location_times)}, got {len(pred_df)}"
            )

        # Convert each row to samples list
        for i, time_period in enumerate(location_times):
            samples = pred_df.iloc[i].tolist()
            results.append(
                {
                    "time_period": pd.Timestamp(time_period).isoformat(),
                    "location": location,
                    "samples": samples,
                }
            )

    # Convert to chapkit DataFrame
    result_df = DataFrame.from_dict(
        {
            "time_period": [r["time_period"] for r in results],
            "location": [r["location"] for r in results],
            "samples": [r["samples"] for r in results],
        }
    )
    return result_df
