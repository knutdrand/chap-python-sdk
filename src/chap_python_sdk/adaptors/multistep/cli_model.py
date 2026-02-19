"""CLI-compatible model using DataFrameMultistepModel with plain pandas DataFrames."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import cyclopts

import pandas as pd  # type: ignore[import-untyped]
import xarray as xr
from sklearn.preprocessing import FunctionTransformer  # type: ignore[import-untyped]

from .config import MultistepConfig
from .model import DataFrameMultistepModel
from .one_step_model import ResidualBootstrapModel
from .pipeline import build_feature_transformer, build_target_pipeline


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
        Pickleable dict with trained model and config.
    """
    index_cols = ["time_period", "location"]
    feature_cols = list(config.exogenous_variables) if config.exogenous_variables else []

    y: pd.DataFrame = data[index_cols + [config.target_variable]]  # pyright: ignore[reportAssignmentType]

    feature_transformer = build_feature_transformer(feature_cols, config)
    x_features: pd.DataFrame = feature_transformer.fit_transform(data[index_cols + feature_cols])  # pyright: ignore[reportAssignmentType]

    target_pipeline = build_target_pipeline(config)
    one_step = ResidualBootstrapModel(config.model_class, config.model_params)
    model = DataFrameMultistepModel(
        one_step,
        config.n_target_lags,
        target_pipeline=target_pipeline,
        target_variable=config.target_variable,
    )
    model.fit(x_features, y)

    return {
        "model": model,
        "feature_transformer": feature_transformer,
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
    df_model: DataFrameMultistepModel = model["model"]
    feature_transformer = model.get("feature_transformer") or FunctionTransformer()

    index_cols = ["time_period", "location"]
    y_historic: pd.DataFrame = historic[index_cols + [restored_config.target_variable]]  # pyright: ignore[reportAssignmentType]

    feature_cols = list(restored_config.exogenous_variables) if restored_config.exogenous_variables else []
    x_future: pd.DataFrame = feature_transformer.transform(future[index_cols + feature_cols])  # pyright: ignore[reportAssignmentType]

    n_steps = future.groupby("location").size().iloc[0]
    predictions = df_model.predict(y_historic, x_future, n_steps, restored_config.n_samples)

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
