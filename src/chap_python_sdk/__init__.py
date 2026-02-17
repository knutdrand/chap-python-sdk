"""Validation and testing framework for chapkit models."""

from chap_python_sdk.adaptors.multistep import MultistepConfig, create_multistep_cli_app, create_multistep_functions
from chap_python_sdk.adaptors.skforecast import SkforecastConfig, create_skforecast_functions
from chap_python_sdk.cli_adapter import create_cli_app
from chap_python_sdk.testing import (
    ExampleData,
    PredictFunction,
    RunInfo,
    TrainFunction,
    ValidationResult,
    assert_consistent_sample_counts,
    assert_no_nan_predictions,
    assert_nonnegative_predictions,
    assert_numeric_samples,
    assert_prediction_shape,
    assert_samples_column,
    assert_time_location_columns,
    assert_valid_predictions,
    assert_wide_format_predictions,
    detect_prediction_format,
    get_example_data,
    has_prediction_samples,
    list_available_datasets,
    predictions_from_long,
    predictions_from_wide,
    predictions_summary,
    predictions_to_long,
    predictions_to_quantiles,
    predictions_to_wide,
    validate_model_io,
    validate_model_io_all,
)

__all__ = [
    # Example data
    "get_example_data",
    "list_available_datasets",
    # Validation
    "validate_model_io",
    "validate_model_io_all",
    # Assertions
    "assert_valid_predictions",
    "assert_prediction_shape",
    "assert_samples_column",
    "assert_consistent_sample_counts",
    "assert_numeric_samples",
    "assert_time_location_columns",
    "assert_wide_format_predictions",
    "assert_nonnegative_predictions",
    "assert_no_nan_predictions",
    # Predictions
    "predictions_to_wide",
    "predictions_from_wide",
    "predictions_to_long",
    "predictions_from_long",
    "detect_prediction_format",
    "has_prediction_samples",
    "predictions_to_quantiles",
    "predictions_summary",
    # Types
    "ExampleData",
    "ValidationResult",
    "TrainFunction",
    "PredictFunction",
    "RunInfo",
    # Adaptors
    "create_multistep_cli_app",
    "create_multistep_functions",
    "MultistepConfig",
    "create_skforecast_functions",
    "SkforecastConfig",
    # CLI
    "create_cli_app",
]
