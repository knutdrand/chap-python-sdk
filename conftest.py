"""Pytest configuration for markdown documentation tests."""

import asyncio
from typing import Any

import pytest
from chapkit.config.schemas import BaseConfig
from chapkit.data import DataFrame

from chap_python_sdk.testing import (
    DataGenerationConfig,
    ExampleData,
    MLServiceInfo,
    PeriodType,
    PredictionValidationError,
    RunInfo,
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
    generate_example_data,
    generate_run_info,
    generate_test_data,
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


async def my_train(
    config: BaseConfig,
    data: DataFrame,
    run_info: Any = None,
    geo: Any = None,
) -> dict[str, float]:
    """Example train function for documentation tests."""
    return {"means": 10.0}


async def my_predict(
    config: BaseConfig,
    model: Any,
    historic: DataFrame,
    future: DataFrame,
    run_info: Any = None,
    geo: Any = None,
) -> DataFrame:
    """Example predict function for documentation tests."""
    samples = [[model["means"]] * 10 for _ in range(len(future))]
    return DataFrame.from_dict(
        {
            "time_period": list(future["time_period"]),
            "location": list(future["location"]),
            "samples": samples,
        }
    )


# Pre-loaded example data for documentation tests
example_data = get_example_data(country="laos", frequency="monthly")

# Default config for documentation tests
config = BaseConfig()


def pytest_markdown_docs_globals() -> dict[str, Any]:
    """Provide common imports and objects to all markdown code blocks."""
    return {
        # Standard library
        "asyncio": asyncio,
        "pytest": pytest,
        # Chapkit
        "BaseConfig": BaseConfig,
        "DataFrame": DataFrame,
        # Pre-defined model functions for examples
        "my_train": my_train,
        "my_predict": my_predict,
        # Pre-loaded data for examples
        "example_data": example_data,
        "config": config,
        # SDK types
        "DataGenerationConfig": DataGenerationConfig,
        "ExampleData": ExampleData,
        "MLServiceInfo": MLServiceInfo,
        "PeriodType": PeriodType,
        "PredictionValidationError": PredictionValidationError,
        "RunInfo": RunInfo,
        "ValidationResult": ValidationResult,
        # Validation functions
        "validate_model_io": validate_model_io,
        "validate_model_io_all": validate_model_io_all,
        # Assertion functions
        "assert_consistent_sample_counts": assert_consistent_sample_counts,
        "assert_no_nan_predictions": assert_no_nan_predictions,
        "assert_nonnegative_predictions": assert_nonnegative_predictions,
        "assert_numeric_samples": assert_numeric_samples,
        "assert_prediction_shape": assert_prediction_shape,
        "assert_samples_column": assert_samples_column,
        "assert_time_location_columns": assert_time_location_columns,
        "assert_valid_predictions": assert_valid_predictions,
        "assert_wide_format_predictions": assert_wide_format_predictions,
        # Prediction functions
        "detect_prediction_format": detect_prediction_format,
        "has_prediction_samples": has_prediction_samples,
        "predictions_from_long": predictions_from_long,
        "predictions_from_wide": predictions_from_wide,
        "predictions_summary": predictions_summary,
        "predictions_to_long": predictions_to_long,
        "predictions_to_quantiles": predictions_to_quantiles,
        "predictions_to_wide": predictions_to_wide,
        # Generator functions
        "generate_example_data": generate_example_data,
        "generate_run_info": generate_run_info,
        "generate_test_data": generate_test_data,
        # Example data functions
        "get_example_data": get_example_data,
        "list_available_datasets": list_available_datasets,
    }
