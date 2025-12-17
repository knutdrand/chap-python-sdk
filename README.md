# chap-python-sdk

A validation and testing framework for chapkit models. This SDK provides tools to test chapkit model implementations against various test datasets with a simple pytest-based testing setup.

## Features

- Test dataset management for validating chapkit models
- Testing utilities for validating model train/predict functions
- Assertion helpers for model I/O validation
- Prediction format conversion utilities
- pytest integration for automated testing workflows

## Requirements

- Python 3.13+
- [chapkit](https://github.com/climateview/chapkit) - Model framework

## Installation

```bash
uv add chap-python-sdk
```

Or with pip:

```bash
pip install chap-python-sdk
```

## Quick Start

### Testing Model Functions

The `chap_python_sdk.testing` module provides utilities for testing models using the chapkit functional interface.

```python
import pytest
from chapkit.config.schemas import BaseConfig
from chapkit.data import DataFrame

from chap_python_sdk.testing import get_example_data, validate_model_io


class MyModelConfig(BaseConfig):
    """Configuration for my model."""

    learning_rate: float = 0.01


async def my_train(config: BaseConfig, data: DataFrame, geo=None):
    """Train the model."""
    return {"means": 10.0}


async def my_predict(config: BaseConfig, model, historic: DataFrame, future: DataFrame, geo=None):
    """Generate predictions."""
    samples = [[model["means"]] * 10 for _ in range(len(future))]
    return DataFrame.from_dict({
        "time_period": list(future["time_period"]),
        "location": list(future["location"]),
        "samples": samples,
    })


@pytest.mark.asyncio
async def test_my_model():
    """Test my model against example data."""
    example_data = get_example_data(country="laos", frequency="monthly")
    config = MyModelConfig()

    result = await validate_model_io(my_train, my_predict, example_data, config)

    assert result.success, f"Validation failed: {result.errors}"
    assert result.n_predictions == 21
    assert result.n_samples >= 1
```

### Using FunctionalModelRunner

If you prefer to bundle your functions into a runner object, you can use chapkit's `FunctionalModelRunner`:

```python
from chapkit import FunctionalModelRunner
from chap_python_sdk.testing import get_example_data, validate_model_io


async def my_train(config, data, geo=None):
    return {"model": "trained"}


async def my_predict(config, model, historic, future, geo=None):
    # prediction logic
    ...


# Create runner from functions
runner = FunctionalModelRunner(on_train=my_train, on_predict=my_predict)

# The SDK also re-exports FunctionalModelRunner for convenience
from chap_python_sdk.testing import FunctionalModelRunner
```

### Assertion Helpers

```python
from chap_python_sdk.testing import (
    assert_valid_predictions,
    assert_prediction_shape,
    assert_samples_column,
)

# Assert that predictions have the expected structure
assert_valid_predictions(predictions, expected_rows=21)

# Assert predictions have correct shape matching future_data
assert_prediction_shape(predictions, future_data)

# Assert samples column contains valid numeric lists
assert_samples_column(predictions, min_samples=1)
```

### Example Data

The SDK includes example datasets for testing:

```python
from chap_python_sdk.testing import get_example_data, list_available_datasets

# List available datasets
datasets = list_available_datasets()
# Returns: [("laos", "monthly"), ...]

# Load example data
example_data = get_example_data(country="laos", frequency="monthly")

# Access individual components
training_data = example_data.training_data      # Historical data for training
historic_data = example_data.historic_data      # Recent observations
future_data = example_data.future_data          # Future periods to predict
expected_predictions = example_data.predictions  # Reference predictions (optional)
```

### Prediction Format Utilities

The SDK supports multiple prediction formats with conversion utilities:

```python
from chap_python_sdk.testing import (
    predictions_to_wide,
    predictions_from_wide,
    predictions_to_long,
    predictions_from_long,
    detect_prediction_format,
)

# Internal format: DataFrame with "samples" column (list of numeric values per row)
# Wide format: sample_0, sample_1, ..., sample_N columns (CHAP CSV output)
# Long format: sample_id, prediction columns (scoringutils format)

# Convert nested to wide (for CHAP output)
wide_predictions = predictions_to_wide(predictions)

# Convert wide to nested (from CHAP CSV)
nested_predictions = predictions_from_wide(wide_dataframe)

# Detect format automatically
format_type = detect_prediction_format(dataframe)  # Returns: "nested", "wide", or "long"
```

## Functional Interface

Models are defined as async functions following the chapkit functional interface:

### Train Function

```python
async def on_train(
    config: BaseConfig,
    data: DataFrame,
    geo: FeatureCollection | None = None,
) -> Any:
    """Train a model and return the trained model object (must be pickleable)."""
    # Training logic here
    return trained_model
```

### Predict Function

```python
async def on_predict(
    config: BaseConfig,
    model: Any,
    historic: DataFrame,
    future: DataFrame,
    geo: FeatureCollection | None = None,
) -> DataFrame:
    """Make predictions using a trained model and return predictions as DataFrame."""
    # Prediction logic here
    return predictions_dataframe
```

### Type Aliases

The SDK provides type aliases for these functions:

```python
from chap_python_sdk.testing import TrainFunction, PredictFunction

# TrainFunction = Callable[[BaseConfig, DataFrame, GeoFeatureCollection | None], Awaitable[Any]]
# PredictFunction = Callable[[BaseConfig, Any, DataFrame, DataFrame, GeoFeatureCollection | None], Awaitable[DataFrame]]
```

### Prediction Output Format

The predict function must return a DataFrame with these columns:

| Column | Type | Description |
|--------|------|-------------|
| time_period | str | Time period identifier (e.g., "2013-04") |
| location | str | Location identifier (e.g., "Bokeo") |
| samples | list[float] | List of prediction samples (1 for deterministic, 100+ for probabilistic) |

## Development

### Setup

```bash
uv sync
```

### Running Tests

```bash
make test
```

### Linting

```bash
make lint
```

## Project Structure

```
chap-python-sdk/
   src/chap_python_sdk/
      __init__.py
      testing/                # Testing utilities
         __init__.py
         validation.py        # Model I/O validation
         assertions.py        # Assert helpers
         predictions.py       # Prediction format utilities
         example_data.py      # Example dataset loading
         types.py             # Type definitions
      data/                   # Bundled test datasets
         ewars_example/
            monthly/
               training_data.csv
               historic_data.csv
               future_data.csv
               predictions.csv
   tests/                     # Test suite
   pyproject.toml
   README.md
```

## Data Formats

### Input Data (CSV)

```csv
time_period,location,rainfall,mean_temperature,disease_cases,population
2000-07,Bokeo,430.119,23.44,0.0,58502.77
2000-08,Bokeo,321.913,23.82,0.0,58502.77
```

### Prediction Output (Nested - Internal)

DataFrame with samples as list column:

| time_period | location | samples               |
|-------------|----------|-----------------------|
| 2013-04     | Bokeo    | [9, 5, 46, ..., 5]    |
| 2013-05     | Bokeo    | [12, 0, 43, ..., 17]  |

### Prediction Output (Wide - CSV)

```csv
time_period,location,sample_0,sample_1,sample_2,...,sample_999
2013-04,Bokeo,9,5,46,...,5
2013-05,Bokeo,12,0,43,...,17
```

## License

AGPL-3.0-or-later

## Related Projects

- [chapkit](https://github.com/climateview/chapkit) - ML/data service modules
- [servicekit](https://github.com/winterop-com/servicekit) - Core service framework
- [chap_r_sdk](https://github.com/climateview/chap_r_sdk) - R version of this SDK
