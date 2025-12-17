# Implementation Plan: chap_python_sdk.testing Module

This document outlines the implementation plan for the `chap_python_sdk.testing` module, which provides testing utilities for validating chapkit model implementations using the functional interface.

## Project Info

- **GitHub**: https://github.com/knutdrand/chap-python-sdk
- **Jira Parent Task**: CLIM-267 (dhis2.atlassian.net)

---

## Current Status

### Completed

- [x] Phase 1: Project setup (pyproject.toml, directory structure, test data)
- [x] Phase 2: Type definitions (types.py)
- [x] Phase 3: Example data loading (example_data.py)
- [x] Phase 4: Prediction format utilities (predictions.py)
- [x] Phase 5: Assertion helpers (assertions.py)
- [x] Phase 6: Model I/O validation (validation.py)
- [x] Phase 7: Public API exports (__init__.py)
- [x] Phase 8: Tests (115 pytest tests passing)
- [x] GitHub repository created and code pushed
- [x] Refactored to functional interface (FunctionalModelRunner)

### Remaining Tasks (Jira Sub-tasks for CLIM-267)

- [ ] **Publish package to PyPI** - Configure and publish the package (optional)

---

## Overview

The testing module enables model developers to:
1. Load example datasets in the format expected by chapkit models
2. Validate that their train/predict functions produce correct output
3. Use assertion helpers for fine-grained testing
4. Convert between prediction formats (nested, wide, long)

## Functional Interface (from chapkit)

The testing module supports the chapkit functional interface via `FunctionalModelRunner`:

```python
from typing import Any, Awaitable, Callable
from geojson_pydantic import FeatureCollection
from chapkit.config.schemas import BaseConfig
from chapkit.data import DataFrame

# Type aliases for model functions
type TrainFunction = Callable[
    [BaseConfig, DataFrame, FeatureCollection | None],
    Awaitable[Any]
]

type PredictFunction = Callable[
    [BaseConfig, Any, DataFrame, DataFrame, FeatureCollection | None],
    Awaitable[DataFrame]
]


# Example train function
async def on_train(
    config: BaseConfig,
    data: DataFrame,
    geo: FeatureCollection | None = None,
) -> Any:
    """Train a model and return the trained model object."""
    return {"model": "trained"}


# Example predict function
async def on_predict(
    config: BaseConfig,
    model: Any,
    historic: DataFrame,
    future: DataFrame,
    geo: FeatureCollection | None = None,
) -> DataFrame:
    """Make predictions using a trained model."""
    return DataFrame.from_dict({...})
```

---

## Phase 1: Project Setup and Core Infrastructure

### 1.1 Update pyproject.toml

Add required dependencies:
- `chapkit` - Model framework (provides DataFrame, BaseConfig, FunctionalModelRunner)
- `pydantic` - Data validation and result schemas
- `pyyaml` - Configuration file handling
- `geojson-pydantic` - GeoJSON types

Dev dependencies:
- `pytest>=8.0`, `pytest-asyncio>=0.23` - Async testing
- `mypy>=1.11`, `pyright>=1.1` - Type checking
- `ruff>=0.5` - Linting/formatting

### 1.2 Create Source Directory Structure

```
src/chap_python_sdk/
   __init__.py
   testing/
      __init__.py
      validation.py
      assertions.py
      predictions.py
      example_data.py
      types.py
```

### 1.3 Create Test Data Directory

Copy test datasets from chap_r_sdk:
```
src/chap_python_sdk/data/
   ewars_example/
      monthly/
         training_data.csv
         historic_data.csv
         future_data.csv
         predictions.csv
         config.yaml
```

---

## Phase 2: Type Definitions (types.py)

### 2.1 Core Type Definitions

```python
"""Type definitions for the testing module."""

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from chapkit.config.schemas import BaseConfig
from chapkit.data import DataFrame
from geojson_pydantic import Feature, FeatureCollection

GeoFeatureCollection = FeatureCollection[Feature[Any, Any]]

# Type aliases for functional model runner interface
type TrainFunction = Callable[
    [BaseConfig, DataFrame, GeoFeatureCollection | None],
    Awaitable[Any]
]
type PredictFunction = Callable[
    [BaseConfig, Any, DataFrame, DataFrame, GeoFeatureCollection | None],
    Awaitable[DataFrame]
]


@dataclass
class ExampleData:
    """Container for example dataset components."""

    training_data: DataFrame
    historic_data: DataFrame
    future_data: DataFrame
    predictions: DataFrame | None = None
    configuration: dict[str, Any] | None = None
    geo: GeoFeatureCollection | None = None


@dataclass
class ValidationResult:
    """Result of model I/O validation."""

    success: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_predictions: int = 0
    n_samples: int = 0
```

---

## Phase 3: Example Data Loading (example_data.py)

### 3.1 Dataset Discovery

```python
def list_available_datasets() -> list[tuple[str, str]]:
    """List all available example datasets as (country, frequency) tuples."""
```

### 3.2 Dataset Loading

```python
def get_example_data(
    country: str,
    frequency: str,
    configuration: dict[str, Any] | None = None,
) -> ExampleData:
    """Load example dataset for the specified country and frequency."""
```

---

## Phase 4: Prediction Format Utilities (predictions.py)

### 4.1 Format Detection

```python
def detect_prediction_format(dataframe: DataFrame) -> str:
    """Detect prediction format: 'nested', 'wide', or 'long'."""
```

### 4.2 Nested <-> Wide Conversion

```python
def predictions_to_wide(predictions: DataFrame) -> DataFrame:
    """Convert nested format (samples column) to wide format (sample_0, sample_1, ...)."""

def predictions_from_wide(predictions: DataFrame) -> DataFrame:
    """Convert wide format to nested format."""
```

### 4.3 Nested <-> Long Conversion

```python
def predictions_to_long(predictions: DataFrame) -> DataFrame:
    """Convert nested format to long format (sample_id, prediction columns)."""

def predictions_from_long(predictions: DataFrame) -> DataFrame:
    """Convert long format to nested format."""
```

### 4.4 Utility Functions

```python
def has_prediction_samples(predictions: DataFrame) -> bool:
    """Check if DataFrame has valid samples column."""

def predictions_to_quantiles(
    predictions: DataFrame,
    probabilities: list[float] | None = None,
) -> DataFrame:
    """Compute quantiles from prediction samples."""

def predictions_summary(
    predictions: DataFrame,
    confidence_intervals: list[float] | None = None,
) -> DataFrame:
    """Add summary statistics (mean, median, CI) to predictions."""
```

---

## Phase 5: Assertion Helpers (assertions.py)

### 5.1 Prediction Structure Assertions

```python
def assert_valid_predictions(
    predictions: DataFrame,
    expected_rows: int | None = None,
) -> None:
    """Assert predictions have valid structure."""

def assert_prediction_shape(
    predictions: DataFrame,
    future_data: DataFrame,
) -> None:
    """Assert predictions match expected shape from future_data."""

def assert_samples_column(
    predictions: DataFrame,
    min_samples: int = 1,
    max_samples: int | None = None,
) -> None:
    """Assert samples column contains valid numeric lists."""
```

### 5.2 Consistency Assertions

```python
def assert_consistent_sample_counts(predictions: DataFrame) -> None:
    """Assert all rows have the same number of samples."""

def assert_numeric_samples(predictions: DataFrame) -> None:
    """Assert all sample values are numeric (int or float)."""

def assert_time_location_columns(predictions: DataFrame) -> None:
    """Assert predictions have time_period and location columns."""
```

---

## Phase 6: Model I/O Validation (validation.py)

### 6.1 Core Validation Function

```python
async def validate_model_io(
    train_function: TrainFunction,
    predict_function: PredictFunction,
    example_data: ExampleData,
    config: BaseConfig | None = None,
) -> ValidationResult:
    """
    Validate model train/predict functions against example data.

    Steps:
    1. Create FunctionalModelRunner from train/predict functions
    2. Call runner.on_train() with training_data
    3. Call runner.on_predict() with historic_data, future_data, and trained model
    4. Validate prediction output structure
    5. Return ValidationResult with success/errors
    """
```

### 6.2 Batch Validation

```python
async def validate_model_io_all(
    train_function: TrainFunction,
    predict_function: PredictFunction,
    config: BaseConfig | None = None,
    country: str | None = None,
    frequency: str | None = None,
) -> ValidationResult:
    """Validate model functions against all matching datasets."""
```

### 6.3 Validation Checks

The validation checks:
- [x] Predictions is a DataFrame
- [x] Has 'samples' column (or can be converted from wide format)
- [x] Samples column contains lists of numeric values
- [x] All rows have the same number of samples
- [x] Row count matches future_data row count
- [x] Has time_period and location columns
- [x] train_function doesn't raise exceptions
- [x] predict_function doesn't raise exceptions

---

## Phase 7: Public API (\_\_init\_\_.py)

### 7.1 Testing Module Exports

```python
# src/chap_python_sdk/testing/__init__.py
"""Testing utilities for chapkit model validation."""

from chapkit import FunctionalModelRunner

from chap_python_sdk.testing.example_data import (
    get_example_data,
    list_available_datasets,
)
from chap_python_sdk.testing.validation import (
    validate_model_io,
    validate_model_io_all,
)
from chap_python_sdk.testing.assertions import (
    assert_valid_predictions,
    assert_prediction_shape,
    assert_samples_column,
    assert_consistent_sample_counts,
    assert_numeric_samples,
    assert_time_location_columns,
)
from chap_python_sdk.testing.predictions import (
    predictions_to_wide,
    predictions_from_wide,
    predictions_to_long,
    predictions_from_long,
    detect_prediction_format,
    has_prediction_samples,
    predictions_to_quantiles,
    predictions_summary,
)
from chap_python_sdk.testing.types import (
    ExampleData,
    ValidationResult,
    TrainFunction,
    PredictFunction,
    GeoFeatureCollection,
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
    "GeoFeatureCollection",
    # Re-export from chapkit
    "FunctionalModelRunner",
]
```

---

## Phase 8: Tests

### 8.1 Test Structure

```
tests/
   __init__.py
   testing/
      __init__.py
      test_example_data.py
      test_predictions.py
      test_assertions.py
      test_validation.py
      test_config_validation.py
      conftest.py  # pytest fixtures
```

### 8.2 Pytest Fixtures (conftest.py)

```python
"""Pytest fixtures for testing chapkit models."""

import pytest
from chap_python_sdk.testing import (
    get_example_data,
    ExampleData,
    TrainFunction,
    PredictFunction,
)


@pytest.fixture
def laos_monthly_data() -> ExampleData:
    """Load Laos monthly example dataset."""
    return get_example_data(country="laos", frequency="monthly")


def create_simple_train_function(n_samples: int = 10) -> TrainFunction:
    """Create a simple train function for testing."""
    async def simple_train(config, data, geo=None):
        return {"mean": 10.0, "n_samples": n_samples}
    return simple_train


def create_simple_predict_function() -> PredictFunction:
    """Create a simple predict function for testing."""
    async def simple_predict(config, model, historic, future, geo=None):
        # Generate predictions
        ...
    return simple_predict


@pytest.fixture
def simple_train_function() -> TrainFunction:
    """Create a simple train function for testing."""
    return create_simple_train_function(n_samples=10)


@pytest.fixture
def simple_predict_function() -> PredictFunction:
    """Create a simple predict function for testing."""
    return create_simple_predict_function()
```

### 8.3 Test Results

- **115 tests passing**
- All ruff/mypy/pyright checks passing

---

## Implementation Order

1. [x] **Phase 1**: Project setup (pyproject.toml, directory structure)
2. [x] **Phase 2**: Type definitions
3. [x] **Phase 3**: Example data loading (copy test data from R SDK)
4. [x] **Phase 4**: Prediction format utilities
5. [x] **Phase 5**: Assertion helpers
6. [x] **Phase 6**: Model I/O validation
7. [x] **Phase 7**: Public API exports
8. [x] **Phase 8**: Tests (115 tests passing)
9. [ ] **Phase 9**: Pytest plugin (optional)

---

## Dependencies Summary

### Runtime
- chapkit - Model framework (DataFrame, BaseConfig, FunctionalModelRunner)
- pydantic>=2.0 - Data validation
- pyyaml>=6.0 - Configuration files
- geojson-pydantic - GeoJSON types

### Development
- pytest>=8.0
- pytest-asyncio>=0.23
- mypy>=1.11
- pyright>=1.1
- ruff>=0.5

---

## Usage Example

```python
"""Example: Testing chapkit model functions."""

import pytest
from chapkit.config.schemas import BaseConfig
from chapkit.data import DataFrame

from chap_python_sdk.testing import (
    get_example_data,
    validate_model_io,
    assert_valid_predictions,
)


class MyModelConfig(BaseConfig):
    """Configuration for my model."""
    learning_rate: float = 0.01


async def my_train(config, data, geo=None):
    """Train the model."""
    return {"means": 10.0}


async def my_predict(config, model, historic, future, geo=None):
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

---

## References

- chapkit: `/Users/knutdr/Sources/chapkit/`
  - `src/chapkit/ml/runner.py` - FunctionalModelRunner, BaseModelRunner
  - `src/chapkit/ml/schemas.py` - ModelRunnerProtocol
  - `src/chapkit/data/__init__.py` - DataFrame type
- chap_r_sdk: `/Users/knutdr/Sources/chap_r_sdk/`
  - `R/validation.R` - Model I/O validation
  - `R/predictions.R` - Prediction format utilities
  - `inst/testdata/` - Example datasets
