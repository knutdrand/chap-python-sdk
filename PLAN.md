# Implementation Plan: chap_python_sdk.testing Module

This document outlines the implementation plan for the `chap_python_sdk.testing` module, which provides testing utilities for validating chapkit model implementations following the `BaseModelRunner` interface.

## Overview

The testing module enables model developers to:
1. Load example datasets in the format expected by chapkit models
2. Validate that their `BaseModelRunner` implementations produce correct output
3. Use assertion helpers for fine-grained testing
4. Convert between prediction formats (nested, wide, long)

## BaseModelRunner Interface (from chapkit)

The testing module supports models implementing the chapkit `BaseModelRunner` interface:

```python
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar
from geojson_pydantic import FeatureCollection
from chapkit.config.schemas import BaseConfig
from chapkit.data import DataFrame

ConfigT = TypeVar("ConfigT", bound=BaseConfig)

class BaseModelRunner(ABC, Generic[ConfigT]):
    """Abstract base class for model runners."""

    async def on_init(self) -> None:
        """Optional initialization hook."""
        pass

    async def on_cleanup(self) -> None:
        """Optional cleanup hook."""
        pass

    @abstractmethod
    async def on_train(
        self,
        config: ConfigT,
        data: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> Any:
        """Train a model and return the trained model object."""
        ...

    @abstractmethod
    async def on_predict(
        self,
        config: ConfigT,
        model: Any,
        historic: DataFrame,
        future: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> DataFrame:
        """Make predictions using a trained model."""
        ...
```

---

## Phase 1: Project Setup and Core Infrastructure

### 1.1 Update pyproject.toml

Add required dependencies:
- `chapkit` - Model framework (provides DataFrame, BaseConfig, BaseModelRunner)
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

from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from chapkit.config.schemas import BaseConfig
from chapkit.data import DataFrame
from geojson_pydantic import FeatureCollection

ConfigT = TypeVar("ConfigT", bound=BaseConfig)


@dataclass
class ExampleData:
    """Container for example dataset components."""

    training_data: DataFrame
    historic_data: DataFrame
    future_data: DataFrame
    predictions: DataFrame | None
    configuration: BaseConfig | None
    geo: FeatureCollection | None


@dataclass
class ValidationResult:
    """Result of model I/O validation."""

    success: bool
    errors: list[str]
    warnings: list[str]
    n_predictions: int
    n_samples: int


class ModelRunnerProtocol(Protocol[ConfigT]):
    """Protocol for chapkit model runner interface."""

    async def on_init(self) -> None: ...
    async def on_cleanup(self) -> None: ...

    async def on_train(
        self,
        config: ConfigT,
        data: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> Any: ...

    async def on_predict(
        self,
        config: ConfigT,
        model: Any,
        historic: DataFrame,
        future: DataFrame,
        geo: FeatureCollection | None = None,
    ) -> DataFrame: ...
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
    config_class: type[BaseConfig] | None = None,
) -> ExampleData:
    """Load example dataset for the specified country and frequency."""
```

### 3.3 DataFrame Loading

```python
def load_data_file(file_path: Path) -> DataFrame:
    """Load CSV file into chapkit DataFrame with automatic column detection."""
```

Time column detection (priority order):
- `time_period`
- `date`
- `week`
- `month`
- `year`

Location column detection (priority order):
- `location`
- `region`
- `district`
- `area`
- `site`

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

### 5.3 Model Runner Assertions

```python
async def assert_model_runner_interface(
    runner: ModelRunnerProtocol,
) -> None:
    """Assert runner implements required BaseModelRunner methods."""

async def assert_train_returns_pickleable(
    runner: ModelRunnerProtocol,
    config: BaseConfig,
    data: DataFrame,
) -> None:
    """Assert on_train returns a pickleable object."""
```

---

## Phase 6: Model I/O Validation (validation.py)

### 6.1 Core Validation Function

```python
async def validate_model_io(
    runner: ModelRunnerProtocol,
    example_data: ExampleData,
    config: BaseConfig | None = None,
) -> ValidationResult:
    """
    Validate model runner against example data.

    Steps:
    1. Call runner.on_init()
    2. Call runner.on_train() with training_data
    3. Call runner.on_predict() with historic_data, future_data, and trained model
    4. Validate prediction output structure
    5. Call runner.on_cleanup()
    6. Return ValidationResult with success/errors
    """
```

### 6.2 Batch Validation

```python
async def validate_model_io_all(
    runner: ModelRunnerProtocol,
    config: BaseConfig | None = None,
    country: str | None = None,
    frequency: str | None = None,
) -> ValidationResult:
    """Validate model runner against all matching datasets."""
```

### 6.3 Validation Checks

The validation should check:
- [ ] Predictions is a DataFrame
- [ ] Has 'samples' column (or can be converted from wide format)
- [ ] Samples column contains lists of numeric values
- [ ] All rows have the same number of samples
- [ ] Row count matches future_data row count
- [ ] Has time_period and location columns
- [ ] on_train doesn't raise exceptions
- [ ] on_predict doesn't raise exceptions
- [ ] on_init/on_cleanup are called properly

---

## Phase 7: Public API (\_\_init\_\_.py)

### 7.1 Testing Module Exports

```python
# src/chap_python_sdk/testing/__init__.py
"""Testing utilities for chapkit model validation."""

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
    assert_model_runner_interface,
    assert_train_returns_pickleable,
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
    ModelRunnerProtocol,
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
    "assert_model_runner_interface",
    "assert_train_returns_pickleable",
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
    "ModelRunnerProtocol",
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
      conftest.py  # pytest fixtures
```

### 8.2 Pytest Fixtures (conftest.py)

```python
"""Pytest fixtures for testing chapkit models."""

import pytest
from chap_python_sdk.testing import get_example_data, ExampleData


@pytest.fixture
def laos_monthly_data() -> ExampleData:
    """Load Laos monthly example dataset."""
    return get_example_data(country="laos", frequency="monthly")


@pytest.fixture
def simple_model_runner():
    """Create a simple deterministic model runner for testing."""
    ...
```

### 8.3 Test Cases

**test_example_data.py:**
- Test list_available_datasets returns expected datasets
- Test get_example_data loads all components
- Test training_data has expected columns
- Test historic_data has expected columns
- Test future_data has expected columns
- Test invalid country/frequency raises error

**test_predictions.py:**
- Test detect_prediction_format for each format
- Test nested to wide conversion
- Test wide to nested conversion
- Test nested to long conversion
- Test long to nested conversion
- Test round-trip conversions preserve data
- Test has_prediction_samples
- Test predictions_to_quantiles
- Test predictions_summary

**test_assertions.py:**
- Test assert_valid_predictions passes for valid data
- Test assert_valid_predictions fails for invalid data
- Test assert_prediction_shape matches future_data
- Test assert_samples_column validates samples
- Test assertion error messages are descriptive

**test_validation.py:**
- Test validate_model_io with deterministic model (1 sample)
- Test validate_model_io with probabilistic model (100 samples)
- Test validation calls on_init and on_cleanup
- Test validation fails when samples column missing
- Test validation fails when row count mismatch
- Test validation fails when on_train raises exception
- Test validation fails when on_predict raises exception
- Test validate_model_io_all runs against all datasets

---

## Phase 9: Pytest Plugin (Optional)

### 9.1 Pytest Markers

```python
# pytest plugin providing markers for model testing
@pytest.mark.chapkit_model
def test_my_model():
    ...
```

### 9.2 Auto-Discovery

Consider auto-discovering BaseModelRunner implementations for testing.

---

## Implementation Order

1. **Phase 1**: Project setup (pyproject.toml, directory structure)
2. **Phase 2**: Type definitions
3. **Phase 3**: Example data loading (copy test data from R SDK)
4. **Phase 4**: Prediction format utilities
5. **Phase 5**: Assertion helpers
6. **Phase 6**: Model I/O validation
7. **Phase 7**: Public API exports
8. **Phase 8**: Tests
9. **Phase 9**: Pytest plugin (optional)

---

## Dependencies Summary

### Runtime
- chapkit - Model framework (DataFrame, BaseConfig, BaseModelRunner)
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
"""Example: Testing a chapkit model runner."""

import pytest
from chapkit.ml.runner import BaseModelRunner
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


class MyModelRunner(BaseModelRunner[MyModelConfig]):
    """My chapkit model implementation."""

    async def on_train(self, config, data, geo=None):
        """Train the model."""
        # Training logic here
        return {"means": data.mean()}

    async def on_predict(self, config, model, historic, future, geo=None):
        """Generate predictions."""
        # Prediction logic here
        samples = [[model["means"]] for _ in range(len(future))]
        return DataFrame({
            "time_period": future["time_period"],
            "location": future["location"],
            "samples": samples,
        })


@pytest.mark.asyncio
async def test_my_model():
    """Test MyModelRunner against example data."""
    runner = MyModelRunner()
    example_data = get_example_data(country="laos", frequency="monthly")
    config = MyModelConfig()

    result = await validate_model_io(runner, example_data, config)

    assert result.success, f"Validation failed: {result.errors}"
    assert result.n_predictions == 21
    assert result.n_samples >= 1
```

---

## References

- chapkit: `/Users/knutdr/Sources/chapkit/`
  - `src/chapkit/ml/runner.py` - BaseModelRunner interface
  - `src/chapkit/ml/schemas.py` - ModelRunnerProtocol
  - `src/chapkit/data/__init__.py` - DataFrame type
- chap_r_sdk: `/Users/knutdr/Sources/chap_r_sdk/`
  - `R/validation.R` - Model I/O validation
  - `R/predictions.R` - Prediction format utilities
  - `inst/testdata/` - Example datasets
