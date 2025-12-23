# Model Testing

This guide covers how to test your chapkit models using the CHAP Python SDK.

## Functional Interface

Models are defined as async functions following the chapkit functional interface.

### Train Function

```python notest
async def on_train(
    config: BaseConfig,
    data: DataFrame,
    run_info: RunInfo | None = None,
    geo: FeatureCollection | None = None,
) -> Any:
    """Train a model and return the trained model object (must be pickleable)."""
    # Training logic here
    return trained_model
```

### Predict Function

```python notest
async def on_predict(
    config: BaseConfig,
    model: Any,
    historic: DataFrame,
    future: DataFrame,
    run_info: RunInfo | None = None,
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

## Validation

### Basic Validation

Use `validate_model_io` to test your model against example data:

```python
import asyncio

# Define model functions
async def on_train(config, data, run_info=None, geo=None):
    return {"mean": 10.0}

async def on_predict(config, model, historic, future, run_info=None, geo=None):
    samples = [[model["mean"]] * 10 for _ in range(len(future))]
    return DataFrame.from_dict({
        "time_period": list(future["time_period"]),
        "location": list(future["location"]),
        "samples": samples,
    })

example_data = get_example_data(country="laos", frequency="monthly")

result = asyncio.run(validate_model_io(on_train, on_predict, example_data))

if result.success:
    print(f"Validation passed with {result.n_predictions} predictions")
else:
    print(f"Validation failed: {result.errors}")
```

### ValidationResult

The validation returns a `ValidationResult` object:

| Attribute | Type | Description |
|-----------|------|-------------|
| `success` | bool | Whether validation passed |
| `errors` | list[str] | List of error messages |
| `warnings` | list[str] | List of warning messages |
| `n_predictions` | int | Number of predictions generated |
| `n_samples` | int | Number of samples per prediction |

### Batch Validation

To validate against multiple datasets:

```python notest
from chap_python_sdk.testing import validate_model_io_all

results = await validate_model_io_all(on_train, on_predict, config)

for dataset, result in results:
    print(f"{dataset}: {'PASS' if result.success else 'FAIL'}")
```

## Prediction Output Format

The predict function must return a DataFrame with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `time_period` | str | Time period identifier (e.g., "2013-04") |
| `location` | str | Location identifier (e.g., "Bokeo") |
| `samples` | list[float] | List of prediction samples |

Example:

```python
predictions = DataFrame.from_dict({
    "time_period": ["2013-04", "2013-05"],
    "location": ["Bokeo", "Bokeo"],
    "samples": [[9, 5, 46], [12, 0, 43]],
})
```

## pytest Integration

### Basic Test

```python notest
import pytest
from chap_python_sdk.testing import get_example_data, validate_model_io


@pytest.mark.asyncio
async def test_my_model():
    """Test my model against example data."""
    example_data = get_example_data(country="laos", frequency="monthly")

    result = await validate_model_io(my_train, my_predict, example_data)

    assert result.success, f"Validation failed: {result.errors}"
```

### Using Fixtures

```python notest
import pytest
from chap_python_sdk.testing import get_example_data, ExampleData


@pytest.fixture
def laos_monthly_data() -> ExampleData:
    """Load Laos monthly example data."""
    return get_example_data(country="laos", frequency="monthly")


@pytest.mark.asyncio
async def test_my_model(laos_monthly_data: ExampleData):
    """Test my model against Laos data."""
    result = await validate_model_io(my_train, my_predict, laos_monthly_data)
    assert result.success
```

## RunInfo Parameters

The SDK supports passing runtime information to models via `RunInfo`:

```python
run_info = RunInfo(
    prediction_length=3,
    additional_continuous_covariates=["extra_covariate"],
)

# RunInfo can be passed to validate_model_io
assert run_info.prediction_length == 3
```

## Next Steps

- Learn about [Test Data Generation](data-generation.md)
- Explore [Assertions](assertions.md)
- Check out [Prediction Formats](prediction-formats.md)
