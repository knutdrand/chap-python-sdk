# Test Data Generation

The SDK can generate synthetic test data based on your model's declared requirements using `MLServiceInfo`. This ensures the generated data matches exactly what your model expects.

## Basic Usage

```python
# Define your model's requirements
service_info = MLServiceInfo(
    required_covariates=["rainfall", "mean_temperature"],
    allow_free_additional_continuous_covariates=False,
    supported_period_type=PeriodType.month,
)

# Configure data generation
config = DataGenerationConfig(
    prediction_length=3,           # Number of periods to predict
    n_locations=5,                 # Number of locations
    n_training_periods=24,         # Training history length
    n_historic_periods=12,         # Recent history for prediction
    seed=42,                       # For reproducibility
)

# Generate test data
example_data = generate_test_data(service_info, config)
```

To use in validation with `asyncio.run()`:

```python
import asyncio

# Generate test data
service_info = MLServiceInfo(
    required_covariates=["rainfall"],
    allow_free_additional_continuous_covariates=False,
    supported_period_type=PeriodType.month,
)
example_data = generate_test_data(service_info, DataGenerationConfig(seed=42))

# Define simple model functions
async def on_train(config, data, run_info=None, geo=None):
    return {"mean": 10.0}

async def on_predict(config, model, historic, future, run_info=None, geo=None):
    samples = [[model["mean"]] * 10 for _ in range(len(future))]
    return DataFrame.from_dict({
        "time_period": list(future["time_period"]),
        "location": list(future["location"]),
        "samples": samples,
    })

# Run validation
result = asyncio.run(validate_model_io(on_train, on_predict, example_data))
assert result.success
```

## MLServiceInfo

The `MLServiceInfo` class describes your model's requirements:

| Attribute | Type | Description |
|-----------|------|-------------|
| `required_covariates` | list[str] | List of required input variables |
| `allow_free_additional_continuous_covariates` | bool | Whether additional covariates are allowed |
| `supported_period_type` | PeriodType | Time period frequency |

### PeriodType Options

- `PeriodType.week` - Weekly data
- `PeriodType.month` - Monthly data
- `PeriodType.year` - Yearly data
- `PeriodType.any` - Any period type

## DataGenerationConfig

Configure the generated data with these options:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prediction_length` | int or None | None | Number of periods to predict. Defaults based on period type (week=4, month=3, year=1) |
| `additional_covariates` | list[str] | [] | Extra covariates to include (model must allow) |
| `n_locations` | int | 3 | Number of locations to generate |
| `n_training_periods` | int | 24 | Number of training periods |
| `n_historic_periods` | int | 12 | Number of historic periods (subset of training) |
| `seed` | int or None | None | Random seed for reproducibility |
| `location_names` | list[str] or None | None | Custom location names (default: Location_A, Location_B, ...) |
| `include_nans` | bool | False | Inject NaN values to test missing value handling |
| `nan_fraction` | float | 0.1 | Fraction of values to replace with NaN (0.0-1.0) |

## Testing Missing Value Handling

Use the `include_nans` parameter to test that your model handles missing values correctly:

```python
service_info = MLServiceInfo(
    required_covariates=["rainfall"],
    allow_free_additional_continuous_covariates=False,
    supported_period_type=PeriodType.month,
)

config = DataGenerationConfig(
    prediction_length=3,
    n_locations=5,
    n_training_periods=24,
    include_nans=True,       # Inject NaN values into numeric columns
    nan_fraction=0.1,        # 10% of values will be NaN
    seed=42,
)

example_data = generate_test_data(service_info, config)

# The generated data will have NaN values in covariates
# (time_period and location columns are never NaN)
```

## Two-Step Generation

For more control, you can generate `RunInfo` and `ExampleData` separately:

```python
service_info = MLServiceInfo(
    required_covariates=["rainfall"],
    allow_free_additional_continuous_covariates=False,
    supported_period_type=PeriodType.month,
)

config = DataGenerationConfig(seed=42)

# Step 1: Generate RunInfo based on model requirements
run_info = generate_run_info(service_info, config)
# run_info.prediction_length and run_info.additional_continuous_covariates are set

# Step 2: Generate ExampleData using both service_info and run_info
example_data = generate_example_data(service_info, run_info, config)
```

## Using Model's Declared Info

If your model exports its `MLServiceInfo`, you can use it directly:

```python notest
from main import info  # Your model's declared MLServiceInfo

from chap_python_sdk.testing import (
    DataGenerationConfig,
    MLServiceInfo,
    PeriodType,
    generate_test_data,
)

# Convert chapkit MLServiceInfo to testing MLServiceInfo
service_info = MLServiceInfo(
    required_covariates=list(info.required_covariates),
    allow_free_additional_continuous_covariates=info.allow_free_additional_continuous_covariates,
    supported_period_type=PeriodType(info.supported_period_type.value),
)

example_data = generate_test_data(service_info, DataGenerationConfig(seed=42))
```

## Built-in Covariate Generators

The SDK includes generators for common covariates:

- `rainfall` - Precipitation values
- `mean_temperature` - Temperature values
- `humidity` - Humidity percentage
- `population` - Population counts

These are automatically used when generating data with matching covariate names.

## Next Steps

- Learn about [Prediction Formats](prediction-formats.md)
- Explore [Assertions](assertions.md)
- Check out the [API Reference](../api/generators.md)
