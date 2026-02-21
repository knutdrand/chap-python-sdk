# CHAP Python SDK

A validation and testing framework for chapkit models. This SDK provides tools to test chapkit model implementations against various test datasets with a simple pytest-based testing setup.

## Features

- **Test dataset management** for validating chapkit models
- **Testing utilities** for validating model train/predict functions
- **Assertion helpers** for model I/O validation
- **CLI Adapter** for creating command-line interfaces from train/predict functions
- **Prediction format conversion** utilities
- **Test data generation** based on model requirements
- **pytest integration** for automated testing workflows

## Requirements

- Python 3.13+
- [chapkit](https://github.com/climateview/chapkit) - Model framework

## Quick Example

```python
import pytest
from chapkit.config.schemas import BaseConfig
from chapkit.data import DataFrame

from chap_python_sdk.testing import get_example_data, validate_model_io


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

    result = await validate_model_io(my_train, my_predict, example_data)

    assert result.success, f"Validation failed: {result.errors}"
```

## Documentation

- [Installation](getting-started/installation.md) - How to install the SDK
- [Quick Start](getting-started/quickstart.md) - Get started quickly
- [Model Testing](user-guide/model-testing.md) - Testing your models
- [API Reference](api/index.md) - Complete API documentation

## Related Projects

- [chapkit](https://github.com/climateview/chapkit) - ML/data service modules
- [servicekit](https://github.com/winterop-com/servicekit) - Core service framework
- [chap_r_sdk](https://github.com/climateview/chap_r_sdk) - R version of this SDK

## License

AGPL-3.0-or-later
