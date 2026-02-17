# my-model - Advanced chapkit CLI Example

A chapkit model demonstrating custom CLI implementation for advanced features.

## Installation

```bash
# Install with uv (recommended)
uv sync

# Or with pip
pip install -e ".[dev]"
```

## Usage

### CLI

Train and predict using the command-line interface:

```bash
# Train model
uv run my-model train-cmd train_data.csv model.pkl

# Generate predictions
uv run my-model predict-cmd model.pkl historic.csv future.csv predictions.csv

# Or use directly with python -m
uv run python -m my_model.cli train-cmd train_data.csv model.pkl
```

### Python API

```python
import asyncio
from chap_python_sdk.testing import get_example_data, validate_model_io
from my_model.model import train, predict, MyModelConfig

# Load example data
example_data = get_example_data(country="laos", frequency="monthly")
config = MyModelConfig()

# Validate model
result = asyncio.run(validate_model_io(train, predict, example_data, config))
print(f"Validation: {'PASSED' if result.success else 'FAILED'}")
```

### CHAP Integration

This model can be used with CHAP via the `MLproject` file:

```bash
chap evaluate --model-name /path/to/my-model --dataset-name ISIMIP_dengue_harmonized
```

## Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=my_model
```

## Development

```bash
# Format code
ruff format src tests

# Lint code
ruff check src tests

# Type check
mypy src
```

## Why Custom CLI?

This model uses a **custom cyclopts CLI** instead of `chap_python_sdk.create_cli_app` because it:

- Uses chapkit `DataFrame` class (not pandas)
- Supports async functions with advanced async patterns
- Accepts optional `run_info` and `geo` parameters for chapkit integration
- Uses `DataFrame.from_csv()` for data loading (chapkit-specific method)
- Requires custom data handling beyond simple CSV reading

## When to Use This Pattern

Use a custom CLI (like this example) when:

- Building production chapkit models
- Need optional parameters beyond config/data
- Using chapkit DataFrame class
- Require custom serialization/deserialization

For simpler pandas-based models, use `create_cli_app` instead:

```python
from chap_python_sdk import create_cli_app
app = create_cli_app(train_func, predict_func, ConfigClass, "model-name")
```

See `src/my_model/cli.py` for the custom CLI implementation pattern, and the [SDK CLI documentation](https://knutdrand.github.io/chap-python-sdk/user-guide/cli/) for details on `create_cli_app`.
