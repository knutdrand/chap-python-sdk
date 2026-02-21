# Building CLI Interfaces

## Overview

The `create_cli_app` function provides a simple way to create command-line interfaces for models that use pandas DataFrames and follow standard train/predict patterns. It automatically generates CLI commands with proper argument parsing, async/sync detection, and output formatting compatible with CHAP's `evaluate2` command.

## When to Use create_cli_app

Use `create_cli_app` when your model:

- **Uses pandas DataFrames** for input/output data
- **Has synchronous or async** train/predict functions
- **Follows standard signatures**: `train(config, data)` and `predict(model, historic_data, future_data)`
- **Doesn't require** optional `run_info` or `geo` parameters
- **Works with CSV files** for data input/output

This covers the majority of simple to moderate complexity models.

## Basic Usage

Here's a minimal example:

```python
from chap_python_sdk import create_cli_app
import pandas as pd
from dataclasses import dataclass

@dataclass
class MyModelConfig:
    learning_rate: float = 0.01
    max_depth: int = 10

def train(config: MyModelConfig, data: pd.DataFrame) -> dict:
    """Train a model on the provided data."""
    # Your training logic here
    model = {"config": config, "trained": True}
    return model

def predict(model: dict, historic_data: pd.DataFrame, future_data: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions using the trained model."""
    # Your prediction logic here
    n_samples = 100
    predictions = pd.DataFrame({
        "samples": [[i] * len(future_data) for i in range(n_samples)]
    })
    return predictions

# Create the CLI app
app = create_cli_app(
    train_func=train,
    predict_func=predict,
    config_class=MyModelConfig,
    name="my-model"
)

if __name__ == "__main__":
    app()
```

## Generated Commands

`create_cli_app` generates two commands:

### train-cmd

Trains a model and saves it to a pickle file.

```bash
my-model train-cmd <train_data.csv> <output_model.pkl> [--learning-rate 0.01] [--max-depth 10]
```

**Arguments**:
- `train_data.csv`: CSV file with training data
- `output_model.pkl`: Path where the trained model will be saved

**Config options**: All fields from your config class become optional CLI flags.

### predict-cmd

Loads a trained model and generates predictions.

```bash
my-model predict-cmd <model.pkl> <historic.csv> <future.csv> <output.csv>
```

**Arguments**:
- `model.pkl`: Trained model file (from train-cmd)
- `historic.csv`: Historical data CSV
- `future.csv`: Future period data CSV
- `output.csv`: Where predictions will be written

### Output Format Conversion

The predictions are automatically converted to **wide format** for compatibility with CHAP's `evaluate2` command:

**Input** (nested samples column):
```python-nt
pd.DataFrame({"samples": [[1, 2, 3], [4, 5, 6]]})
```

**Output CSV** (wide format):
```csv
sample_0,sample_1,sample_2
1,2,3
4,5,6
```

This format allows `chap evaluate2` to compute proper quantile-based metrics.

## Async Function Support

`create_cli_app` automatically detects and handles async functions:

```python-nt
import asyncio

async def async_train(config: MyModelConfig, data: pd.DataFrame) -> dict:
    """Async training function."""
    await asyncio.sleep(0.1)  # Simulate async operation
    return {"model": "trained"}

async def async_predict(model: dict, historic: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    """Async prediction function."""
    await asyncio.sleep(0.1)
    return pd.DataFrame({"samples": [[1, 2, 3]]})

# Works exactly the same way
app = create_cli_app(
    train_func=async_train,
    predict_func=async_predict,
    config_class=MyModelConfig,
    name="async-model"
)
```

The CLI will automatically use `asyncio.run()` when needed.

## Advanced Pattern: Custom CLI for chapkit Models

For more complex models, you may need to create a **custom CLI** instead of using `create_cli_app`. This is necessary when:

- **Using chapkit DataFrames** instead of pandas (requires `DataFrame.from_csv()`)
- **Accepting optional parameters** like `run_info` or `geo` in train/predict functions
- **Custom data loading** logic beyond simple CSV reading
- **Different function signatures** than the standard pattern

### Example: my-model

The `my-model/` directory in this repository shows a complete example of a custom CLI for an advanced chapkit model. Key differences:

```python-nt
# Custom CLI (my-model/src/my_model/cli.py)
from cyclopts import App
from chapkit import DataFrame

app = App(name="my-model")

@app.command
def train(
    train_data: Path,
    model_path: Path,
    run_info: Optional[RunInfo] = None,  # Optional chapkit parameter
    geo: Optional[GeoData] = None,       # Optional chapkit parameter
):
    # Use chapkit DataFrame
    data = DataFrame.from_csv(train_data)
    # ... rest of custom logic
```

See `my-model/src/my_model/cli.py` for the complete implementation pattern.

## Configuration Classes

The `config_class` parameter defines the model configuration schema:

```python
from dataclasses import dataclass

@dataclass
class AdvancedConfig:
    # Required parameters (no default)
    model_type: str

    # Optional parameters (with defaults)
    learning_rate: float = 0.01
    max_depth: int = 10
    use_gpu: bool = False
```

These become CLI arguments:

```bash
my-model train-cmd data.csv model.pkl \
    --model-type "random_forest" \
    --learning-rate 0.05 \
    --max-depth 20 \
    --use-gpu
```

## Complete Example

Here's a full working example you can run:

```python
# my_simple_model.py
from chap_python_sdk import create_cli_app
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class SimpleConfig:
    """Configuration for a simple baseline model."""
    window_size: int = 7
    n_samples: int = 100

def train(config: SimpleConfig, data: pd.DataFrame) -> dict:
    """Train a simple moving average model."""
    return {
        "mean": data["value"].tail(config.window_size).mean(),
        "std": data["value"].tail(config.window_size).std(),
        "config": config
    }

def predict(
    model: dict,
    historic_data: pd.DataFrame,
    future_data: pd.DataFrame
) -> pd.DataFrame:
    """Generate predictions using the trained model."""
    n_future = len(future_data)
    n_samples = model["config"].n_samples

    # Generate samples from normal distribution
    samples = np.random.normal(
        loc=model["mean"],
        scale=model["std"],
        size=(n_future, n_samples)
    )

    return pd.DataFrame({"samples": [list(s) for s in samples]})

app = create_cli_app(
    train_func=train,
    predict_func=predict,
    config_class=SimpleConfig,
    name="simple-model"
)

if __name__ == "__main__":
    app()
```

**Usage**:

```bash
# Create sample data
echo "date,value" > train.csv
echo "2024-01-01,10" >> train.csv
echo "2024-01-02,12" >> train.csv
echo "2024-01-03,11" >> train.csv

echo "date" > future.csv
echo "2024-01-04" >> future.csv
echo "2024-01-05" >> future.csv

# Train the model
python my_simple_model.py train-cmd train.csv model.pkl --window-size 3

# Generate predictions
python my_simple_model.py predict-cmd model.pkl train.csv future.csv predictions.csv

# View predictions
cat predictions.csv
```

This produces predictions in wide format ready for `chap evaluate2`.
