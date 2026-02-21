# CLI Adapter

Utilities for creating command-line interfaces from model train/predict functions.

## create_cli_app

::: chap_python_sdk.cli_adapter.create_cli_app
    options:
      show_source: true
      show_root_heading: false

## Notes

- **DataFrame Support**: Currently supports pandas DataFrames only. Data is loaded using `pd.read_csv()`.
- **Async Detection**: Automatically detects async functions using `asyncio.iscoroutinefunction()` and wraps them appropriately.
- **Output Format**: Converts nested `samples` column to wide format (`sample_0`, `sample_1`, ...) for CHAP `evaluate2` compatibility.
- **Advanced Use Cases**: For chapkit models with `run_info`/`geo` parameters or custom data loading, use a custom CLI with cyclopts directly (see `my-model/` example).

## Generated CLI Structure

The created app provides two commands:

### train-cmd

```
<app-name> train-cmd <train_data.csv> <output_model.pkl> [config options]
```

Trains a model and serializes it to a pickle file.

### predict-cmd

```
<app-name> predict-cmd <model.pkl> <historic.csv> <future.csv> <output.csv>
```

Loads a trained model and generates predictions in wide format.

## Example

```python
from chap_python_sdk import create_cli_app
from dataclasses import dataclass
import pandas as pd

@dataclass
class Config:
    alpha: float = 0.1

def train(config: Config, data: pd.DataFrame) -> dict:
    return {"alpha": config.alpha}

def predict(model: dict, historic: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"samples": [[1, 2, 3]]})

app = create_cli_app(train, predict, Config, "my-model")
```

See the [CLI guide](../user-guide/cli.md) for comprehensive documentation and examples.
