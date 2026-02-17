# MultistepModel CLI

## Overview

The MultistepModel CLI provides a ready-to-use command-line model based on the recursive trajectory sampler. It wraps any scikit-learn regressor with residual bootstrapping to produce multi-step probabilistic forecasts, and exposes the workflow as `train-cmd` / `predict-cmd` CLI subcommands compatible with `chap evaluate2`.

No async code or chapkit DataFrames are needed — the CLI model works entirely with pandas DataFrames and CSV files.

## Quick Start

### Minimal `model.py`

```python notest
from chap_python_sdk import create_multistep_cli_app

app = create_multistep_cli_app()

if __name__ == "__main__":
    app()
```

That's it. This gives you a fully functional CLI model with sensible defaults (GradientBoostingRegressor, 12 target lags, 200 prediction samples).

### Running the CLI

```bash
# Train
python model.py train-cmd train_data.csv model.pkl

# Predict
python model.py predict-cmd model.pkl historic.csv future.csv predictions.csv
```

## Data Format

### Training / Historic Data

CSV in long format with at minimum these columns:

| Column | Type | Description |
|--------|------|-------------|
| `time_period` | date string | Time period (e.g. `2020-01-01`) |
| `location` | string | Location identifier (e.g. `Bokeo`) |
| `disease_cases` | float | Target variable |

Example:

```csv
time_period,location,disease_cases
2020-01-01,Bokeo,45.0
2020-01-01,Luang,32.0
2020-02-01,Bokeo,52.0
2020-02-01,Luang,38.0
```

Additional exogenous columns (e.g. `rainfall`, `mean_temperature`) can be included — see [Using Exogenous Variables](#using-exogenous-variables).

### Future Data

CSV with `time_period` and `location` columns (no target column). If using exogenous variables, include those columns too.

```csv
time_period,location,rainfall,mean_temperature
2022-01-01,Bokeo,85.2,28.1
2022-01-01,Luang,72.4,26.3
2022-02-01,Bokeo,91.0,27.5
2022-02-01,Luang,68.9,25.8
```

### Output Predictions

The CLI writes predictions in wide format (`sample_0`, `sample_1`, ...) compatible with `chap evaluate2`:

```csv
time_period,location,sample_0,sample_1,...,sample_199
2022-01-01T00:00:00,Bokeo,42.1,51.3,...,47.8
2022-01-01T00:00:00,Luang,30.2,35.6,...,33.1
```

## Configuration

### Default Configuration

When you call `create_multistep_cli_app()` without arguments, it uses `MultistepConfig` with these defaults:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_target_lags` | `12` | Number of lagged target values used as features |
| `n_samples` | `200` | Number of sampled trajectories per prediction |
| `model_class` | `sklearn.ensemble.GradientBoostingRegressor` | Scikit-learn regressor class |
| `model_params` | `{"n_estimators": 100, "max_depth": 3, ...}` | Parameters passed to the regressor |
| `exogenous_variables` | `None` | List of exogenous column names |
| `target_variable` | `disease_cases` | Name of the target column |

### Custom Configuration

Pass a `MultistepConfig` to customize:

```python notest
from chap_python_sdk import MultistepConfig, create_multistep_cli_app

config = MultistepConfig(
    n_target_lags=6,
    n_samples=500,
    model_class="sklearn.ensemble.RandomForestRegressor",
    model_params={
        "n_estimators": 200,
        "max_depth": 5,
        "random_state": 42,
    },
    target_variable="disease_cases",
    exogenous_variables=["rainfall", "mean_temperature"],
)

app = create_multistep_cli_app(config)

if __name__ == "__main__":
    app()
```

### Using a Different Regressor

Any scikit-learn regressor that implements `fit(X, y)` and `predict(X)` can be used. Specify it as a dotted import path:

```python notest
config = MultistepConfig(
    model_class="sklearn.linear_model.Ridge",
    model_params={"alpha": 1.0},
)
```

Other examples:
- `sklearn.ensemble.RandomForestRegressor`
- `sklearn.ensemble.HistGradientBoostingRegressor`
- `sklearn.linear_model.Lasso`
- `sklearn.neighbors.KNeighborsRegressor`

## Using Exogenous Variables

To incorporate climate or other covariates:

1. Include the columns in your training data CSV
2. Include the same columns in your future data CSV
3. List them in the config

```python notest
config = MultistepConfig(
    exogenous_variables=["rainfall", "mean_temperature"],
)

app = create_multistep_cli_app(config)
```

The model will use these variables as additional features alongside the lagged target values at each forecast step.

## How It Works

### Training (`train-cmd`)

1. Reads the CSV into a pandas DataFrame
2. Pivots long format to a 2D xarray DataArray `(location, time)`
3. Creates a `ResidualBootstrapModel` wrapping the configured sklearn regressor
4. Wraps it in a `MultistepModel` with the specified number of target lags
5. Calls `fit_multi(y, X)` which pools all locations into one training set — each time step becomes a training row with `[exogenous..., y(t-n), ..., y(t-1)]` as features and `y(t)` as target
6. Saves the trained model dict (model object, locations, config) as a pickle file

### Prediction (`predict-cmd`)

1. Loads the pickled model dict
2. Converts historic data to xarray, takes the last `n_target_lags` time steps as the lag window
3. Converts future data to xarray for exogenous features
4. For each location, runs recursive trajectory sampling:
   - At each step, builds a feature vector from the lag window (+ exogenous if available)
   - Predicts using the one-step model
   - Adds resampled training residuals to create stochastic samples
   - Rolls the lag window forward with the sampled value
   - Repeats for `n_steps`
5. Converts the `(location, trajectory, step)` predictions to wide-format CSV

### Architecture

```
MultistepConfig           Configuration (regressor class, lags, samples, etc.)
    |
    v
ResidualBootstrapModel    Wraps any sklearn regressor + stores training residuals
    |
    v
MultistepModel            Recursive multi-step forecaster using lag features
    |
    v
create_multistep_cli_app  Wires train/predict into a cyclopts CLI via create_cli_app
```

## Using train/predict Directly

You can also use the `train` and `predict` functions programmatically without the CLI:

```python notest
import pandas as pd
from chap_python_sdk.adaptors.multistep.cli_model import train, predict
from chap_python_sdk.adaptors.multistep.config import MultistepConfig

config = MultistepConfig(n_target_lags=6, n_samples=100)

# Train
train_data = pd.read_csv("train_data.csv")
model = train(config, train_data)

# Predict
historic = pd.read_csv("historic.csv")
future = pd.read_csv("future.csv")
predictions = predict(config, model, historic, future)

# predictions is a DataFrame with columns: time_period, location, samples
# where samples is a list of floats per row
print(predictions.head())
```

## Data Requirements

- **Minimum time points**: Your training data must have more time points than `n_target_lags` per location. With the default `n_target_lags=12`, you need at least 13 monthly observations per location.
- **Consistent locations**: All locations must appear at all time points (balanced panel).
- **No duplicates**: Each `(time_period, location)` combination must be unique.
- **Numeric target**: The target column must contain numeric values.

## Integration with `chap evaluate2`

The output format is directly compatible with CHAP's evaluation pipeline:

```bash
# Train and predict
python model.py train-cmd train.csv model.pkl
python model.py predict-cmd model.pkl historic.csv future.csv predictions.csv

# Evaluate
chap evaluate2 --predictions predictions.csv --actuals actuals.csv
```

## Next Steps

- [Building CLI Interfaces](cli.md) — General `create_cli_app` documentation
- [Model Testing](model-testing.md) — Testing models with the SDK
- [Prediction Formats](prediction-formats.md) — Format conversion utilities
