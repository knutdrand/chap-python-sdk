# A Simple Multistep Model

This tutorial builds the simplest possible multistep model: a deterministic recursive forecaster with no lags, no exogenous features, and no transformations. We use the Laos EWARS dataset to show the basic workflow.

## Loading the data

```python notest
from chap_python_sdk import get_example_data

data = get_example_data("laos", "monthly")

training = data.training_data
historic = data.historic_data
future = data.future_data

print(f"Training: {len(training)} rows")
print(f"Locations: {training['location'].nunique()}")
print(f"Time range: {training['time_period'].min()} to {training['time_period'].max()}")
```

The training data has disease case counts for multiple Laos districts from 2000 to 2013, with monthly resolution.

## The simplest model

A deterministic multistep model uses a single sklearn regressor to make point predictions. At each forecast step, it feeds the previous prediction forward as input.

```python notest
from sklearn.linear_model import Ridge
from chap_python_sdk.adaptors.multistep_model import DeterministicMultistepModel
from chap_python_sdk.adaptors.multistep.model import DataFrameMultistepModel
import pandas as pd

# Prepare data: just time_period, location, and target
index_cols = ["time_period", "location"]
target = "disease_cases"

X_train = training[index_cols]
y_train = training[index_cols + [target]]
```

With `n_target_lags=1`, the model uses only the previous time step's value to predict the next:

```python notest
import numpy as np
from chap_python_sdk.adaptors.multistep_model import DeterministicMultistepModel

# Simple AR(1) model using Ridge regression
model = DeterministicMultistepModel(
    one_step_model=Ridge(),
    n_target_lags=1,
)

# For a single location
bokeo = training[training["location"] == "Bokeo"]
y = bokeo[target].to_numpy().astype(float)

model.fit(y)

# Predict 3 steps ahead
predictions = model.predict(y[-1:], n_steps=3)
print(f"Predictions: {predictions}")
```

## What this model does

1. **Fit**: builds a lag matrix where each row is `[y(t-1)]` and the target is `y(t)`. Fits Ridge regression on this.
2. **Predict step 1**: uses the last observed value as input, gets $\hat{y}_1$
3. **Predict step 2**: uses $\hat{y}_1$ as input, gets $\hat{y}_2$
4. **Predict step 3**: uses $\hat{y}_2$ as input, gets $\hat{y}_3$

This is *recursive* forecasting — errors compound at each step.

## Limitations

This model is deliberately minimal:

- **No seasonality**: it does not know that January is different from July
- **No location effects**: it does not distinguish between districts
- **No exogenous features**: rainfall and temperature are ignored
- **No uncertainty**: we get a single trajectory, not a distribution
- **Only 1 lag**: it cannot capture patterns longer than one month

The following tutorials add each of these features one at a time.

## Adding more lags

Increasing `n_target_lags` lets the model capture longer patterns:

```python notest
# AR(12) model — can learn annual cycles
model = DeterministicMultistepModel(
    one_step_model=Ridge(),
    n_target_lags=12,
)
model.fit(y)
predictions = model.predict(y[-12:], n_steps=3)
```

With 12 lags on monthly data, the model can learn that cases in January tend to be similar to cases in the previous January.

## Adding uncertainty with residual bootstrapping

To get prediction intervals instead of point predictions, use the full `DataFrameMultistepModel` with a `ResidualBootstrapModel`:

```python notest
from chap_python_sdk.adaptors.multistep.one_step_model import ResidualBootstrapModel
from chap_python_sdk.adaptors.multistep.model import DataFrameMultistepModel

one_step = ResidualBootstrapModel(
    "sklearn.linear_model.Ridge",
    {"alpha": 1.0},
)

df_model = DataFrameMultistepModel(
    one_step_model=one_step,
    n_target_lags=12,
)

df_model.fit(X_train, y_train)

# Get 200 sampled trajectories as a wide DataFrame
predictions = df_model.predict(
    y_historic=y_train,
    X_future=future[index_cols],
    n_steps=3,
    n_samples=200,
)
print(predictions.head())
# Columns: location, time_step, sample_0, sample_1, ..., sample_199
```

Each `sample_*` column is one possible future trajectory. The spread of these samples gives you prediction intervals.

## Next steps

- [Adding Effects](adding_effects.md): seasonal, location, lagged features, and transformations
- [Abstractions and Libraries](abstractions_and_libraries.md): how these concepts map to other time series libraries
