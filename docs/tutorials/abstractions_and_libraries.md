# Abstractions and Library Comparison

This tutorial explains the abstractions used by the multistep model and how they map to other time series libraries.

## The four layers

The multistep model is built from four composable layers:

```
┌─────────────────────────────────────────┐
│ 4. Feature Transformations              │
│    (sklearn pipelines)                  │
├─────────────────────────────────────────┤
│ 3. Multistep Recursive Model            │
│    (lag management + recursive predict) │
├─────────────────────────────────────────┤
│ 2. Uncertainty Wrapper                  │
│    (residual bootstrapping)             │
├─────────────────────────────────────────┤
│ 1. One-Step Regressor                   │
│    (any sklearn estimator)              │
└─────────────────────────────────────────┘
```

### Layer 1: One-step regressor

Any sklearn model that implements `fit(X, y)` and `predict(X)`. This is the atomic building block.

```python notest
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(n_estimators=100, max_depth=3)
```

The regressor knows nothing about time — it sees a flat feature matrix where each row is one training example.

### Layer 2: Uncertainty wrapper

`ResidualBootstrapModel` wraps a regressor to add non-parametric uncertainty. After fitting, it stores the training residuals. At prediction time, it resamples from these residuals to generate stochastic predictions.

```python notest
from chap_python_sdk.adaptors.multistep.one_step_model import ResidualBootstrapModel

one_step = ResidualBootstrapModel(
    model_class="sklearn.ensemble.GradientBoostingRegressor",
    model_params={"n_estimators": 100},
)
```

This implements the `OneStepModel` protocol:

- `fit(X, y)` — fits the regressor and stores residuals
- `predict_proba(X)` — returns a `Distribution` that can be sampled

### Layer 3: Multistep recursive model

`MultistepModel` handles the time dimension. It:

1. Builds lag matrices from the target series
2. Concatenates lags with exogenous features
3. Recursively generates multi-step forecasts by feeding each prediction back as input

```python notest
from chap_python_sdk.adaptors.multistep_model import MultistepModel

ms = MultistepModel(one_step_model=one_step, n_target_lags=12)
```

For deterministic forecasting (no uncertainty), use `DeterministicMultistepModel`.

For per-step feature selection (dropping unavailable lags), use `PerStepMultistepModel`.

### Layer 4: Feature transformations

Standard sklearn transformers handle preprocessing:

| Transformer | Purpose |
|---|---|
| `SeasonEncoder` | One-hot encode month/season from `time_period` |
| `LocationEncoder` | One-hot encode location |
| `InteractionTransformer` | Cross-product of location and season |
| `FeatureLagger` | Add lagged exogenous features |
| `build_target_pipeline()` | Log transform + standardize target |
| `build_feature_transformer()` | Standardize covariates |

All follow the sklearn `fit`/`transform` interface and can be composed in pipelines.

## How `DataFrameMultistepModel` ties it together

`DataFrameMultistepModel` is the high-level API that combines layers 1-3:

```python notest
from chap_python_sdk.adaptors.multistep.model import DataFrameMultistepModel

df_model = DataFrameMultistepModel(
    one_step_model=one_step,      # Layer 1+2
    n_target_lags=12,             # Layer 3
    target_pipeline=target_pipe,  # Layer 4 (target only)
)

df_model.fit(X_train, y_train)
predictions = df_model.predict(y_historic, X_future, n_steps=3, n_samples=200)
```

It handles:

- DataFrame-to-xarray conversion for multi-location data
- Target transformations (with inverse on predictions)
- Multi-location pooling (all locations trained in one model)

## Comparison with other libraries

### skforecast

[skforecast](https://skforecast.org/) is the closest analogue. It wraps sklearn regressors for recursive forecasting.

| Concept | This library | skforecast |
|---|---|---|
| One-step model | Any sklearn regressor | Any sklearn regressor |
| Recursive forecast | `MultistepModel` | `ForecasterAutoreg` |
| Multi-location | `fit_multi` / `predict_multi` | `ForecasterRecursiveMultiSeries` |
| Target lags | `n_target_lags` | `lags` parameter |
| Exogenous features | `exogenous_variables` | `exog` parameter |
| Uncertainty | Residual bootstrapping | Bootstrapped or conformal intervals |
| Feature transforms | sklearn pipelines | `transformer_series` / `transformer_exog` |

Key differences:

- skforecast has a richer API for model selection and backtesting
- This library focuses on the `chap evaluate2` pipeline and chapkit integration
- This library uses explicit residual resampling; skforecast supports multiple interval methods

### statsmodels

[statsmodels](https://www.statsmodels.org/) provides classical time series models (ARIMA, VAR, exponential smoothing).

| Concept | This library | statsmodels |
|---|---|---|
| Autoregression | Target lags as features | AR/ARIMA order parameter |
| Seasonality | `SeasonEncoder` or seasonal lags | SARIMA seasonal order |
| Exogenous | Feature columns | `exog` parameter in ARIMAX |
| Uncertainty | Residual bootstrap | Parametric (normal) intervals |
| Multi-location | Pooled regression | VAR (vector autoregression) |

Key differences:

- statsmodels uses parametric models with known likelihood functions
- This library is non-parametric — any sklearn regressor works
- statsmodels gives interpretable coefficients; tree-based models here do not
- statsmodels ARIMA handles differencing natively; here you would preprocess

### prophet

[Prophet](https://facebook.github.io/prophet/) by Meta decomposes time series into trend, seasonality, and holidays.

| Concept | This library | Prophet |
|---|---|---|
| Seasonality | `SeasonEncoder` (manual) | Fourier terms (automatic) |
| Trend | Captured by lags | Piecewise linear/logistic |
| Exogenous | Feature columns | `add_regressor()` |
| Uncertainty | Residual bootstrap | Bayesian (MAP + sampling) |
| Multi-location | Pooled model | Separate model per series |

Key differences:

- Prophet is designed for business forecasting with strong trends and holidays
- This library is designed for epidemiological data with location pooling
- Prophet fits one model per series; this library pools all locations

## Similar concepts in R

### forecast package

The R `forecast` package provides `auto.arima()`, `ets()`, and related functions.

```r
# R equivalent of our AR model with exogenous features
library(forecast)

model <- auto.arima(y, xreg = X_train)
predictions <- forecast(model, h = 3, xreg = X_future)
```

Mapping:

| This library | R forecast |
|---|---|
| `n_target_lags` | ARIMA `order` (p, d, q) |
| `ResidualBootstrapModel` | `forecast(..., bootstrap = TRUE)` |
| `MultistepModel` recursive predict | Built into `forecast()` |

### tidymodels + modeltime

The [modeltime](https://business-science.github.io/modeltime/) ecosystem brings sklearn-like flexibility to R:

```r
library(modeltime)
library(tidymodels)

# Similar to our approach: wrap any model for time series
model_spec <- boost_tree(trees = 100) %>%
  set_engine("xgboost") %>%
  set_mode("regression")

recipe <- recipe(disease_cases ~ ., data = training) %>%
  step_lag(disease_cases, lag = 1:12) %>%
  step_dummy(location) %>%
  step_date(time_period, features = "month")

workflow <- workflow() %>%
  add_model(model_spec) %>%
  add_recipe(recipe)

fit <- fit(workflow, data = training)
```

This is the closest R analogue to our approach:

| This library | tidymodels + modeltime |
|---|---|
| `SeasonEncoder` | `step_date(features = "month")` |
| `LocationEncoder` | `step_dummy(location)` |
| `FeatureLagger` | `step_lag()` |
| `build_target_pipeline()` | `step_log()`, `step_normalize()` |
| `InteractionTransformer` | `step_interact()` |
| `GradientBoostingRegressor` | `boost_tree() %>% set_engine("xgboost")` |

The key insight is the same in both ecosystems: time series forecasting can be framed as a supervised learning problem by engineering the right features (lags, seasons, locations) from the temporal structure.
