# Adding Effects One at a Time

This tutorial starts from the simple model in [Simple Multistep](simple_multistep.md) and adds each modeling effect incrementally, showing how it improves predictions.

## Setup

```python notest
from chap_python_sdk import get_example_data
from chap_python_sdk.adaptors.multistep.one_step_model import ResidualBootstrapModel
from chap_python_sdk.adaptors.multistep.model import DataFrameMultistepModel
from chap_python_sdk.adaptors.multistep.config import MultistepConfig
import pandas as pd
import numpy as np

data = get_example_data("laos", "monthly")
training = data.training_data
future = data.future_data

index_cols = ["time_period", "location"]
target = "disease_cases"
```

## 1. Seasonal effects

Disease incidence follows seasonal patterns driven by climate. Without seasonal features, the model must infer seasonality entirely from lagged target values.

The `SeasonEncoder` extracts month from `time_period` and adds one-hot columns:

```python notest
from chap_python_sdk.adaptors.multistep.pipeline import SeasonEncoder

encoder = SeasonEncoder()

# Apply to training data
training_with_season = encoder.fit_transform(training)
print([c for c in training_with_season.columns if c.startswith("month_")])
# ['month_1', 'month_2', ..., 'month_12']
```

You can also use a custom season mapping for coarser groupings:

```python notest
# Map months to wet/dry seasons for Laos
laos_seasons = {
    1: "dry", 2: "dry", 3: "dry", 4: "dry",
    5: "wet", 6: "wet", 7: "wet", 8: "wet", 9: "wet",
    10: "wet", 11: "dry", 12: "dry",
}
encoder = SeasonEncoder(season_mapping=laos_seasons)
training_with_season = encoder.fit_transform(training)
print([c for c in training_with_season.columns if c.startswith("season_")])
# ['season_dry', 'season_wet']
```

**Why it helps**: the model can learn that case counts are typically higher during wet season without needing to infer this from lags alone.

## 2. Location effects

Different districts have different baseline case rates. The `LocationEncoder` adds one-hot columns:

```python notest
from chap_python_sdk.adaptors.multistep.pipeline import LocationEncoder

loc_encoder = LocationEncoder()
training_with_location = loc_encoder.fit_transform(training)

location_cols = [c for c in training_with_location.columns if c.startswith("location_")]
print(location_cols)
# ['location_Bokeo', 'location_Champasak', ...]
```

**Why it helps**: without location features, the model pools all districts and learns a single relationship. With location one-hots, it can learn district-specific intercepts.

## 3. Location x Season interactions

Some districts may have stronger seasonality than others. The `InteractionTransformer` creates cross-product features:

```python notest
from chap_python_sdk.adaptors.multistep.pipeline import InteractionTransformer

# First encode both location and season
season_encoder = SeasonEncoder(season_mapping=laos_seasons)
loc_encoder = LocationEncoder()

df = season_encoder.fit_transform(training)
df = loc_encoder.fit_transform(df)

# Then add interactions
interaction = InteractionTransformer(left_prefix="location_", right_prefix="season_")
df_with_interactions = interaction.fit_transform(df)

interaction_cols = [c for c in df_with_interactions.columns if "_x_" in c]
print(interaction_cols[:4])
# ['location_Bokeo_x_season_dry', 'location_Bokeo_x_season_wet', ...]
```

**Why it helps**: allows the model to learn that, for example, Champasak has a very strong wet-season effect while Bokeo does not.

## 4. Lagged exogenous features

Rainfall and temperature affect disease transmission, but often with a delay. The `FeatureLagger` adds lagged values of exogenous variables:

```python notest
from chap_python_sdk.adaptors.multistep.pipeline import FeatureLagger

lagger = FeatureLagger(n_lags=3, feature_cols=["rainfall", "mean_temperature"])
training_lagged = lagger.fit_transform(training)

lag_cols = lagger.lag_columns
print(lag_cols)
# ['rainfall_lag1', 'rainfall_lag2', 'rainfall_lag3',
#  'mean_temperature_lag1', 'mean_temperature_lag2', 'mean_temperature_lag3']
```

Lagging introduces NaN values for the first `n_lags` rows per location. These must be dropped before fitting:

```python notest
# Drop rows with NaN from lagging
training_clean = training_lagged.dropna(subset=lag_cols)
```

This is also available via the config:

```python notest
config = MultistepConfig(
    exogenous_variables=["rainfall", "mean_temperature"],
    n_feature_lags=3,
)
```

**Why it helps**: the model can learn that high rainfall 1-2 months ago leads to more cases now (mosquito breeding cycle).

## 5. Lagged target

The target lags (`n_target_lags`) are the most important features for recursive forecasting. They capture the autoregressive structure of the time series.

```python notest
# More lags capture longer memory
config = MultistepConfig(n_target_lags=12)  # Annual cycle

# Fewer lags for faster training, less memory
config = MultistepConfig(n_target_lags=3)   # Short-term only
```

The right number depends on the data:

- **Monthly data**: 12 lags captures a full year cycle
- **Weekly data**: 52 lags captures a full year, but 12-16 is often sufficient
- **Short series**: use fewer lags to avoid losing too many rows to the lag window

## 6. Log transform

Disease counts are non-negative and right-skewed. Log-transforming the target stabilizes variance and makes the distribution more symmetric:

```python notest
config = MultistepConfig(log_transform_target=True)
```

This applies `log(1 + y)` before fitting and `exp(x) - 1` to predictions. The `+1` handles zero counts.

**Why it helps**: without the transform, a model optimizing squared error focuses disproportionately on large outbreaks. The log transform makes the model equally precise across all scales.

## 7. Feature standardization

Standardizing covariates to zero mean and unit variance:

```python notest
config = MultistepConfig(standardize_covariates=True)
```

This is most important for linear models (Ridge, Lasso) where coefficient magnitudes are meaningful. Tree-based models (GradientBoosting, RandomForest) are scale-invariant and do not need this.

## 8. Rate transform (dividing by population)

Converting case counts to rates makes locations comparable:

```python notest
# Manual rate transform before training
training["disease_rate"] = training["disease_cases"] / training["population"]
```

This is not built into MultistepConfig but can be done as a preprocessing step. Use `disease_rate` as the target variable:

```python notest
config = MultistepConfig(target_variable="disease_rate")
```

**Why it helps**: a district with 100 cases out of 1,000 people is very different from 100 cases out of 1,000,000 people.

## Putting it all together

The CLI model configuration handles most of these effects:

```python notest
config = MultistepConfig(
    n_target_lags=12,
    n_samples=200,
    model_class="sklearn.ensemble.GradientBoostingRegressor",
    model_params={"n_estimators": 100, "max_depth": 3, "random_state": 42},
    exogenous_variables=["rainfall", "mean_temperature"],
    n_feature_lags=3,
    log_transform_target=True,
    standardize_target=False,
    standardize_covariates=True,
)
```

For the transformers (location, season, interaction), build a preprocessing pipeline:

```python notest
from sklearn.pipeline import Pipeline as SkPipeline

preprocessing = SkPipeline([
    ("season", SeasonEncoder(season_mapping=laos_seasons)),
    ("location", LocationEncoder()),
    ("interaction", InteractionTransformer(left_prefix="location_", right_prefix="season_")),
])

training_transformed = preprocessing.fit_transform(training)
```

## Next steps

- [Abstractions and Libraries](abstractions_and_libraries.md): how the multistep model maps to other time series frameworks
