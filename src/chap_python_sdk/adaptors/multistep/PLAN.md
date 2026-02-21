# Multistep Model Enhancement Plan

## Overview

This plan implements the features outlined in `design2.md`: new sklearn-compatible transformers, a deterministic multistep model variant, and a tutorial series using the Laos EWARS dataset.

---

## Current State

The multistep module already has:
- `MultistepModel` / `DataFrameMultistepModel` for recursive multi-step forecasting
- `ResidualBootstrapModel` for uncertainty via residual resampling
- `FeatureLagger` for lagged exogenous features
- `build_target_pipeline()` / `build_feature_transformer()` for sklearn pipelines
- Multi-location pooling via xarray
- CLI and chapkit adaptor integration
- Test suite across all layers

---

## Work Items

### WI-1: Multistep model outputs pd.DataFrame

**Goal:** Ensure the multistep model can return predictions as a `pd.DataFrame` directly (not just xarray).

**Context:** `DataFrameMultistepModel.predict()` currently returns `xr.DataArray`. Add a method or option to return a DataFrame with columns `(location, time_period, sample_0..sample_n)` or long format.

**Files to modify:**
- `src/chap_python_sdk/adaptors/multistep/model.py`

**Tests:**
- `tests/adaptors/multistep/test_model.py` — add test for DataFrame output

---

### WI-2: Location one-hot encoder transformer

**Goal:** Create an sklearn-compatible transformer that adds one-hot encoded location columns to the feature DataFrame.

**Details:**
- Subclass `BaseEstimator, TransformerMixin`
- `fit()`: learn unique locations from a `location` column
- `transform()`: add one-hot columns (e.g. `location_VientianePrefecture`, `location_Savannakhet`), drop original `location` column
- Handle unseen locations gracefully (zeros or error)

**Files to create/modify:**
- `src/chap_python_sdk/adaptors/multistep/pipeline.py` — add `LocationEncoder` class

**Tests:**
- `tests/adaptors/multistep/test_pipeline.py` — add `TestLocationEncoder`

---

### WI-3: Season one-hot encoder transformer

**Goal:** Create an sklearn-compatible transformer that adds seasonal features from `time_period`.

**Details:**
- Subclass `BaseEstimator, TransformerMixin`
- `fit()`: learn unique seasons (extract month from `time_period`, map to season)
- `transform()`: add one-hot season columns (e.g. `season_dry`, `season_wet`, or month-based)
- Constructor parameter to choose encoding scheme (month, quarter, custom season mapping)

**Files to create/modify:**
- `src/chap_python_sdk/adaptors/multistep/pipeline.py` — add `SeasonEncoder` class

**Tests:**
- `tests/adaptors/multistep/test_pipeline.py` — add `TestSeasonEncoder`

---

### WI-4: Location x Season interaction transformer

**Goal:** Create an sklearn-compatible transformer that creates interaction features between location and season.

**Details:**
- Subclass `BaseEstimator, TransformerMixin`
- Expects location and season columns already present (can be one-hot or categorical)
- Creates interaction terms (e.g. `loc_Vientiane_x_season_dry`)
- Can be chained after WI-2 and WI-3 in a pipeline

**Dependencies:** WI-2, WI-3 (conceptually, but can be implemented independently using test data)

**Files to create/modify:**
- `src/chap_python_sdk/adaptors/multistep/pipeline.py` — add `InteractionTransformer` class

**Tests:**
- `tests/adaptors/multistep/test_pipeline.py` — add `TestInteractionTransformer`

---

### WI-5: Per-step feature lag removal model

**Goal:** Create a multistep model variant that trains `n_steps` separate models, each removing lagged features that would not be available at that forecast horizon.

**Details:**
- At step `k`, features with lag < `k` are unavailable and should be dropped
- Takes a callback `get_lag_idx(column_name) -> int | None` that returns the lag index for a column (or `None` if not a lagged column)
- At step `k`, columns where `get_lag_idx(col) < k` are removed
- Each of the `n_steps` sub-models is fitted on the appropriately reduced feature set

**Files to create/modify:**
- `src/chap_python_sdk/adaptors/multistep/model.py` or new file `per_step_model.py`

**Tests:**
- New test file or extend `test_model.py`

---

### WI-6: Deterministic recursive multistep model

**Goal:** Create a multistep model variant that recursively predicts only the point estimate (mean/median/mode) without sampling trajectories.

**Details:**
- Similar to `MultistepModel` but without residual bootstrapping
- Each step feeds the point prediction forward as input to the next step
- Faster than bootstrap version, useful when uncertainty is not needed
- Should still support multi-location pooling

**Files to create/modify:**
- `src/chap_python_sdk/adaptors/multistep_model.py` — add `DeterministicMultistepModel` or a `deterministic=True` flag

**Tests:**
- `tests/test_multistep_model.py` — add deterministic model tests

---

### WI-7: Tutorial — Linear regression assumptions

**Goal:** Tutorial page explaining linear regression assumptions, how they break in time series, and mitigations.

**Details (three sections):**
1. What are the assumptions of Linear Regression (independence, homoscedasticity, normality, linearity)
2. How they are broken in our time series models (autocorrelation, heteroscedasticity, non-stationarity, non-linear effects)
3. How we alleviate each broken assumption (lagged features, log transforms, differencing, seasonal features, etc.)

**Files to create:**
- `docs/tutorials/linear_regression_assumptions.md`

**Visualization assets (Altair):**
- Residual plots, ACF plots, QQ plots using the Laos dataset
- Save to `docs/images/`

---

### WI-8: Tutorial — Simple multistep model (no lags, deterministic)

**Goal:** Tutorial showing the simplest multistep model using the Laos EWARS dataset.

**Details:**
- Load the Laos dataset via `get_example_data("laos", "monthly")`
- Fit a simple linear regression with no lags, deterministic predictions
- Show predictions vs actuals
- Discuss limitations (no seasonality, no location effects, no lags)

**Dependencies:** WI-6 (deterministic model)

**Files to create:**
- `docs/tutorials/simple_multistep.md`

---

### WI-9: Tutorial — Adding effects one at a time

**Goal:** Tutorial series showing how adding each effect improves the model.

**Sections (each builds on previous):**
1. Seasonal effects (categorical month/season encoding) — uses WI-3
2. Location effects (one-hot location) — uses WI-2
3. Lagged exogenous features — uses existing `FeatureLagger`
4. Lagged target — uses existing `MultistepModel` with `n_target_lags`
5. Log transform — uses existing `build_target_pipeline(log=True)`
6. Feature standardization — uses existing `build_feature_transformer()`
7. Rate transform (divide by population) — new preprocessing step

**Dependencies:** WI-2, WI-3, WI-6

**Files to create:**
- `docs/tutorials/adding_effects.md`

---

### WI-10: Tutorial — Abstractions and library comparison

**Goal:** Tutorial showing the abstractions used and how they map to other time series libraries.

**Sections:**
1. Sklearn one-step model abstraction
2. skpro wrapper for uncertainty
3. Multistep model recursive structure
4. Feature transformations as sklearn pipelines
5. How these concepts appear in other time series libraries (skforecast, statsmodels, etc.)
6. A similar model framework in R (forecast package, tidymodels)

**Files to create:**
- `docs/tutorials/abstractions_and_libraries.md`

---

## Suggested Execution Order

```
Phase 1 — Core Components (parallelizable):
  WI-1  DataFrame output
  WI-2  LocationEncoder
  WI-3  SeasonEncoder
  WI-6  Deterministic multistep model

Phase 2 — Dependent Components:
  WI-4  Interaction transformer (after WI-2, WI-3)
  WI-5  Per-step feature lag removal

Phase 3 — Tutorials (after Phase 1-2):
  WI-7  Linear regression assumptions
  WI-8  Simple multistep (after WI-6)
  WI-9  Adding effects (after WI-2, WI-3, WI-6)
  WI-10 Abstractions and libraries
```

## Notes

- All transformers must follow sklearn's `BaseEstimator` / `TransformerMixin` API
- All new code needs tests — run `make test` and `make lint` after each WI
- Use the Laos EWARS dataset (`get_example_data("laos", "monthly")`) for all tutorials
- Tutorials should use Altair for visualizations, saved to `docs/images/`
- Animations (if any) should use Manim
