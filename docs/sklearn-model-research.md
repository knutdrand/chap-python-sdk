# SKLearnModel Research: Implementing Time Series Forecasting with Generic Models

This document summarizes research on implementing darts-like SKLearnModel functionality for chap-python-sdk, including probabilistic forecasting capabilities.

## Table of Contents

- [Executive Summary](#executive-summary)
- [What Darts SKLearnModel Provides](#what-darts-sklearnmodel-provides)
- [How Lagged Features Work](#how-lagged-features-work)
- [Implementation Details from Darts](#implementation-details-from-darts)
- [Recursive vs One-Shot Prediction](#recursive-vs-one-shot-prediction)
- [Probabilistic Forecasting](#probabilistic-forecasting)
- [Library Comparison](#library-comparison)
- [Recommendations](#recommendations)

## Executive Summary

**Goal:** Understand how to implement SKLearnModel-like functionality that wraps any sklearn regressor for time series forecasting.

**Key Findings:**
- Darts SKLearnModel provides automatic lag feature engineering but NOT true probabilistic recursion
- Core implementation requires ~200-500 lines for basic functionality
- True probabilistic recursive forecasting requires sampling at each step (darts doesn't do this)
- **skforecast** provides the best sklearn wrapper with true probabilistic support
- Implementation effort: 2-4 weeks for full featured version, 5-7 days for MVP

**Recommendation:** Use skforecast as reference implementation or provide minimal utilities and let users choose their framework.

---

## What Darts SKLearnModel Provides

### Core Functionality

Darts' `SKLearnModel` (and `RegressionModel`) wraps any scikit-learn compatible regressor to transform time series forecasting into supervised learning:

```python notest
from darts.models import RegressionModel
from sklearn.ensemble import RandomForestRegressor

model = RegressionModel(
    lags=12,                      # Use last 12 values of target
    lags_past_covariates=12,      # Use last 12 values of covariates
    lags_future_covariates=(0,6), # Use 6 future covariate values
    output_chunk_length=1,         # Predict 1 step at a time
    model=RandomForestRegressor(n_estimators=100)
)

# Automatic feature engineering happens here
model.fit(target_series, past_covariates=covariate_series)

# Automatic recursive forecasting happens here
predictions = model.predict(n=3, series=target_series, past_covariates=covariates)
```

### What Happens Under the Hood

1. **Training Phase:**
   - Calls `create_lagged_training_data()` to build feature matrix X and labels y
   - X contains: [12 target lags, 12 × n_covariates covariate lags, future covariate lags]
   - Fits sklearn model: `sklearn_model.fit(X, y)`
   - Stores lag configuration

2. **Prediction Phase:**
   - Extracts most recent lags from input series
   - Makes prediction for next step
   - If n > output_chunk_length, recursively feeds predictions back as features
   - Returns TimeSeries with predictions

### Key Parameters

- **lags**: Target series lags
  - Integer: `lags=3` → uses [-1, -2, -3]
  - List: `lags=[-1, -3, -5]` → specific lags
  - Dict: `lags={'component_A': 3, 'default_lags': 2}` → component-specific

- **lags_past_covariates**: Historical covariate lags (must be < 0)
- **lags_future_covariates**: Future covariate lags (can be ≥ 0)
- **output_chunk_length**: Number of steps predicted per forward pass
- **multi_models**: If True, train separate model per horizon

---

## How Lagged Features Work

### Conceptual Overview

Transform time series into supervised learning by creating features from past values:

```
Original series (temperature):
Time:  0    1    2    3    4    5    6
Temp: 20   22   21   23   25   24   26

With lags=[-3, -2, -1], output_chunk_length=1:

Training data (X, y):
t  | lag-3  lag-2  lag-1 | target
3  |  20     22     21   |   23
4  |  22     21     23   |   25
5  |  21     23     25   |   24
6  |  23     25     24   |   26

First 3 timesteps lost (need history for lags)
```

### Feature Matrix Structure

For `lags=[-2, -1]` with 2 components and 3 covariates:

```
X shape: (n_observations, n_features, n_samples)

Features ordered as:
[comp0_lag-2, comp1_lag-2, comp0_lag-1, comp1_lag-1,
 cov0_lag-2, cov1_lag-2, cov2_lag-2, cov0_lag-1, cov1_lag-1, cov2_lag-1,
 static_cov0, static_cov1]
```

### Label Matrix Structure

For `output_chunk_length=3` with 2 components:

```
y shape: (n_observations, output_chunk_length * n_components)

Labels ordered as:
[comp0_t+1, comp1_t+1, comp0_t+2, comp1_t+2, comp0_t+3, comp1_t+3]
```

---

## Implementation Details from Darts

### Tabularization Module

Darts implements lag feature creation in `darts/utils/data/tabularization/tabularization.py`:

**Key Functions:**

1. **`create_lagged_training_data()`**: Creates X and y for training
2. **`create_lagged_prediction_data()`**: Creates X for inference
3. **`create_lagged_data()`**: Core implementation (called by above)

**Two Algorithmic Approaches:**

1. **Moving Window Method** (same frequency):
   - Uses `strided_moving_window()` for efficient extraction
   - Extracts windows: [t - max_lag, ..., t - min_lag]
   - Faster, more memory intensive
   - Optimized via NumPy stride tricks

2. **Time Intersection Method** (mixed frequency):
   - Finds common timestamps across series
   - Offsets indices by lag values
   - More flexible, slower

**Performance:**
- Original implementation optimized in PR #1399
- Eliminated loops over lag values → vectorized operations
- 10x speedup for simple cases
- 40x speedup for complex cases (10+ lags)
- 200x speedup for mixed-frequency data

### Example Implementation (Simplified)

```python notest
def create_lagged_features(df, target_col, lags, covariate_cols=None):
    """Create sliding window features manually."""
    features = []
    labels = []

    max_lag = max(abs(lag) for lag in lags)

    for i in range(max_lag, len(df)):
        # Target lags
        row_features = []
        for lag in sorted(lags, reverse=True):  # [-3, -2, -1]
            idx = i + lag  # i-3, i-2, i-1
            row_features.append(df[target_col].iloc[idx])

        # Covariate lags
        if covariate_cols:
            for col in covariate_cols:
                for lag in sorted(lags, reverse=True):
                    idx = i + lag
                    row_features.append(df[col].iloc[idx])

        features.append(row_features)
        labels.append(df[target_col].iloc[i])

    return np.array(features), np.array(labels)
```

---

## Recursive vs One-Shot Prediction

### Recursive Prediction (output_chunk_length=1)

Predict one step at a time, feeding predictions back as inputs:

```python notest
# Example: Predict 3 steps with lags=3
history = [20, 22, 21]

# Step 1
features_1 = [21, 22, 20]  # lags: -1, -2, -3
pred_1 = model.predict(features_1)  # → 23

# Step 2 (uses pred_1)
features_2 = [23, 21, 22]  # pred_1 becomes lag-1
pred_2 = model.predict(features_2)  # → 25

# Step 3 (uses pred_1 and pred_2)
features_3 = [25, 23, 21]
pred_3 = model.predict(features_3)  # → 24

result = [23, 25, 24]
```

**Pros:**
- Flexible prediction horizon
- Can predict any number of steps

**Cons:**
- Error accumulation (predictions use predictions)
- Slower for long horizons

### One-Shot Prediction (output_chunk_length=3)

Predict all steps in single forward pass:

```python notest
# Model predicts 3 steps simultaneously
features = [21, 22, 20]
predictions = model.predict(features)  # → [23, 25, 24]
```

**Pros:**
- Faster
- Less error accumulation
- Can learn multi-step dependencies

**Cons:**
- Fixed prediction horizon
- Need to retrain for different horizons

### Multi-Models Strategy

**`multi_models=True` (default):**
- Train separate model for each forecast horizon
- Model 1: predict t+1 using lags [-1, -2, -3]
- Model 2: predict t+2 using lags [-2, -3, -4]
- Model 3: predict t+3 using lags [-3, -4, -5]

**`multi_models=False`:**
- Single model outputs vector of all steps
- Features shift back by (output_chunk_length - n) for step n

---

## Probabilistic Forecasting

### What Darts Provides (Limited)

Darts supports quantile regression but **NOT true probabilistic recursion**:

```python notest
model = RegressionModel(
    lags=12,
    output_chunk_length=1,
    likelihood="quantile",
    quantiles=[0.1, 0.5, 0.9]
)

predictions = model.predict(n=3, num_samples=100)
```

**Problem:**
- Recursively predicts using median (50th percentile)
- Generates samples around that single deterministic trajectory
- Does NOT sample different trajectories at each recursive step

### True Probabilistic Recursion

Generate multiple trajectories by sampling at each step:

```python notest
def probabilistic_recursive_forecast(
    model, history, n_steps, n_samples, dispersion
):
    """Generate n_samples probabilistic trajectories."""
    trajectories = []

    for sample_idx in range(n_samples):
        trajectory = []
        current_history = history.copy()

        for step in range(n_steps):
            # Step 1: Predict mean
            pred_mean = model.predict(current_history)

            # Step 2: Sample from distribution
            sampled_value = np.random.negative_binomial(
                n=dispersion,
                p=dispersion/(dispersion + pred_mean)
            )

            # Step 3: Feed SAMPLED value back (not mean!)
            trajectory.append(sampled_value)
            current_history = np.append(current_history, sampled_value)

        trajectories.append(trajectory)

    return trajectories  # (n_samples, n_steps)
```

**Key difference:** Each trajectory branches differently because randomness is injected at each recursive step.

### Why This Matters: Fan Charts

**Deterministic recursion (darts default):**
```
Narrow intervals - just uncertainty around one path
 30 ┤     ╱╲
 25 ┤    ╱──╲
 20 ┤   ╱────╲
 15 ┤  ╱──────╲
```

**Probabilistic trajectories (true sampling):**
```
Widening intervals - uncertainty compounds
 40 ┤     ╱─────────  ← High trajectories
 30 ┤    ╱═════════   ← Median
 20 ┤   ╱─────────    ← Low trajectories
 10 ┤  ╱
    └──────────────
     narrow → wide
```

---

## Library Comparison

### Libraries Supporting True Probabilistic Recursion

| Library | Sklearn Support | Probabilistic Method | Recursive Sampling | Ease of Use | Best For |
|---------|----------------|----------------------|-------------------|-------------|----------|
| **skforecast** | ✅ Any sklearn | Bootstrapping, Conformal, Quantile | ✅ Yes | ⭐⭐⭐⭐⭐ | **BEST** |
| **sktime** | ✅ Any sklearn | Bootstrapping, Conformal, skpro | ✅ Yes | ⭐⭐⭐ | Comprehensive |
| **GluonTS** | ❌ Own models | Distribution sampling | ✅ Yes | ⭐⭐⭐ | Deep learning |
| **PyMC** | ⚠️ Manual wrap | Full Bayesian MCMC | ✅ Yes | ⭐⭐ | Research |
| **pytorch-forecasting** | ❌ Own models | Quantile/distribution | ✅ Yes | ⭐⭐⭐ | Deep learning |
| **Prophet** | ❌ Own model | Monte Carlo trends | ⚠️ Partial | ⭐⭐⭐⭐ | Business |
| **Darts** | ✅ Any sklearn | Quantile (no sampling) | ❌ No | ⭐⭐⭐⭐⭐ | Prototyping |
| **MLForecast** | ✅ Any sklearn | ❌ Limited | ❌ No | ⭐⭐⭐⭐ | Point prediction |

### Detailed Library Analysis

#### 1. skforecast - RECOMMENDED

**Full sklearn wrapper with true probabilistic forecasting.**

```python notest
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.ensemble import RandomForestRegressor

forecaster = ForecasterAutoreg(
    regressor=RandomForestRegressor(n_estimators=100),
    lags=12
)

forecaster.fit(y=train_data)

# Get 500 probabilistic trajectories
samples = forecaster.predict_bootstrapping(steps=3, n_boot=500)
# Returns: (3 steps, 500 samples)
```

**Probabilistic methods:**
1. **Bootstrapping** - Samples from residuals at each step
2. **Conformal Prediction** - Distribution-free calibrated intervals
3. **Quantile Regression** - Direct quantile estimation

**Pros:**
- Lightweight (~5MB vs darts ~100MB)
- Works with any sklearn model
- True recursive sampling
- Excellent documentation

**Cons:**
- Simpler than darts (fewer bells and whistles)
- No TimeSeries abstraction

#### 2. sktime - COMPREHENSIVE

**Full time series framework with probabilistic support.**

```python notest
from sktime.forecasting.compose import make_reduction, BaggingForecaster
from sktime.transformations.bootstrap import TSBootstrapAdapter

base_forecaster = make_reduction(
    RandomForestRegressor(),
    strategy="recursive",
    window_length=12
)

prob_forecaster = BaggingForecaster(
    forecaster=base_forecaster,
    n_estimators=100,
    bootstrap_transformer=TSBootstrapAdapter(sp=12)
)

predictions = prob_forecaster.predict_quantiles(
    fh=[1,2,3],
    alpha=[0.1, 0.5, 0.9]
)
```

**Pros:**
- Very comprehensive
- Multiple probabilistic methods
- Active development

**Cons:**
- Complex API
- Steeper learning curve

#### 3. GluonTS - DEEP LEARNING FOCUS

**Primarily for deep learning models with probabilistic support.**

```python notest
from gluonts.model.deepar import DeepAREstimator

estimator = DeepAREstimator(
    freq="M",
    prediction_length=12,
    num_samples=100
)

predictor = estimator.train(training_data)
forecast = predictor.predict(test_data)

# forecast.samples: (100, 12) - true trajectories
```

**Pros:**
- Best probabilistic deep learning models
- Production-ready (AWS)
- True trajectory sampling

**Cons:**
- No sklearn wrapper
- Requires more data
- More complex setup

#### 4. PyMC - BAYESIAN INFERENCE

**Full Bayesian approach with MCMC sampling.**

```python notest
import pymc as pm

with pm.Model() as model:
    rho = pm.Normal("rho", mu=0, sigma=1, shape=3)
    sigma = pm.HalfNormal("sigma", sigma=1)
    y = pm.AR("y", rho=rho, sigma=sigma, observed=data)
    trace = pm.sample(2000)

with model:
    future_y = pm.AR("future_y", rho=rho, sigma=sigma, shape=12)
    posterior_predictive = pm.sample_posterior_predictive(trace)
```

**Pros:**
- True Bayesian inference
- Full uncertainty quantification
- Great for research

**Cons:**
- Slow (MCMC sampling)
- Requires Bayesian expertise
- Not sklearn-compatible

---

## Recommendations

### For chap-python-sdk

**Three Options:**

#### Option 1: Recommend skforecast (Minimal Effort)
- Add skforecast to documentation as recommended approach
- Provide example integration code
- Let users handle implementation

**Pros:**
- Zero implementation effort
- Users get mature, tested library
- Full probabilistic support

**Cons:**
- Another dependency for users
- Less control over API

#### Option 2: Build Minimal SKLearnModel Wrapper (Medium Effort)

Implement core functionality inspired by skforecast:

```python notest
from chap_python_sdk.models import SKLearnForecaster

class SKLearnForecaster:
    """Minimal sklearn wrapper for time series forecasting."""

    def __init__(
        self,
        model,
        lags: int | list[int] = 12,
        n_samples: int = 100,
    ):
        self.model = model
        self.lags = lags
        self.n_samples = n_samples

    async def train(self, config, data, run_info, geo):
        """Train with automatic lag feature engineering."""
        X, y = self._create_lagged_features(data)
        self.model.fit(X, y)
        self._estimate_residuals(X, y)
        return {"model": self.model, "residuals": self.residuals}

    async def predict(self, config, model, historic, future, run_info, geo):
        """Predict with bootstrapped uncertainty."""
        return self._predict_bootstrapping(
            model, historic, future, self.n_samples
        )
```

**Implementation effort:** 5-7 days
- Lag feature extraction: 2 days
- Training/prediction: 2 days
- Bootstrap sampling: 2 days
- Testing: 1 day

**Pros:**
- Users get simple API
- No heavy dependencies
- Full control over implementation

**Cons:**
- Maintenance burden
- Won't match skforecast's features initially

#### Option 3: Utilities Only (Lightweight)

Provide building blocks, let users compose:

```python notest
from chap_python_sdk.utils import create_lagged_features, bootstrap_forecast

# User implements their own
X_train, y_train = create_lagged_features(data, lags=12)
model.fit(X_train, y_train)

samples = bootstrap_forecast(
    model,
    history=data,
    n_steps=3,
    n_samples=100,
    lags=12
)
```

**Implementation effort:** 3-5 days

**Pros:**
- Minimal implementation
- Maximum flexibility
- Low maintenance

**Cons:**
- Users do more work
- Less opinionated

### Recommended Approach: Option 1 + Option 3

1. **Document skforecast** as recommended solution for production
2. **Provide utility functions** for users who want custom implementations
3. **Add examples** showing both approaches

This gives users:
- Quick start (use skforecast)
- Flexibility (use utilities)
- Education (see how it works)

---

## Case Study: automatic-model

The automatic-model project uses darts' RegressionModel wrapper:

```python notest
from darts.models import RegressionModel
from sklearn.ensemble import RandomForestRegressor

model = RegressionModel(
    lags=12,
    lags_past_covariates=12,
    output_chunk_length=1,
    model=RandomForestRegressor(n_estimators=100, max_depth=10)
)

model.fit(target_series, past_covariates=covariate_series)
predictions = model.predict(n=n_periods, series=combined_target, ...)
```

**What darts provides:**
- Automatic lag feature engineering (saved ~200 lines)
- TimeSeries abstraction for date handling
- Automatic recursive forecasting
- Easy sklearn model swapping

**What it doesn't provide:**
- True probabilistic trajectories (they manually add uncertainty afterwards)
- Lightweight dependency (darts is ~100MB)

**Could be improved by:**
- Switching to skforecast for true probabilistic recursion
- Or manually implementing probabilistic sampling loop
- This would give proper fan charts with widening intervals

---

## Implementation Estimates

### MVP (Basic SKLearnModel-like wrapper)

**Features:**
- Basic lag feature engineering
- Single-step recursive prediction
- Works with any sklearn model
- No covariates initially

**Effort:** 5-7 days
- Lag extraction: 2 days
- Recursive prediction: 2 days
- Integration: 1 day
- Testing: 2 days

### Full Featured Version

**Features:**
- Multiple lag specifications (int, list, dict)
- Covariate support (past/future)
- Multi-step prediction strategies
- Probabilistic via bootstrapping
- Conformal prediction intervals

**Effort:** 2-4 weeks
- Core lag engineering: 3-5 days
- Covariate handling: 2-3 days
- Probabilistic sampling: 3-5 days
- Conformal prediction: 2-3 days
- Integration & testing: 3-5 days

### Complexity Breakdown

**Simple:**
- Basic lag creation from single time series
- Point predictions only
- Single lag specification format

**Medium:**
- Multiple lag formats
- Covariate support
- Bootstrap uncertainty

**Complex:**
- Component-specific lags
- Mixed-frequency handling
- Conformal calibration
- Advanced sampling strategies

---

## References

### Darts
- [Darts Documentation](https://unit8co.github.io/darts/)
- [Linear Regression Model API](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.linear_regression_model.html)
- [GitHub: unit8co/darts](https://github.com/unit8co/darts)
- [PR #1399: Tabularization Refactoring](https://github.com/unit8co/darts/pull/1399)

### skforecast
- [skforecast Documentation](https://skforecast.org/)
- [Probabilistic Forecasting Guide](https://skforecast.org/0.14.0/user_guides/probabilistic-forecasting.html)
- [Conformal Prediction](https://skforecast.org/0.15.1/user_guides/probabilistic-forecasting-conformal-prediction.html)

### sktime
- [sktime: Probabilistic Forecasting](https://www.sktime.net/en/stable/examples/01b_forecasting_proba.html)
- [RecursiveTimeSeriesRegressionForecaster](https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.forecasting.compose.RecursiveTimeSeriesRegressionForecaster.html)

### GluonTS
- [GluonTS: Probabilistic Time Series Models (Paper)](https://arxiv.org/pdf/1906.05264)
- [GluonTS Documentation](https://ts.gluon.ai/stable/)
- [GitHub: awslabs/gluonts](https://github.com/awslabs/gluonts)

### PyMC
- [Forecasting with Structural AR Timeseries](https://www.pymc.io/projects/examples/en/latest/time_series/Forecasting_with_structural_timeseries.html)
- [Out-of-Model Predictions](https://www.pymc-labs.com/blog-posts/out-of-model-predictions-with-pymc)

### Other
- [Prophet: Uncertainty Intervals](https://facebook.github.io/prophet/docs/uncertainty_intervals.html)
- [pytorch-forecasting: TFT Tutorial](https://pytorch-forecasting.readthedocs.io/en/v1.4.0/tutorials/stallion.html)
- [Conformal Prediction with sklearn](https://towardsdatascience.com/time-series-forecasting-with-conformal-prediction-intervals-scikit-learn-is-all-you-need-4b68143a027a/)

---

## Appendix: Code Examples

### Example 1: Manual Lag Feature Creation

```python notest
import numpy as np
import pandas as pd

def create_lagged_features(
    df: pd.DataFrame,
    target_col: str,
    lags: list[int],
    covariate_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create lagged features for time series forecasting.

    Args:
        df: DataFrame with time series data
        target_col: Name of target column
        lags: List of lag values (e.g., [1, 2, 3, 6, 12])
        covariate_cols: Optional list of covariate columns

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target values (n_samples,)
    """
    max_lag = max(lags)
    features = []
    labels = []

    for i in range(max_lag, len(df)):
        row_features = []

        # Target lags
        for lag in sorted(lags):
            idx = i - lag
            row_features.append(df[target_col].iloc[idx])

        # Covariate lags
        if covariate_cols:
            for col in covariate_cols:
                for lag in sorted(lags):
                    idx = i - lag
                    row_features.append(df[col].iloc[idx])

        features.append(row_features)
        labels.append(df[target_col].iloc[i])

    return np.array(features), np.array(labels)
```

### Example 2: Recursive Forecasting with Bootstrap

```python notest
def bootstrap_recursive_forecast(
    model,
    history: np.ndarray,
    n_steps: int,
    n_samples: int,
    lags: list[int],
    residuals: np.ndarray,
) -> np.ndarray:
    """Generate bootstrapped recursive forecasts.

    Args:
        model: Fitted sklearn model
        history: Historical values
        n_steps: Number of steps to forecast
        n_samples: Number of bootstrap samples
        lags: Lag specification
        residuals: Training residuals for sampling

    Returns:
        Samples array of shape (n_steps, n_samples)
    """
    samples = np.zeros((n_steps, n_samples))

    for sample_idx in range(n_samples):
        current_history = history.copy()

        for step in range(n_steps):
            # Extract features from lags
            features = np.array([current_history[-lag] for lag in sorted(lags)])
            features = features.reshape(1, -1)

            # Predict mean
            pred_mean = model.predict(features)[0]

            # Sample residual and add to prediction
            residual = np.random.choice(residuals)
            sampled_value = pred_mean + residual

            # Store and update history
            samples[step, sample_idx] = sampled_value
            current_history = np.append(current_history, sampled_value)

    return samples
```

### Example 3: Conformal Prediction Intervals

```python notest
def conformal_prediction_interval(
    model,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    X_test: np.ndarray,
    alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute conformal prediction intervals.

    Args:
        model: Fitted sklearn model
        X_calib: Calibration features
        y_calib: Calibration targets
        X_test: Test features
        alpha: Miscoverage rate (0.1 for 90% coverage)

    Returns:
        lower_bounds: Lower prediction bounds
        upper_bounds: Upper prediction bounds
    """
    # Compute calibration errors
    y_calib_pred = model.predict(X_calib)
    calibration_scores = np.abs(y_calib - y_calib_pred)

    # Compute quantile of calibration scores
    q = np.quantile(calibration_scores, 1 - alpha)

    # Predict on test set
    y_test_pred = model.predict(X_test)

    # Construct intervals
    lower_bounds = y_test_pred - q
    upper_bounds = y_test_pred + q

    return lower_bounds, upper_bounds
```

### Example 4: Integration with chapkit

```python notest
from typing import Any
from chapkit import BaseConfig
from chapkit.data import DataFrame
from geojson_pydantic import FeatureCollection
from sklearn.ensemble import RandomForestRegressor
import numpy as np

class SKLearnModelConfig(BaseConfig):
    """Configuration for sklearn time series model."""
    lags: int = 12
    n_samples: int = 100
    bootstrap: bool = True


async def on_train(
    config: SKLearnModelConfig,
    data: DataFrame,
    run_info: RunInfo,
    geo: FeatureCollection | None = None,
) -> Any:
    """Train sklearn model with lag features."""
    df = data.to_pandas()

    # Create lagged features
    X, y = create_lagged_features(
        df,
        target_col="disease_cases",
        lags=list(range(1, config.lags + 1)),
        covariate_cols=["rainfall", "temperature"]
    )

    # Train model
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    # Compute residuals for bootstrapping
    y_pred = model.predict(X)
    residuals = y - y_pred

    return {
        "model": model,
        "lags": list(range(1, config.lags + 1)),
        "residuals": residuals,
        "last_values": df["disease_cases"].tail(config.lags).values
    }


async def on_predict(
    config: SKLearnModelConfig,
    model: Any,
    historic: DataFrame,
    future: DataFrame,
    run_info: RunInfo,
    geo: FeatureCollection | None = None,
) -> DataFrame:
    """Generate predictions with bootstrap uncertainty."""
    future_df = future.to_pandas()
    n_steps = len(future_df)

    # Generate bootstrap samples
    samples = bootstrap_recursive_forecast(
        model=model["model"],
        history=model["last_values"],
        n_steps=n_steps,
        n_samples=config.n_samples,
        lags=model["lags"],
        residuals=model["residuals"]
    )

    # Format as wide DataFrame with sample columns
    result = future_df[["time_period", "location"]].copy()
    for i in range(config.n_samples):
        result[f"sample_{i}"] = samples[:, i]

    return DataFrame.from_pandas(result)
```

---

*Document created: 2026-02-09*
*Last updated: 2026-02-09*
