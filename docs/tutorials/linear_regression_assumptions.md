# Linear Regression Assumptions in Time Series

## What are the assumptions of Linear Regression?

Linear regression relies on several key assumptions about the data. When these hold, ordinary least squares (OLS) gives the best linear unbiased estimator (BLUE). When they break, our estimates and uncertainty intervals become unreliable.

### 1. Independence of observations

Each observation should be independent of the others. The residual for one observation should carry no information about the residual for another.

### 2. Homoscedasticity (constant variance)

The variance of the residuals should be the same across all levels of the predictor variables. A model should not be more uncertain at some times or places than others.

### 3. Normality of residuals

The residuals should be approximately normally distributed. This matters most for constructing confidence intervals and hypothesis tests.

### 4. Linearity

The relationship between predictors and the target should be linear. Non-linear patterns in the residuals indicate that the model is misspecified.

### 5. No multicollinearity

Predictor variables should not be too strongly correlated with each other. High collinearity inflates the variance of coefficient estimates.

---

## How are these assumptions broken in our time series models?

When we model disease case counts across locations and time, nearly every assumption is violated.

### Autocorrelation breaks independence

Disease cases at time $t$ are correlated with cases at $t-1$, $t-2$, etc. An outbreak does not start and stop within a single time period. This means residuals are correlated across time, which:

- Makes standard errors too small
- Leads to overconfident prediction intervals
- Biases model selection criteria

### Heteroscedasticity from count data

Disease counts follow distributions where the variance scales with the mean (e.g. Poisson). Locations with more cases naturally have higher variance. Seasonal peaks also show more variability than troughs.

### Non-normality from count data

Case counts are:

- Non-negative integers (not continuous)
- Right-skewed (many zeros and small values, occasional large outbreaks)
- Often zero-inflated

### Non-linear relationships

- The effect of rainfall on disease may be non-linear (some rain enables mosquito breeding, too much rain washes away breeding sites)
- Temperature effects are often U-shaped or threshold-based
- Seasonal patterns are periodic, not linear

### Spatial correlation

Observations from nearby locations are correlated. An outbreak in one district often spreads to neighbors. This is a form of non-independence that standard linear regression ignores.

---

## How can we alleviate the broken assumptions?

The multistep model framework provides several tools to address each violation.

### Lagged target features address autocorrelation

By including $y_{t-1}, y_{t-2}, \ldots, y_{t-k}$ as features, we explicitly model the temporal dependence. The model learns the autoregressive structure, and the residuals become closer to independent.

```python notest
from chap_python_sdk.adaptors.multistep.config import MultistepConfig

# Use 12 monthly lags to capture annual cycles
config = MultistepConfig(n_target_lags=12)
```

### Log transform stabilizes variance

Applying $\log(1 + y)$ before modeling:

- Pulls in large values, reducing the impact of outliers
- Stabilizes variance (since for Poisson-like data, $\text{Var}(Y) \propto E[Y]$)
- Makes the distribution more symmetric

```python notest
config = MultistepConfig(log_transform_target=True)
```

### Feature standardization reduces multicollinearity

Centering and scaling covariates to zero mean and unit variance:

- Improves numerical stability
- Makes coefficient magnitudes comparable
- Helps gradient-based optimizers converge

```python notest
config = MultistepConfig(standardize_covariates=True)
```

### Seasonal features capture periodic patterns

Adding one-hot encoded month or season columns lets the model learn periodic effects without requiring a linear relationship with time.

```python notest
from chap_python_sdk.adaptors.multistep.pipeline import SeasonEncoder

encoder = SeasonEncoder()  # Adds month_1 through month_12
```

### Location features capture spatial heterogeneity

One-hot encoding locations allows the model to learn location-specific intercepts, accounting for baseline differences in case rates.

```python notest
from chap_python_sdk.adaptors.multistep.pipeline import LocationEncoder

encoder = LocationEncoder()  # Adds location_Bokeo, location_Champasak, etc.
```

### Rate transform (dividing by population) addresses scale differences

Converting counts to rates (cases per capita) makes locations with different population sizes comparable and further stabilizes variance.

### Non-linear models

Using tree-based regressors like GradientBoostingRegressor instead of plain linear regression handles non-linear relationships automatically, without needing to specify the functional form.

```python notest
config = MultistepConfig(
    model_class="sklearn.ensemble.GradientBoostingRegressor",
    model_params={"n_estimators": 100, "max_depth": 3},
)
```

### Residual bootstrapping for honest uncertainty

Since the residuals are not perfectly normal, we use residual bootstrapping instead of parametric intervals. This resamples the actual training residuals to generate prediction trajectories, giving uncertainty estimates that respect the true error distribution.

---

## Summary

| Assumption | Violation in time series | Mitigation |
|---|---|---|
| Independence | Autocorrelation | Lagged target features |
| Homoscedasticity | Variance scales with mean | Log transform |
| Normality | Count data, skewed | Residual bootstrapping |
| Linearity | Non-linear climate effects | Tree-based models |
| No multicollinearity | Correlated covariates | Feature standardization |
| — | Spatial heterogeneity | Location encoding |
| — | Seasonal periodicity | Season encoding |

The following tutorials show how to apply these mitigations step by step using the multistep model framework.
