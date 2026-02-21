# Uncertainty Estimation for Count Data with Scikit-Learn Models

Research on methods for estimating prediction intervals and uncertainty for count data
when using RandomForest or other sklearn-compatible models.

## 1. Quantile Regression Forests

Extends Random Forests by keeping the full distribution of responses in each leaf node,
allowing you to request arbitrary quantiles at prediction time.

- **Library:** [quantile-forest](https://github.com/zillow/quantile-forest) (by Zillow) --
  well-maintained, Cython-optimized, sklearn-compatible drop-in replacement.
- **Pros:** Non-parametric, adapts interval width to local data density, no distributional
  assumptions, single model for any quantile.
- **Cons:** Predicted quantiles are continuous (need rounding/clipping to >= 0), higher memory
  since it stores all leaf responses.

```python notest
from quantile_forest import RandomForestQuantileRegressor

qrf = RandomForestQuantileRegressor(n_estimators=500)
qrf.fit(X_train, y_train)
y_pred = qrf.predict(X_test, quantiles=[0.05, 0.50, 0.95])
```

## 2. Conformal Prediction (MAPIE / CQR)

Distribution-free framework providing prediction intervals with **guaranteed marginal coverage**.
Conformalized Quantile Regression (CQR) combines quantile regression with a conformal calibration
step for adaptive intervals.

- **Library:** [MAPIE](https://github.com/scikit-learn-contrib/MAPIE) (scikit-learn-contrib) --
  wraps any sklearn estimator. Also [crepes](https://github.com/henrikbostrom/crepes).
- **Pros:** Coverage guarantee regardless of data distribution, model-agnostic, low computational
  overhead beyond the base model.
- **Cons:** Marginal guarantee (not conditional -- coverage may vary in subregions), requires a
  calibration split, intervals are continuous.

```python notest
from mapie.regression import MapieQuantileRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

base = HistGradientBoostingRegressor(loss="poisson")
mapie = MapieQuantileRegressor(estimator=base, cv="split", alpha=0.1)
mapie.fit(X_train, y_train)
y_pred, y_pis = mapie.predict(X_test)
```

## 3. Per-Tree Bootstrap Predictions

Extract predictions from each tree in a Random Forest via `rf.estimators_` and compute
percentiles of the per-tree distribution.

- **Library:** Built into sklearn. Also
  [forestci](https://contrib.scikit-learn.org/forest-confidence-interval/) for the Infinitesimal
  Jackknife variance estimator (Wager et al. 2014).
- **Pros:** Zero additional dependencies, very cheap.
- **Cons:** Captures only ensemble variance (not data noise), so intervals are typically
  **too narrow**. `forestci` gives variance, not intervals -- converting to intervals requires
  a distributional assumption.

```python notest
import numpy as np

predictions = np.array([tree.predict(X_test) for tree in rf.estimators_])
lower = np.percentile(predictions, 5, axis=0)
upper = np.percentile(predictions, 95, axis=0)
```

## 4. NGBoost (Natural Gradient Boosting)

Gradient boosting that outputs full parametric probability distributions. Supports **Poisson
distribution** natively, so you get proper count-valued prediction intervals directly.

- **Library:** [ngboost](https://github.com/stanfordmlgroup/ngboost), sklearn-compatible API.
- **Pros:** Proper Poisson PMF -- intervals are automatically non-negative and integer-valued.
  No post-processing needed.
- **Cons:** Poisson assumes mean = variance (equidispersion). No built-in Negative Binomial for
  overdispersed data. Slower than standard gradient boosting.

```python notest
import scipy.stats
from ngboost import NGBRegressor
from ngboost.distns import Poisson

ngb = NGBRegressor(Dist=Poisson, n_estimators=500)
ngb.fit(X_train, y_train)
dist = ngb.pred_dist(X_test)
lower = scipy.stats.poisson.ppf(0.05, mu=dist.mean())
upper = scipy.stats.poisson.ppf(0.95, mu=dist.mean())
```

## 5. Poisson Loss in Sklearn Gradient Boosting

`HistGradientBoostingRegressor(loss="poisson")` optimizes Poisson deviance with a log-link,
ensuring non-negative predictions. For intervals, you can either:

- **Assume Poisson:** use `scipy.stats.poisson.ppf(q, mu=y_pred)` to get intervals from the
  predicted mean.
- **Train quantile models:** use `loss="quantile"` with separate models for lower/upper bounds.

```python notest
import scipy.stats
from sklearn.ensemble import HistGradientBoostingRegressor

model = HistGradientBoostingRegressor(loss="poisson")
model.fit(X_train, y_train)
mu = model.predict(X_test)
lower = scipy.stats.poisson.ppf(0.05, mu=mu)
upper = scipy.stats.poisson.ppf(0.95, mu=mu)
```

## 6. Bayesian Approaches

- **PyMC / Bambi:** Full Bayesian inference with Poisson, Negative Binomial, or Zero-Inflated
  likelihoods. Most statistically principled for counts, but NOT sklearn-compatible and
  computationally expensive (MCMC).
- **BayesianRidge (sklearn):** Assumes Gaussian likelihood -- **not appropriate for count data**.

## 7. Why Gaussian Intervals Fail for Counts

- Counts are discrete and non-negative; Gaussian intervals are continuous and can go negative.
- Count data is heteroscedastic (variance grows with mean); Gaussian intervals assume constant
  variance.
- Count distributions are right-skewed for small means; Gaussian intervals are symmetric.
- Zero-inflation and overdispersion are common and unaddressed by Gaussian assumptions.

## Recommendations Summary

| Scenario | Best approach | Runner-up |
|---|---|---|
| General-purpose, sklearn only | HistGBR quantile regression | Per-tree predictions |
| Need coverage guarantees | Conformal (MAPIE + CQR) | Quantile forest |
| Well-described by Poisson | NGBoost with Poisson | HistGBR Poisson + scipy |
| Overdispersed data | Quantile forest or CQR | PyMC with NegBin |
| Zero-inflated data | PyMC/Bambi (ZINB) | Quantile forest |
| Large datasets (100k+) | HistGBR quantile or CQR | Quantile forest |

**Top 3 overall:**

1. **Conformal CQR via MAPIE** for coverage guarantees
2. **Quantile Regression Forests** for non-parametric flexibility
3. **NGBoost with Poisson** for proper probabilistic count predictions

**Always post-process** (except NGBoost Poisson): clip lower bounds to 0 and round outward
to integers.

## References

- [quantile-forest (Zillow)](https://github.com/zillow/quantile-forest)
- [MAPIE (scikit-learn-contrib)](https://github.com/scikit-learn-contrib/MAPIE)
- [crepes](https://github.com/henrikbostrom/crepes)
- [NGBoost (Stanford)](https://github.com/stanfordmlgroup/ngboost)
- [forestci (scikit-learn-contrib)](https://contrib.scikit-learn.org/forest-confidence-interval/)
- [Wager, Hastie, Efron 2014 -- Confidence Intervals for Random Forests](https://jmlr.org/papers/volume15/wager14a/wager14a.pdf)
- [Conformalized Quantile Regression (Romano et al. 2019)](https://arxiv.org/pdf/1905.03222)
- [sklearn HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)
- [sklearn Prediction Intervals for Gradient Boosting](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html)
