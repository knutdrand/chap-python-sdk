"""Tests for ResidualBootstrapModel and ResidualDistribution."""

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn", reason="scikit-learn not installed")

from chap_python_sdk.adaptors.multistep.one_step_model import (  # noqa: E402
    ResidualBootstrapModel,
    ResidualDistribution,
)


class TestResidualDistribution:
    """Tests for ResidualDistribution."""

    def test_sample_shape(self) -> None:
        """Output has shape (n_samples, n_rows)."""
        predictions = np.array([10.0, 20.0, 30.0])
        residuals = np.array([-1.0, 0.0, 1.0, 2.0, -2.0])
        dist = ResidualDistribution(predictions, residuals)
        samples = dist.sample(50)
        assert samples.shape == (50, 3)

    def test_samples_nonnegative(self) -> None:
        """All samples are clamped to >= 0."""
        predictions = np.array([1.0, 0.5])
        residuals = np.array([-10.0, -5.0, 0.0, 5.0])
        dist = ResidualDistribution(predictions, residuals)
        samples = dist.sample(200)
        assert np.all(samples >= 0)

    def test_samples_have_variability(self) -> None:
        """Samples are not all identical."""
        predictions = np.array([10.0, 20.0])
        residuals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        dist = ResidualDistribution(predictions, residuals)
        samples = dist.sample(100)
        assert not np.all(samples == samples[0])


class TestResidualBootstrapModel:
    """Tests for ResidualBootstrapModel."""

    def test_fit_stores_residuals(self) -> None:
        """After fit, residuals are stored."""
        model = ResidualBootstrapModel(
            model_class="sklearn.linear_model.Ridge",
            model_params={"alpha": 1.0},
        )
        rng = np.random.default_rng(42)
        X = rng.normal(size=(50, 3))
        y = X @ np.array([1.0, 2.0, 3.0]) + rng.normal(0, 0.1, size=50)

        model.fit(X, y)
        assert len(model._residuals) == 50  # pyright: ignore[reportPrivateUsage]
        # Residuals should be small for a well-fit model
        assert np.std(model._residuals) < 1.0  # pyright: ignore[reportPrivateUsage]

    def test_predict_proba_returns_distribution(self) -> None:
        """predict_proba returns a ResidualDistribution."""
        model = ResidualBootstrapModel(
            model_class="sklearn.linear_model.Ridge",
            model_params={"alpha": 1.0},
        )
        rng = np.random.default_rng(42)
        X = rng.normal(size=(50, 3))
        y = X @ np.array([1.0, 2.0, 3.0])

        model.fit(X, y)
        dist = model.predict_proba(X[:5])

        assert isinstance(dist, ResidualDistribution)
        samples = dist.sample(10)
        assert samples.shape == (10, 5)

    def test_gradient_boosting_regressor(self) -> None:
        """Works with GradientBoostingRegressor."""
        model = ResidualBootstrapModel(
            model_class="sklearn.ensemble.GradientBoostingRegressor",
            model_params={"n_estimators": 10, "random_state": 42},
        )
        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 2))
        y = np.abs(X[:, 0] * 2 + X[:, 1] + rng.normal(0, 0.1, size=100))

        model.fit(X, y)
        dist = model.predict_proba(X[:3])
        samples = dist.sample(20)
        assert samples.shape == (20, 3)

    def test_different_sklearn_regressors(self) -> None:
        """Works with various sklearn regressors."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(50, 2))
        y = np.abs(X[:, 0] + rng.normal(0, 0.1, size=50))

        for model_class in [
            "sklearn.linear_model.Ridge",
            "sklearn.linear_model.Lasso",
            "sklearn.tree.DecisionTreeRegressor",
        ]:
            model = ResidualBootstrapModel(model_class=model_class, model_params={})
            model.fit(X, y)
            dist = model.predict_proba(X[:2])
            samples = dist.sample(5)
            assert samples.shape == (5, 2), f"Failed for {model_class}"
