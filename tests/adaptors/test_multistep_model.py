"""Tests for the recursive trajectory sampler (MultistepModel)."""

from __future__ import annotations

from typing import Any

import numpy as np

from chap_python_sdk.adaptors.multistep_model import (
    MultistepDistribution,
    MultistepModel,
    _build_lag_matrix,  # pyright: ignore[reportPrivateUsage]
)


class NormalDistribution:
    """Normal(mean, 1) distribution for testing."""

    def __init__(self, means: np.ndarray) -> None:
        """Store per-row means."""
        self._means = means

    def sample(self, n_samples: int) -> np.ndarray:
        """Draw samples with shape (n_samples, n_rows)."""
        rng = np.random.default_rng()
        return rng.normal(loc=self._means, size=(n_samples, len(self._means)))


class TrivialOneStepModel:
    """Returns Normal(last_lag, 1). Ignores training."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """No-op fit."""

    def predict_proba(self, X: np.ndarray) -> NormalDistribution:
        """Predict Normal centered on the last lag column."""
        last_lag = X[:, -1]
        return NormalDistribution(last_lag)


class TestBuildLagMatrix:
    """Tests for _build_lag_matrix helper."""

    def test_shape(self) -> None:
        """Lag matrix has (n - n_lags) rows and n_lags columns."""
        y = np.arange(10, dtype=float)
        lag_matrix = _build_lag_matrix(y, 3)
        assert lag_matrix.shape == (7, 3)

    def test_values(self) -> None:
        """Each row contains the correct lagged values, oldest to newest."""
        y = np.array([10, 20, 30, 40, 50], dtype=float)
        lag_matrix = _build_lag_matrix(y, 2)
        # Row 0 predicts y[2]=30, lags are [y[0], y[1]] = [10, 20]
        np.testing.assert_array_equal(lag_matrix[0], [10, 20])
        # Row 1 predicts y[3]=40, lags are [y[1], y[2]] = [20, 30]
        np.testing.assert_array_equal(lag_matrix[1], [20, 30])
        # Row 2 predicts y[4]=50, lags are [y[2], y[3]] = [30, 40]
        np.testing.assert_array_equal(lag_matrix[2], [30, 40])

    def test_single_lag(self) -> None:
        """Single lag produces a one-column matrix."""
        y = np.array([1, 2, 3, 4], dtype=float)
        lag_matrix = _build_lag_matrix(y, 1)
        assert lag_matrix.shape == (3, 1)
        np.testing.assert_array_equal(lag_matrix[:, 0], [1, 2, 3])


class TestMultistepModelFit:
    """Tests for MultistepModel.fit."""

    def test_fit_without_exogenous(self) -> None:
        """Fit completes without exogenous features."""
        model = TrivialOneStepModel()
        ms = MultistepModel(model, n_target_lags=2)
        y = np.arange(10, dtype=float)
        ms.fit(y)  # Should not raise

    def test_fit_with_exogenous(self) -> None:
        """Fit completes with exogenous features."""
        model = TrivialOneStepModel()
        ms = MultistepModel(model, n_target_lags=2)
        y = np.arange(10, dtype=float)
        X = np.ones((10, 3))
        ms.fit(y, X)  # Should not raise

    def test_fit_passes_correct_shapes(self) -> None:
        """Verify the one-step model receives correctly shaped data."""

        class RecordingModel:
            def __init__(self) -> None:
                self.X_shape: tuple[int, ...] | None = None
                self.y_shape: tuple[int, ...] | None = None

            def fit(self, X: np.ndarray, y: np.ndarray) -> None:
                self.X_shape = X.shape
                self.y_shape = y.shape

            def predict_proba(self, X: np.ndarray) -> Any:
                pass

        recorder = RecordingModel()
        ms = MultistepModel(recorder, n_target_lags=3)
        y = np.arange(20, dtype=float)
        ms.fit(y)

        assert recorder.X_shape == (17, 3)  # 20 - 3 rows, 3 lag columns
        assert recorder.y_shape == (17,)

    def test_fit_with_exogenous_correct_shapes(self) -> None:
        """Exogenous columns are prepended to lag columns."""

        class RecordingModel:
            def __init__(self) -> None:
                self.X_shape: tuple[int, ...] | None = None

            def fit(self, X: np.ndarray, y: np.ndarray) -> None:
                self.X_shape = X.shape

            def predict_proba(self, X: np.ndarray) -> Any:
                pass

        recorder = RecordingModel()
        ms = MultistepModel(recorder, n_target_lags=2)
        y = np.arange(10, dtype=float)
        X = np.ones((10, 4))
        ms.fit(y, X)

        # 8 rows, 4 exog + 2 lags = 6 columns
        assert recorder.X_shape == (8, 6)


class TestMultistepDistribution:
    """Tests for MultistepDistribution.sample."""

    def test_sample_shape(self) -> None:
        """Output shape is (n_samples, n_steps)."""
        model = TrivialOneStepModel()
        previous_y = np.array([1.0, 2.0, 3.0])
        dist = MultistepDistribution(
            model=model,
            previous_y=previous_y,
            n_steps=5,
            n_target_lags=3,
            X=None,
        )
        samples = dist.sample(100)
        assert samples.shape == (100, 5)

    def test_trajectories_are_stochastic(self) -> None:
        """Sampled trajectories are not all identical."""
        model = TrivialOneStepModel()
        previous_y = np.array([1.0, 2.0])
        dist = MultistepDistribution(
            model=model,
            previous_y=previous_y,
            n_steps=3,
            n_target_lags=2,
            X=None,
        )
        samples = dist.sample(50)
        assert not np.all(samples == samples[0])

    def test_with_exogenous_features(self) -> None:
        """Sampling works when exogenous features are provided."""
        model = TrivialOneStepModel()
        previous_y = np.array([5.0, 6.0])
        X = np.ones((4, 2))
        dist = MultistepDistribution(
            model=model,
            previous_y=previous_y,
            n_steps=4,
            n_target_lags=2,
            X=X,
        )
        samples = dist.sample(30)
        assert samples.shape == (30, 4)


class TestMultistepModelEndToEnd:
    """End-to-end tests with synthetic AR data."""

    def test_ar1_synthetic(self) -> None:
        """Full round-trip: generate AR(1) data, fit, predict, sample."""
        rng = np.random.default_rng(42)
        n = 100
        y = np.empty(n)
        y[0] = 0.0
        for t in range(1, n):
            y[t] = 0.8 * y[t - 1] + rng.normal(0, 1)

        model = TrivialOneStepModel()
        ms = MultistepModel(model, n_target_lags=1)
        ms.fit(y)

        previous_y = y[-1:]
        dist = ms.predict_proba(previous_y, n_steps=10)
        samples = dist.sample(100)

        assert samples.shape == (100, 10)
        assert not np.all(samples == samples[0])

    def test_ar2_synthetic(self) -> None:
        """AR(2) data with 2 lags."""
        rng = np.random.default_rng(123)
        n = 100
        y = np.empty(n)
        y[0] = 0.0
        y[1] = 0.5
        for t in range(2, n):
            y[t] = 0.5 * y[t - 1] + 0.3 * y[t - 2] + rng.normal(0, 0.5)

        model = TrivialOneStepModel()
        ms = MultistepModel(model, n_target_lags=2)
        ms.fit(y)

        previous_y = y[-2:]
        dist = ms.predict_proba(previous_y, n_steps=5)
        samples = dist.sample(200)

        assert samples.shape == (200, 5)
        assert not np.all(samples == samples[0])
