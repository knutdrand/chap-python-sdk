"""Tests for the recursive trajectory sampler (MultistepModel)."""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from chap_python_sdk.adaptors.multistep_model import (
    DeterministicMultistepModel,
    MultistepDistribution,
    MultistepModel,
    _build_lag_matrix,  # pyright: ignore[reportPrivateUsage]
    _build_lag_matrix_xr,  # pyright: ignore[reportPrivateUsage]
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


class TestBuildLagMatrixXr:
    """Tests for _build_lag_matrix_xr helper."""

    def test_single_location(self) -> None:
        """Shape and values match numpy version for a single location."""
        y_np = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y_xr = xr.DataArray(y_np, dims=["time"])

        result = _build_lag_matrix_xr(y_xr, n_lags=2)

        assert result.dims == ("lag", "time")
        assert result.shape == (2, 3)
        # Row 0 (lag=0) for time index 0 predicts y[2]=30, lag is y[0]=10
        np.testing.assert_array_equal(result.sel(lag=0).values, [10.0, 20.0, 30.0])
        np.testing.assert_array_equal(result.sel(lag=1).values, [20.0, 30.0, 40.0])

    def test_multi_location(self) -> None:
        """Verify (lag, location, time) shape and per-location correctness."""
        data = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [10.0, 20.0, 30.0, 40.0, 50.0],
            ]
        )
        y = xr.DataArray(data, dims=["location", "time"])

        result = _build_lag_matrix_xr(y, n_lags=2)

        assert set(result.dims) == {"lag", "location", "time"}
        assert result.sizes["lag"] == 2
        assert result.sizes["location"] == 2
        assert result.sizes["time"] == 3

        # Location 0: lags for the first valid target (y=3)
        np.testing.assert_array_equal(result.sel(lag=0).isel(location=0).values, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result.sel(lag=1).isel(location=0).values, [2.0, 3.0, 4.0])
        # Location 1: lags for the first valid target (y=30)
        np.testing.assert_array_equal(result.sel(lag=0).isel(location=1).values, [10.0, 20.0, 30.0])
        np.testing.assert_array_equal(result.sel(lag=1).isel(location=1).values, [20.0, 30.0, 40.0])


class TestFitMulti:
    """Tests for MultistepModel.fit_multi."""

    def test_pools_locations(self) -> None:
        """Pooled sample count = n_locs * (T - n_lags) rows."""

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
        ms = MultistepModel(recorder, n_target_lags=2)

        n_locs, T = 3, 10
        y = xr.DataArray(
            np.arange(n_locs * T, dtype=float).reshape(n_locs, T),
            dims=["location", "time"],
        )
        ms.fit_multi(y)

        expected_rows = n_locs * (T - 2)  # 3 * 8 = 24
        assert recorder.X_shape == (expected_rows, 2)
        assert recorder.y_shape == (expected_rows,)

    def test_with_exog(self) -> None:
        """Feature count = n_exog + n_lags when exogenous features provided."""

        class RecordingModel:
            def __init__(self) -> None:
                self.X_shape: tuple[int, ...] | None = None

            def fit(self, X: np.ndarray, y: np.ndarray) -> None:
                self.X_shape = X.shape

            def predict_proba(self, X: np.ndarray) -> Any:
                pass

        recorder = RecordingModel()
        n_lags = 2
        ms = MultistepModel(recorder, n_target_lags=n_lags)

        n_locs, T, n_exog = 2, 8, 3
        y = xr.DataArray(
            np.arange(n_locs * T, dtype=float).reshape(n_locs, T),
            dims=["location", "time"],
        )
        X = xr.DataArray(
            np.ones((n_locs, T, n_exog)),
            dims=["location", "time", "feature"],
        )
        ms.fit_multi(y, X)

        expected_rows = n_locs * (T - n_lags)  # 2 * 6 = 12
        expected_cols = n_exog + n_lags  # 3 + 2 = 5
        assert recorder.X_shape == (expected_rows, expected_cols)


class TestPredictMulti:
    """Tests for MultistepModel.predict_multi."""

    def test_shape(self) -> None:
        """Output has dims (location, trajectory, step)."""
        model = TrivialOneStepModel()
        ms = MultistepModel(model, n_target_lags=2)

        n_locs, n_steps, n_samples = 3, 5, 10
        previous_y = xr.DataArray(
            np.arange(n_locs * 2, dtype=float).reshape(n_locs, 2),
            dims=["location", "time"],
            coords={"location": ["A", "B", "C"]},
        )
        result = ms.predict_multi(previous_y, n_steps=n_steps, n_samples=n_samples)

        assert result.dims == ("location", "trajectory", "step")
        assert result.shape == (n_locs, n_samples, n_steps)
        np.testing.assert_array_equal(result.coords["location"].values, ["A", "B", "C"])

    def test_roundtrip(self) -> None:
        """fit_multi + predict_multi end-to-end produces valid output."""
        model = TrivialOneStepModel()
        ms = MultistepModel(model, n_target_lags=2)

        rng = np.random.default_rng(42)
        n_locs, T = 2, 50
        data = rng.normal(size=(n_locs, T))
        y = xr.DataArray(data, dims=["location", "time"], coords={"location": ["X", "Y"]})

        ms.fit_multi(y)

        previous_y = y.isel(time=slice(-2, None))
        result = ms.predict_multi(previous_y, n_steps=5, n_samples=20)

        assert result.shape == (2, 20, 5)
        # Trajectories should be stochastic
        assert not np.all(result.isel(location=0).values == result.isel(location=0).values[0])


class TrivialDeterministicModel:
    """Returns the mean of last lag as prediction."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """No-op fit."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return last lag value as prediction."""
        return X[:, -1]


class TestDeterministicMultistepModel:
    """Tests for DeterministicMultistepModel."""

    def test_fit_without_exogenous(self) -> None:
        """Fit without exogenous features should succeed."""
        model = TrivialDeterministicModel()
        ms = DeterministicMultistepModel(model, n_target_lags=2)
        y = np.arange(10, dtype=float)
        ms.fit(y)

    def test_predict_shape(self) -> None:
        """Predict returns correct shape."""
        model = TrivialDeterministicModel()
        ms = DeterministicMultistepModel(model, n_target_lags=2)
        y = np.arange(10, dtype=float)
        ms.fit(y)
        preds = ms.predict(y[-2:], n_steps=5)
        assert preds.shape == (5,)

    def test_predict_deterministic(self) -> None:
        """Same input produces same output (no stochasticity)."""
        model = TrivialDeterministicModel()
        ms = DeterministicMultistepModel(model, n_target_lags=2)
        y = np.arange(10, dtype=float)
        ms.fit(y)
        preds1 = ms.predict(y[-2:], n_steps=5)
        preds2 = ms.predict(y[-2:], n_steps=5)
        np.testing.assert_array_equal(preds1, preds2)

    def test_predict_multi_shape(self) -> None:
        """Multi-location predict returns correct shape and dims."""
        model = TrivialDeterministicModel()
        ms = DeterministicMultistepModel(model, n_target_lags=2)
        n_locs, T = 3, 10
        y = xr.DataArray(
            np.arange(n_locs * T, dtype=float).reshape(n_locs, T),
            dims=["location", "time"],
            coords={"location": ["A", "B", "C"]},
        )
        ms.fit_multi(y)
        previous_y = y.isel(time=slice(-2, None))
        result = ms.predict_multi(previous_y, n_steps=5)
        assert result.dims == ("location", "step")
        assert result.shape == (3, 5)

    def test_predict_multi_deterministic(self) -> None:
        """Multi-location predictions are deterministic."""
        model = TrivialDeterministicModel()
        ms = DeterministicMultistepModel(model, n_target_lags=2)
        y = xr.DataArray(
            np.arange(20, dtype=float).reshape(2, 10),
            dims=["location", "time"],
            coords={"location": ["X", "Y"]},
        )
        ms.fit_multi(y)
        prev = y.isel(time=slice(-2, None))
        r1 = ms.predict_multi(prev, n_steps=3)
        r2 = ms.predict_multi(prev, n_steps=3)
        np.testing.assert_array_equal(r1.values, r2.values)

    def test_with_exogenous(self) -> None:
        """Fit and predict with exogenous features."""
        model = TrivialDeterministicModel()
        ms = DeterministicMultistepModel(model, n_target_lags=2)
        y = np.arange(10, dtype=float)
        X = np.ones((10, 3))
        ms.fit(y, X)
        X_future = np.ones((5, 3))
        preds = ms.predict(y[-2:], n_steps=5, X=X_future)
        assert preds.shape == (5,)
