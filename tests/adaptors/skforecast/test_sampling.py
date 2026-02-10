"""Tests for bootstrap sampling strategy."""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

skforecast = pytest.importorskip("skforecast", reason="skforecast not installed")

from skforecast.recursive import ForecasterRecursiveMultiSeries  # type: ignore[import-untyped]  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # type: ignore[import-untyped]  # noqa: E402

from chap_python_sdk.adaptors.skforecast.sampling import bootstrap_recursive_samples  # noqa: E402


class TestBootstrapRecursiveSamples:
    """Tests for bootstrap_recursive_samples function."""

    def test_output_shape(self) -> None:
        """Test that output has correct shape."""
        target_data = pd.DataFrame(
            {
                "A": [10, 20, 30, 40, 50],
                "B": [15, 25, 35, 45, 55],
            },
            index=pd.date_range("2023-01", periods=5, freq="MS"),
        )

        forecaster = ForecasterRecursiveMultiSeries(
            regressor=RandomForestRegressor(n_estimators=10, random_state=42),
            lags=2,
        )
        forecaster.fit(series=target_data)

        residuals = {
            "A": np.array([1.0, -1.0, 0.5, -0.5]),
            "B": np.array([2.0, -2.0, 1.0, -1.0]),
        }

        result = bootstrap_recursive_samples(
            forecaster=forecaster,
            residuals_by_location=residuals,
            n_steps=3,
            n_samples=50,
            exog_future=None,
            locations=["A", "B"],
        )

        assert len(result) == 2
        assert "A" in result
        assert "B" in result
        assert result["A"].shape == (3, 50)
        assert result["B"].shape == (3, 50)

    def test_non_negative_predictions(self) -> None:
        """Test that predictions are non-negative."""
        target_data = pd.DataFrame(
            {
                "A": [10, 20, 30, 40, 50],
            },
            index=pd.date_range("2023-01", periods=5, freq="MS"),
        )

        forecaster = ForecasterRecursiveMultiSeries(
            regressor=RandomForestRegressor(n_estimators=10, random_state=42),
            lags=2,
        )
        forecaster.fit(series=target_data)

        residuals = {
            "A": np.array([-100.0, -50.0, 10.0, 5.0]),
        }

        result = bootstrap_recursive_samples(
            forecaster=forecaster,
            residuals_by_location=residuals,
            n_steps=2,
            n_samples=20,
            exog_future=None,
            locations=["A"],
        )

        assert (result["A"].values >= 0).all()

    def test_with_exogenous_variables(self) -> None:
        """Test sampling with exogenous variables."""
        target_data = pd.DataFrame(
            {
                "A": [10, 20, 30, 40, 50],
            },
            index=pd.date_range("2023-01", periods=5, freq="MS"),
        )

        exog_data = pd.DataFrame(
            {
                "rainfall_A": [100, 150, 200, 180, 120],
            },
            index=pd.date_range("2023-01", periods=5, freq="MS"),
        )

        forecaster = ForecasterRecursiveMultiSeries(
            regressor=RandomForestRegressor(n_estimators=10, random_state=42),
            lags=2,
        )
        forecaster.fit(series=target_data, exog=exog_data)

        exog_future = pd.DataFrame(
            {
                "rainfall_A": [130, 140],
            },
            index=pd.date_range("2023-06", periods=2, freq="MS"),
        )

        residuals = {
            "A": np.array([1.0, -1.0, 0.5, -0.5]),
        }

        result = bootstrap_recursive_samples(
            forecaster=forecaster,
            residuals_by_location=residuals,
            n_steps=2,
            n_samples=10,
            exog_future=exog_future,
            locations=["A"],
        )

        assert result["A"].shape == (2, 10)

    def test_variability_in_samples(self) -> None:
        """Test that different samples have variability."""
        target_data = pd.DataFrame(
            {
                "A": [10, 20, 30, 40, 50],
            },
            index=pd.date_range("2023-01", periods=5, freq="MS"),
        )

        forecaster = ForecasterRecursiveMultiSeries(
            regressor=RandomForestRegressor(n_estimators=10, random_state=42),
            lags=2,
        )
        forecaster.fit(series=target_data)

        residuals = {
            "A": np.array([5.0, -5.0, 3.0, -3.0, 2.0, -2.0]),
        }

        result = bootstrap_recursive_samples(
            forecaster=forecaster,
            residuals_by_location=residuals,
            n_steps=2,
            n_samples=100,
            exog_future=None,
            locations=["A"],
        )

        std_dev = result["A"].std(axis=1)
        assert (std_dev > 0).all()

    def test_empty_residuals_fallback(self) -> None:
        """Test that empty residuals fall back to zero variance."""
        target_data = pd.DataFrame(
            {
                "A": [10, 20, 30, 40, 50],
            },
            index=pd.date_range("2023-01", periods=5, freq="MS"),
        )

        forecaster = ForecasterRecursiveMultiSeries(
            regressor=RandomForestRegressor(n_estimators=10, random_state=42),
            lags=2,
        )
        forecaster.fit(series=target_data)

        residuals = {
            "A": np.array([]),
        }

        result = bootstrap_recursive_samples(
            forecaster=forecaster,
            residuals_by_location=residuals,
            n_steps=2,
            n_samples=10,
            exog_future=None,
            locations=["A"],
        )

        assert result["A"].shape == (2, 10)
