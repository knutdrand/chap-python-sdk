"""Tests for bootstrap sampling strategy."""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

skforecast = pytest.importorskip("skforecast", reason="skforecast not installed")

from skforecast.recursive import ForecasterRecursiveMultiSeries  # type: ignore[import-untyped]  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # type: ignore[import-untyped]  # noqa: E402

from chap_python_sdk.adaptors.skforecast.sampling import bootstrap_recursive_samples  # noqa: E402


def _fit_forecaster(
    locations: list[str], n_rows: int = 8, exog: pd.DataFrame | None = None
) -> ForecasterRecursiveMultiSeries:
    """Helper to create a fitted forecaster."""
    rng = np.random.default_rng(42)
    data = {loc: rng.poisson(50, size=n_rows).astype(float) for loc in locations}
    target = pd.DataFrame(data, index=pd.date_range("2023-01", periods=n_rows, freq="MS"))

    forecaster = ForecasterRecursiveMultiSeries(
        estimator=RandomForestRegressor(n_estimators=10, random_state=42),
        lags=2,
    )
    forecaster.fit(series=target, exog=exog)
    return forecaster


class TestBootstrapRecursiveSamples:
    """Tests for bootstrap_recursive_samples function."""

    def test_output_shape(self) -> None:
        """Test that output has correct shape."""
        forecaster = _fit_forecaster(["A", "B"])

        residuals = {
            "A": {0: np.array([1.0, -1.0, 0.5]), 1: np.array([-0.5, 0.3])},
            "B": {0: np.array([2.0, -2.0, 1.0]), 1: np.array([-1.0, 0.5])},
        }

        result = bootstrap_recursive_samples(
            forecaster=forecaster,
            residuals_by_step=residuals,
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
        forecaster = _fit_forecaster(["A"])

        residuals = {
            "A": {0: np.array([-100.0, -50.0, 10.0, 5.0])},
        }

        result = bootstrap_recursive_samples(
            forecaster=forecaster,
            residuals_by_step=residuals,
            n_steps=2,
            n_samples=20,
            exog_future=None,
            locations=["A"],
        )

        assert (result["A"].values >= 0).all()  # pyright: ignore[reportAttributeAccessIssue]

    def test_with_exogenous_variables(self) -> None:
        """Test sampling with exogenous variables."""
        exog_data = pd.DataFrame(
            {"rainfall_A": np.random.default_rng(0).normal(100, 20, 8)},
            index=pd.date_range("2023-01", periods=8, freq="MS"),
        )
        forecaster = _fit_forecaster(["A"], exog=exog_data)

        exog_future = pd.DataFrame(
            {"rainfall_A": [130.0, 140.0]},
            index=pd.date_range("2023-09", periods=2, freq="MS"),
        )

        residuals = {
            "A": {0: np.array([1.0, -1.0, 0.5, -0.5])},
        }

        result = bootstrap_recursive_samples(
            forecaster=forecaster,
            residuals_by_step=residuals,
            n_steps=2,
            n_samples=10,
            exog_future=exog_future,
            locations=["A"],
        )

        assert result["A"].shape == (2, 10)

    def test_variability_in_samples(self) -> None:
        """Test that different samples have variability."""
        forecaster = _fit_forecaster(["A"])

        residuals = {
            "A": {0: np.array([5.0, -5.0, 3.0, -3.0, 2.0, -2.0])},
        }

        result = bootstrap_recursive_samples(
            forecaster=forecaster,
            residuals_by_step=residuals,
            n_steps=2,
            n_samples=100,
            exog_future=None,
            locations=["A"],
        )

        std_dev = result["A"].std(axis=1)
        assert (std_dev > 0).all()  # pyright: ignore[reportAttributeAccessIssue]

    def test_empty_residuals_fallback(self) -> None:
        """Test that empty residuals fall back to zero variance."""
        forecaster = _fit_forecaster(["A"])

        residuals: dict[str, dict[int, np.ndarray]] = {
            "A": {0: np.array([])},
        }

        result = bootstrap_recursive_samples(
            forecaster=forecaster,
            residuals_by_step=residuals,
            n_steps=2,
            n_samples=10,
            exog_future=None,
            locations=["A"],
        )

        assert result["A"].shape == (2, 10)

    def test_step_specific_residuals(self) -> None:
        """Test that different steps use different residual distributions."""
        forecaster = _fit_forecaster(["A"])

        # Step 0 has small residuals, step 1 has large residuals
        residuals = {
            "A": {
                0: np.array([0.1, -0.1, 0.05, -0.05]),
                1: np.array([10.0, -10.0, 8.0, -8.0]),
            },
        }

        np.random.seed(42)
        result = bootstrap_recursive_samples(
            forecaster=forecaster,
            residuals_by_step=residuals,
            n_steps=2,
            n_samples=200,
            exog_future=None,
            locations=["A"],
        )

        # Step 1 should have larger variance than step 0
        std_step0 = result["A"].iloc[0].std()
        std_step1 = result["A"].iloc[1].std()
        assert std_step1 > std_step0

    def test_missing_location_residuals(self) -> None:
        """Test fallback when a location has no residuals."""
        forecaster = _fit_forecaster(["A", "B"])

        # Only A has residuals, B is missing
        residuals: dict[str, dict[int, np.ndarray]] = {
            "A": {0: np.array([1.0, -1.0])},
        }

        result = bootstrap_recursive_samples(
            forecaster=forecaster,
            residuals_by_step=residuals,
            n_steps=2,
            n_samples=10,
            exog_future=None,
            locations=["A", "B"],
        )

        assert result["A"].shape == (2, 10)
        assert result["B"].shape == (2, 10)
