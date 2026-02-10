"""Tests for forecaster wrapper."""

import pandas as pd  # type: ignore[import-untyped]
import pytest

skforecast = pytest.importorskip("skforecast", reason="skforecast not installed")

from chap_python_sdk.adaptors.skforecast import SkforecastConfig  # noqa: E402
from chap_python_sdk.adaptors.skforecast.forecaster import SkforecastWrapper  # noqa: E402


class TestSkforecastWrapper:
    """Tests for SkforecastWrapper class."""

    def test_fit_basic(self) -> None:
        """Test basic fitting of the forecaster."""
        config = SkforecastConfig(lags=2, model_params={"n_estimators": 10, "random_state": 42})

        wrapper = SkforecastWrapper(config)

        target_data = pd.DataFrame(
            {
                "A": [10, 20, 30, 40, 50],
                "B": [15, 25, 35, 45, 55],
            },
            index=pd.date_range("2023-01", periods=5, freq="MS"),
        )

        wrapper.fit(target_data, exog_wide=None)

        assert wrapper.forecaster is not None
        assert wrapper.forecaster.is_fitted

    def test_fit_with_exogenous(self) -> None:
        """Test fitting with exogenous variables."""
        config = SkforecastConfig(
            lags=2, exogenous_variables=["rainfall"], model_params={"n_estimators": 10, "random_state": 42}
        )

        wrapper = SkforecastWrapper(config)

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

        wrapper.fit(target_data, exog_wide=exog_data)

        assert wrapper.forecaster is not None
        assert wrapper.forecaster.is_fitted

    def test_residuals_computed(self) -> None:
        """Test that residuals are computed during fit."""
        config = SkforecastConfig(lags=2, use_bootstrapping=True, model_params={"n_estimators": 10, "random_state": 42})

        wrapper = SkforecastWrapper(config)

        target_data = pd.DataFrame(
            {
                "A": [10, 20, 30, 40, 50],
                "B": [15, 25, 35, 45, 55],
            },
            index=pd.date_range("2023-01", periods=5, freq="MS"),
        )

        wrapper.fit(target_data, exog_wide=None)

        assert "A" in wrapper.residuals_by_location
        assert "B" in wrapper.residuals_by_location
        assert len(wrapper.residuals_by_location["A"]) > 0

    def test_predict_samples(self) -> None:
        """Test generating prediction samples."""
        config = SkforecastConfig(lags=2, n_samples=20, model_params={"n_estimators": 10, "random_state": 42})

        wrapper = SkforecastWrapper(config)

        target_data = pd.DataFrame(
            {
                "A": [10, 20, 30, 40, 50],
            },
            index=pd.date_range("2023-01", periods=5, freq="MS"),
        )

        wrapper.fit(target_data, exog_wide=None)

        samples = wrapper.predict_samples(steps=3, exog_future=None, n_samples=20)

        assert "A" in samples
        assert samples["A"].shape == (3, 20)

    def test_predict_before_fit_raises_error(self) -> None:
        """Test that prediction before fit raises an error."""
        config = SkforecastConfig(lags=2)
        wrapper = SkforecastWrapper(config)

        with pytest.raises(RuntimeError, match="Forecaster must be fitted"):
            wrapper.predict_samples(steps=1, exog_future=None, n_samples=10)

    def test_different_model_class(self) -> None:
        """Test using a different sklearn model class."""
        config = SkforecastConfig(lags=2, model_class="sklearn.linear_model.Ridge", model_params={"alpha": 1.0})

        wrapper = SkforecastWrapper(config)

        target_data = pd.DataFrame(
            {
                "A": [10, 20, 30, 40, 50],
            },
            index=pd.date_range("2023-01", periods=5, freq="MS"),
        )

        wrapper.fit(target_data, exog_wide=None)

        assert wrapper.forecaster is not None
        assert wrapper.forecaster.is_fitted

    def test_list_lags(self) -> None:
        """Test using list of lags."""
        config = SkforecastConfig(lags=[1, 3, 6], model_params={"n_estimators": 10, "random_state": 42})

        wrapper = SkforecastWrapper(config)

        target_data = pd.DataFrame(
            {
                "A": [10, 20, 30, 40, 50, 60, 70, 80],
            },
            index=pd.date_range("2023-01", periods=8, freq="MS"),
        )

        wrapper.fit(target_data, exog_wide=None)

        assert wrapper.forecaster is not None
        assert wrapper.forecaster.is_fitted
