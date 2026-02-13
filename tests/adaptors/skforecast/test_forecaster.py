"""Tests for forecaster wrapper."""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

skforecast = pytest.importorskip("skforecast", reason="skforecast not installed")

from chap_python_sdk.adaptors.skforecast import SkforecastConfig  # noqa: E402
from chap_python_sdk.adaptors.skforecast.forecaster import (  # noqa: E402
    SkforecastWrapper,
    _resolve_transformer,  # pyright: ignore[reportPrivateUsage]
)


def _make_target(n_rows: int = 18, locations: list[str] | None = None) -> pd.DataFrame:
    """Create a target DataFrame with enough rows for backtesting."""
    if locations is None:
        locations = ["A", "B"]
    rng = np.random.default_rng(42)
    data = {loc: rng.poisson(50, size=n_rows).astype(float) for loc in locations}
    return pd.DataFrame(data, index=pd.date_range("2022-01", periods=n_rows, freq="MS"))


class TestResolveTransformer:
    """Tests for _resolve_transformer helper."""

    def test_standard_scaler(self) -> None:
        """Test resolving StandardScaler."""
        from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

        t = _resolve_transformer("StandardScaler")
        assert isinstance(t, StandardScaler)

    def test_minmax_scaler(self) -> None:
        """Test resolving MinMaxScaler."""
        from sklearn.preprocessing import MinMaxScaler  # pyright: ignore[reportMissingTypeStubs]

        t = _resolve_transformer("MinMaxScaler")
        assert isinstance(t, MinMaxScaler)

    def test_none_returns_none(self) -> None:
        """Test that None input returns None."""
        assert _resolve_transformer(None) is None

    def test_unknown_raises(self) -> None:
        """Test that unknown transformer name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown transformer"):
            _resolve_transformer("FooScaler")


class TestSkforecastWrapper:
    """Tests for SkforecastWrapper class."""

    def test_fit_basic(self) -> None:
        """Test basic fitting of the forecaster."""
        config = SkforecastConfig(
            lags=2,
            differentiation=None,
            transformer_series=None,
            use_bootstrapping=False,
        )
        wrapper = SkforecastWrapper(config)
        target_data = _make_target(n_rows=8)

        wrapper.fit(target_data, exog_wide=None)

        assert wrapper.forecaster is not None
        assert wrapper.forecaster.is_fitted

    def test_fit_with_differentiation_and_transformer(self) -> None:
        """Test fitting with differentiation and transformer enabled."""
        config = SkforecastConfig(lags=2, use_bootstrapping=False)
        wrapper = SkforecastWrapper(config)
        target_data = _make_target(n_rows=12)

        wrapper.fit(target_data, exog_wide=None)

        assert wrapper.forecaster is not None
        assert wrapper.forecaster.is_fitted

    def test_fit_with_exogenous(self) -> None:
        """Test fitting with exogenous variables."""
        config = SkforecastConfig(
            lags=2,
            exogenous_variables=["rainfall"],
            differentiation=None,
            transformer_series=None,
            use_bootstrapping=False,
        )
        wrapper = SkforecastWrapper(config)

        target_data = _make_target(n_rows=8, locations=["A"])
        exog_data = pd.DataFrame(
            {"rainfall_A": np.random.default_rng(0).normal(100, 20, 8)},
            index=target_data.index,
        )

        wrapper.fit(target_data, exog_wide=exog_data)

        assert wrapper.forecaster is not None
        assert wrapper.forecaster.is_fitted

    def test_residuals_by_step_computed(self) -> None:
        """Test that multi-step residuals are computed during fit."""
        config = SkforecastConfig(
            lags=2,
            use_bootstrapping=True,
            n_prediction_steps=2,
            differentiation=None,
            transformer_series=None,
        )
        wrapper = SkforecastWrapper(config)
        target_data = _make_target(n_rows=18)

        wrapper.fit(target_data, exog_wide=None)

        assert "A" in wrapper.residuals_by_step
        assert "B" in wrapper.residuals_by_step
        # Each location should have step 0 and step 1
        assert 0 in wrapper.residuals_by_step["A"]
        assert 1 in wrapper.residuals_by_step["A"]
        assert len(wrapper.residuals_by_step["A"][0]) > 0

    def test_predict_samples(self) -> None:
        """Test generating prediction samples."""
        config = SkforecastConfig(
            lags=2,
            n_samples=20,
            n_prediction_steps=2,
            differentiation=None,
            transformer_series=None,
        )
        wrapper = SkforecastWrapper(config)
        target_data = _make_target(n_rows=18, locations=["A"])

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
        config = SkforecastConfig(
            lags=2,
            model_class="sklearn.linear_model.Ridge",
            model_params={"alpha": 1.0},
            differentiation=None,
            transformer_series=None,
            use_bootstrapping=False,
        )
        wrapper = SkforecastWrapper(config)
        target_data = _make_target(n_rows=8, locations=["A"])

        wrapper.fit(target_data, exog_wide=None)

        assert wrapper.forecaster is not None
        assert wrapper.forecaster.is_fitted

    def test_list_lags(self) -> None:
        """Test using list of lags."""
        config = SkforecastConfig(
            lags=[1, 3, 6],
            differentiation=None,
            transformer_series=None,
            use_bootstrapping=False,
        )
        wrapper = SkforecastWrapper(config)
        target_data = _make_target(n_rows=12, locations=["A"])

        wrapper.fit(target_data, exog_wide=None)

        assert wrapper.forecaster is not None
        assert wrapper.forecaster.is_fitted

    def test_refit(self) -> None:
        """Test refitting the forecaster on new data."""
        config = SkforecastConfig(
            lags=2,
            differentiation=None,
            transformer_series=None,
            use_bootstrapping=False,
        )
        wrapper = SkforecastWrapper(config)
        target_data = _make_target(n_rows=8)
        wrapper.fit(target_data, exog_wide=None)

        # Refit with different data
        new_data = _make_target(n_rows=10)
        wrapper.refit(new_data, exog_wide=None)

        assert wrapper.forecaster is not None
        assert wrapper.forecaster.is_fitted
