"""Tests for skforecast configuration."""

import pytest

skforecast = pytest.importorskip("skforecast", reason="skforecast not installed")

from chap_python_sdk.adaptors.skforecast import SkforecastConfig  # noqa: E402


class TestSkforecastConfig:
    """Tests for SkforecastConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = SkforecastConfig()

        assert config.lags == 12
        assert config.n_samples == 100
        assert config.use_bootstrapping is True
        assert config.exogenous_variables is None
        assert config.model_class == "sklearn.ensemble.RandomForestRegressor"
        assert config.model_params == {}
        assert config.encoding == "onehot"

    def test_custom_lags_int(self) -> None:
        """Test configuration with integer lags."""
        config = SkforecastConfig(lags=6)

        assert config.lags == 6

    def test_custom_lags_list(self) -> None:
        """Test configuration with list of lags."""
        config = SkforecastConfig(lags=[1, 3, 6, 12])

        assert config.lags == [1, 3, 6, 12]

    def test_custom_n_samples(self) -> None:
        """Test configuration with custom sample count."""
        config = SkforecastConfig(n_samples=50)

        assert config.n_samples == 50

    def test_exogenous_variables(self) -> None:
        """Test configuration with exogenous variables."""
        config = SkforecastConfig(exogenous_variables=["rainfall", "temperature"])

        assert config.exogenous_variables == ["rainfall", "temperature"]

    def test_model_params(self) -> None:
        """Test configuration with model parameters."""
        config = SkforecastConfig(model_params={"n_estimators": 50, "max_depth": 10})

        assert config.model_params["n_estimators"] == 50
        assert config.model_params["max_depth"] == 10

    def test_different_model_class(self) -> None:
        """Test configuration with different sklearn model."""
        config = SkforecastConfig(model_class="sklearn.linear_model.Ridge")

        assert config.model_class == "sklearn.linear_model.Ridge"

    def test_encoding_options(self) -> None:
        """Test configuration with different encoding."""
        config = SkforecastConfig(encoding="ordinal")

        assert config.encoding == "ordinal"

    def test_disable_bootstrapping(self) -> None:
        """Test configuration with bootstrapping disabled."""
        config = SkforecastConfig(use_bootstrapping=False)

        assert config.use_bootstrapping is False
