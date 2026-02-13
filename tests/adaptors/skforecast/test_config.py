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
        assert config.n_samples == 200
        assert config.use_bootstrapping is True
        assert config.exogenous_variables is None
        assert config.model_class == "sklearn.ensemble.GradientBoostingRegressor"
        assert config.model_params["n_estimators"] == 100
        assert config.model_params["max_depth"] == 3
        assert config.model_params["learning_rate"] == 0.1
        assert config.model_params["min_samples_leaf"] == 3
        assert config.model_params["random_state"] == 42
        assert config.encoding == "onehot"
        assert config.differentiation == 1
        assert config.transformer_series == "StandardScaler"
        assert config.refit_on_predict is True
        assert config.n_prediction_steps == 3

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

    def test_differentiation_none(self) -> None:
        """Test disabling differentiation."""
        config = SkforecastConfig(differentiation=None)

        assert config.differentiation is None

    def test_differentiation_custom(self) -> None:
        """Test custom differentiation order."""
        config = SkforecastConfig(differentiation=2)

        assert config.differentiation == 2

    def test_transformer_series_none(self) -> None:
        """Test disabling transformer."""
        config = SkforecastConfig(transformer_series=None)

        assert config.transformer_series is None

    def test_transformer_series_custom(self) -> None:
        """Test custom transformer name."""
        config = SkforecastConfig(transformer_series="MinMaxScaler")

        assert config.transformer_series == "MinMaxScaler"

    def test_refit_on_predict_disabled(self) -> None:
        """Test disabling refit on predict."""
        config = SkforecastConfig(refit_on_predict=False)

        assert config.refit_on_predict is False

    def test_n_prediction_steps_custom(self) -> None:
        """Test custom prediction steps."""
        config = SkforecastConfig(n_prediction_steps=5)

        assert config.n_prediction_steps == 5
