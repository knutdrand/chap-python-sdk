"""Tests for MultistepConfig."""

from chap_python_sdk.adaptors.multistep.config import MultistepConfig


class TestMultistepConfig:
    """Tests for MultistepConfig defaults and construction."""

    def test_default_values(self) -> None:
        """Default config has expected values."""
        config = MultistepConfig()
        assert config.n_target_lags == 12
        assert config.n_samples == 200
        assert config.model_class == "sklearn.ensemble.GradientBoostingRegressor"
        assert config.target_variable == "disease_cases"
        assert config.exogenous_variables is None

    def test_default_model_params(self) -> None:
        """Default model params match GBR defaults."""
        config = MultistepConfig()
        assert config.model_params["n_estimators"] == 100
        assert config.model_params["max_depth"] == 3
        assert config.model_params["learning_rate"] == 0.1
        assert config.model_params["min_samples_leaf"] == 3
        assert config.model_params["random_state"] == 42

    def test_custom_construction(self) -> None:
        """Custom values override defaults."""
        config = MultistepConfig(
            n_target_lags=6,
            n_samples=50,
            model_class="sklearn.linear_model.Ridge",
            model_params={"alpha": 1.0},
            exogenous_variables=["rainfall"],
            target_variable="cases",
        )
        assert config.n_target_lags == 6
        assert config.n_samples == 50
        assert config.model_class == "sklearn.linear_model.Ridge"
        assert config.model_params == {"alpha": 1.0}
        assert config.exogenous_variables == ["rainfall"]
        assert config.target_variable == "cases"

    def test_model_dump_roundtrip(self) -> None:
        """Config survives model_dump/reconstruction."""
        original = MultistepConfig(n_target_lags=8, n_samples=100)
        dumped = original.model_dump()
        restored = MultistepConfig(**dumped)
        assert restored.n_target_lags == original.n_target_lags
        assert restored.n_samples == original.n_samples
        assert restored.model_class == original.model_class
        assert restored.model_params == original.model_params

    def test_extra_fields_allowed(self) -> None:
        """BaseConfig allows extra fields."""
        config = MultistepConfig(custom_field="hello")  # type: ignore[call-arg]
        assert config.custom_field == "hello"  # type: ignore[attr-defined]
