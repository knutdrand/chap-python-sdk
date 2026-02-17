"""Integration tests for multistep adaptor with chapkit validation framework."""

import pytest

sklearn = pytest.importorskip("sklearn", reason="scikit-learn not installed")

from chap_python_sdk import (  # noqa: E402
    MultistepConfig,
    create_multistep_functions,
    get_example_data,
    validate_model_io,
)


class TestMultistepIntegration:
    """Integration tests for multistep adaptor."""

    @pytest.mark.asyncio
    async def test_import_error_without_sklearn(self) -> None:
        """Test that helpful error is raised when scikit-learn not installed."""
        import sys
        from unittest.mock import patch

        with patch.dict(sys.modules, {"sklearn": None, "sklearn.base": None}):
            from chap_python_sdk.adaptors.multistep import SKLEARN_AVAILABLE

            if not SKLEARN_AVAILABLE:
                from chap_python_sdk.adaptors.multistep import create_multistep_functions

                with pytest.raises(ImportError, match="scikit-learn is not installed"):
                    create_multistep_functions()

    @pytest.mark.asyncio
    async def test_basic_train_predict_cycle(self) -> None:
        """Test basic training and prediction with multistep model."""
        config = MultistepConfig(
            n_target_lags=3,
            n_samples=20,
            model_params={"n_estimators": 10, "random_state": 42},
        )

        train_function, predict_function = create_multistep_functions(config)

        example_data = get_example_data("laos", "monthly")

        result = await validate_model_io(
            train_function,
            predict_function,
            example_data,
        )

        assert result.success
        assert result.n_samples == 20

    @pytest.mark.asyncio
    async def test_with_exogenous_variables(self) -> None:
        """Test multistep model with exogenous variables."""
        config = MultistepConfig(
            n_target_lags=3,
            n_samples=30,
            exogenous_variables=["rainfall", "mean_temperature"],
            model_params={"n_estimators": 10, "random_state": 42},
        )

        train_function, predict_function = create_multistep_functions(config)

        example_data = get_example_data("laos", "monthly")

        result = await validate_model_io(
            train_function,
            predict_function,
            example_data,
        )

        assert result.success
        assert result.n_samples == 30

    @pytest.mark.asyncio
    async def test_with_ridge_regression(self) -> None:
        """Test multistep model with Ridge regression."""
        config = MultistepConfig(
            n_target_lags=4,
            n_samples=25,
            model_class="sklearn.linear_model.Ridge",
            model_params={"alpha": 1.0},
        )

        train_function, predict_function = create_multistep_functions(config)

        example_data = get_example_data("laos", "monthly")

        result = await validate_model_io(
            train_function,
            predict_function,
            example_data,
        )

        assert result.success
        assert result.n_samples == 25

    @pytest.mark.asyncio
    async def test_default_configuration(self) -> None:
        """Test multistep model with default configuration."""
        train_function, predict_function = create_multistep_functions()

        example_data = get_example_data("laos", "monthly")

        result = await validate_model_io(
            train_function,
            predict_function,
            example_data,
        )

        assert result.success
        assert result.n_samples == 200
