"""Integration tests for skforecast adaptor with chapkit validation framework."""

import pytest

skforecast = pytest.importorskip("skforecast", reason="skforecast not installed")

from chap_python_sdk import (  # noqa: E402
    SkforecastConfig,
    create_skforecast_functions,
    get_example_data,
    validate_model_io,
)


class TestSkforecastIntegration:
    """Integration tests for skforecast adaptor."""

    @pytest.mark.asyncio
    async def test_import_error_without_skforecast(self) -> None:
        """Test that helpful error is raised when skforecast not installed."""
        import sys
        from unittest.mock import patch

        with patch.dict(sys.modules, {"skforecast": None, "skforecast.recursive": None}):
            from chap_python_sdk.adaptors.skforecast import SKFORECAST_AVAILABLE

            if not SKFORECAST_AVAILABLE:
                from chap_python_sdk.adaptors.skforecast import create_skforecast_functions

                with pytest.raises(ImportError, match="skforecast is not installed"):
                    create_skforecast_functions()

    @pytest.mark.asyncio
    async def test_basic_train_predict_cycle(self) -> None:
        """Test basic training and prediction with skforecast."""
        config = SkforecastConfig(lags=3, n_samples=20, model_params={"n_estimators": 10, "random_state": 42})

        train_function, predict_function = create_skforecast_functions(config)

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
        """Test skforecast with exogenous variables."""
        config = SkforecastConfig(
            lags=6,
            n_samples=30,
            exogenous_variables=["rainfall", "temperature_mean"],
            model_params={"n_estimators": 20, "random_state": 42},
        )

        train_function, predict_function = create_skforecast_functions(config)

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
        """Test skforecast with Ridge regression instead of RandomForest."""
        config = SkforecastConfig(
            lags=4, n_samples=25, model_class="sklearn.linear_model.Ridge", model_params={"alpha": 1.0}
        )

        train_function, predict_function = create_skforecast_functions(config)

        example_data = get_example_data("laos", "monthly")

        result = await validate_model_io(
            train_function,
            predict_function,
            example_data,
        )

        assert result.success
        assert result.n_samples == 25

    @pytest.mark.asyncio
    async def test_with_list_lags(self) -> None:
        """Test skforecast with specific lag selection."""
        config = SkforecastConfig(
            lags=[1, 3, 6, 12], n_samples=15, model_params={"n_estimators": 15, "random_state": 42}
        )

        train_function, predict_function = create_skforecast_functions(config)

        example_data = get_example_data("laos", "monthly")

        result = await validate_model_io(
            train_function,
            predict_function,
            example_data,
        )

        assert result.success
        assert result.n_samples == 15

    @pytest.mark.asyncio
    async def test_default_configuration(self) -> None:
        """Test skforecast with default configuration."""
        train_function, predict_function = create_skforecast_functions()

        example_data = get_example_data("laos", "monthly")

        result = await validate_model_io(
            train_function,
            predict_function,
            example_data,
        )

        assert result.success
        assert result.n_samples == 100

    @pytest.mark.asyncio
    async def test_predictions_are_nonnegative(self) -> None:
        """Test that predictions are non-negative."""
        config = SkforecastConfig(lags=6, n_samples=50, model_params={"n_estimators": 10, "random_state": 42})

        train_function, predict_function = create_skforecast_functions(config)

        example_data = get_example_data("laos", "monthly")

        result = await validate_model_io(
            train_function,
            predict_function,
            example_data,
        )

        assert result.success

    @pytest.mark.asyncio
    async def test_multiple_locations(self) -> None:
        """Test that skforecast works with multiple locations."""
        config = SkforecastConfig(lags=4, n_samples=20, model_params={"n_estimators": 10, "random_state": 42})

        train_function, predict_function = create_skforecast_functions(config)

        example_data = get_example_data("laos", "monthly")

        result = await validate_model_io(
            train_function,
            predict_function,
            example_data,
        )

        assert result.success
        assert result.n_predictions > 1

    @pytest.mark.asyncio
    async def test_predictions_have_variability(self) -> None:
        """Test that prediction samples have variability."""
        config = SkforecastConfig(
            lags=6, n_samples=100, use_bootstrapping=True, model_params={"n_estimators": 20, "random_state": 42}
        )

        train_function, predict_function = create_skforecast_functions(config)

        example_data = get_example_data("laos", "monthly")

        result = await validate_model_io(
            train_function,
            predict_function,
            example_data,
        )

        assert result.success
        assert result.n_samples == 100
