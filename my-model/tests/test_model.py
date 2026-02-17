"""Tests for my-model model."""

import pytest

from chap_python_sdk.testing import get_example_data, validate_model_io

from my_model.model import MyModelConfig, predict, train


@pytest.mark.asyncio
async def test_model_with_laos_monthly_data() -> None:
    """Test model against Laos monthly example data."""
    example_data = get_example_data(country="laos", frequency="monthly")
    config = MyModelConfig()

    result = await validate_model_io(train, predict, example_data, config)

    assert result.success, f"Validation failed: {result.errors}"
    assert result.n_predictions > 0
    assert result.n_samples >= 1


@pytest.mark.asyncio
async def test_model_predictions_shape() -> None:
    """Test that predictions have the correct shape."""
    example_data = get_example_data(country="laos", frequency="monthly")
    config = MyModelConfig()

    # Train the model
    model = await train(config, example_data.training_data, example_data.run_info)

    # Generate predictions
    predictions = await predict(
        config,
        model,
        example_data.historic_data,
        example_data.future_data,
        example_data.run_info,
    )

    # Check predictions shape matches future data
    assert len(predictions) == len(example_data.future_data)
    assert "time_period" in predictions.columns
    assert "location" in predictions.columns
    assert "samples" in predictions.columns
