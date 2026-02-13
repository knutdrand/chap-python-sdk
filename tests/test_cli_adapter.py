"""Tests for CLI adapter."""

import pickle
import tempfile
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pytest
from pydantic import BaseModel, Field

from chap_python_sdk.cli_adapter import create_cli_app


class TestConfig(BaseModel):
    """Test configuration."""

    n_samples: int = Field(default=10)


def mock_train(
    config: TestConfig,
    data: pd.DataFrame,
) -> dict[str, float]:
    """Mock training function."""
    mean_value = data["value"].mean()
    return {"mean": float(mean_value)}


def mock_predict(
    config: TestConfig,
    model: dict[str, float],
    historic: pd.DataFrame,
    future: pd.DataFrame,
) -> pd.DataFrame:
    """Mock prediction function."""
    predictions = future.copy()
    predictions["samples"] = [[model["mean"]] * config.n_samples for _ in range(len(future))]
    return predictions


def test_create_cli_app() -> None:
    """Test that create_cli_app returns a cyclopts App."""
    app = create_cli_app(
        train_func=mock_train,
        predict_func=mock_predict,
        config_class=TestConfig,
        name="test-model",
    )

    assert app is not None
    assert hasattr(app, "__call__")


def test_cli_train_predict_integration() -> None:
    """Test full train and predict workflow via CLI."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create test data CSV files
        train_csv = tmpdir_path / "train.csv"
        train_data = pd.DataFrame(
            {
                "time_period": [1, 2, 3],
                "location": ["A", "A", "A"],
                "value": [10.0, 20.0, 30.0],
            }
        )
        train_data.to_csv(train_csv, index=False)

        future_csv = tmpdir_path / "future.csv"
        future_data = pd.DataFrame(
            {
                "time_period": [4, 5],
                "location": ["A", "A"],
            }
        )
        future_data.to_csv(future_csv, index=False)

        historic_csv = tmpdir_path / "historic.csv"
        historic_data = pd.DataFrame(
            {
                "time_period": [1, 2, 3],
                "location": ["A", "A", "A"],
                "value": [10.0, 20.0, 30.0],
            }
        )
        historic_data.to_csv(historic_csv, index=False)

        model_pkl = tmpdir_path / "model.pkl"
        predictions_csv = tmpdir_path / "predictions.csv"

        # Create CLI app
        app = create_cli_app(
            train_func=mock_train,
            predict_func=mock_predict,
            config_class=TestConfig,
            name="test-model",
        )

        # Simulate CLI train command
        command, bound, _ = app.parse_args(["train-cmd", str(train_csv), str(model_pkl)])
        command(*bound.args, **bound.kwargs)

        # Verify model was saved
        assert model_pkl.exists()
        with open(model_pkl, "rb") as f:
            model = pickle.load(f)
        assert "mean" in model
        assert model["mean"] == pytest.approx(20.0)

        # Simulate CLI predict command
        command, bound, _ = app.parse_args(
            ["predict-cmd", str(model_pkl), str(historic_csv), str(future_csv), str(predictions_csv)]
        )
        command(*bound.args, **bound.kwargs)

        # Verify predictions were saved
        assert predictions_csv.exists()
        predictions = pd.read_csv(predictions_csv)
        # samples column is expanded to sample_0, sample_1, ... in CSV
        assert any(col.startswith("sample_") for col in predictions.columns)
        assert len(predictions) == 2
