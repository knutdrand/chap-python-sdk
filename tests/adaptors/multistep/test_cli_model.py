"""Tests for multistep CLI model."""

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from chap_python_sdk.adaptors.multistep.cli_model import (
    create_multistep_cli_app,
    predict,
    train,
)
from chap_python_sdk.adaptors.multistep.config import MultistepConfig


def _make_training_data(n_times: int = 24, locations: list[str] | None = None) -> pd.DataFrame:
    """Create synthetic training data with enough time points for lag features."""
    if locations is None:
        locations = ["loc_A", "loc_B"]
    rng = np.random.default_rng(42)
    rows = []
    for loc in locations:
        for t in range(n_times):
            year = 2020 + t // 12
            month = (t % 12) + 1
            rows.append(
                {
                    "time_period": f"{year}-{month:02d}-01",
                    "location": loc,
                    "disease_cases": float(rng.poisson(50)),
                }
            )
    return pd.DataFrame(rows)


def _make_training_data_with_exog(
    n_times: int = 24,
    locations: list[str] | None = None,
) -> pd.DataFrame:
    """Create synthetic training data with exogenous variables."""
    df = _make_training_data(n_times=n_times, locations=locations)
    rng = np.random.default_rng(123)
    df["rainfall"] = rng.uniform(0, 100, size=len(df))
    df["mean_temperature"] = rng.uniform(15, 35, size=len(df))
    return df


def _make_future_data(locations: list[str] | None = None, n_steps: int = 3) -> pd.DataFrame:
    """Create future periods DataFrame."""
    if locations is None:
        locations = ["loc_A", "loc_B"]
    rows = []
    for loc in locations:
        for t in range(n_steps):
            rows.append(
                {
                    "time_period": f"2022-{t + 1:02d}-01",
                    "location": loc,
                }
            )
    return pd.DataFrame(rows)


def _make_future_data_with_exog(
    locations: list[str] | None = None,
    n_steps: int = 3,
) -> pd.DataFrame:
    """Create future periods DataFrame with exogenous variables."""
    df = _make_future_data(locations=locations, n_steps=n_steps)
    rng = np.random.default_rng(456)
    df["rainfall"] = rng.uniform(0, 100, size=len(df))
    df["mean_temperature"] = rng.uniform(15, 35, size=len(df))
    return df


def test_create_cli_app() -> None:
    """Test that create_multistep_cli_app returns a cyclopts App."""
    app = create_multistep_cli_app()
    assert app is not None
    assert hasattr(app, "__call__")


def test_train_predict_roundtrip() -> None:
    """Test full train → predict roundtrip via CLI parse_args."""
    # CLI uses default config (n_target_lags=12, n_samples=200),
    # so we need enough time points to have samples after lag trimming.
    train_data = _make_training_data(n_times=24)
    historic_data = _make_training_data(n_times=24)
    future_data = _make_future_data(n_steps=3)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Write CSV files
        train_csv = tmpdir_path / "train.csv"
        train_data.to_csv(train_csv, index=False)

        historic_csv = tmpdir_path / "historic.csv"
        historic_data.to_csv(historic_csv, index=False)

        future_csv = tmpdir_path / "future.csv"
        future_data.to_csv(future_csv, index=False)

        model_pkl = tmpdir_path / "model.pkl"
        predictions_csv = tmpdir_path / "predictions.csv"

        app = create_multistep_cli_app()

        # Train via CLI
        command, bound, _ = app.parse_args(["train-cmd", str(train_csv), str(model_pkl)])
        command(*bound.args, **bound.kwargs)
        assert model_pkl.exists()

        # Predict via CLI
        command, bound, _ = app.parse_args(
            ["predict-cmd", str(model_pkl), str(historic_csv), str(future_csv), str(predictions_csv)]
        )
        command(*bound.args, **bound.kwargs)
        assert predictions_csv.exists()

        # Verify predictions CSV has sample columns (default n_samples=200)
        predictions = pd.read_csv(predictions_csv)
        sample_cols = [c for c in predictions.columns if c.startswith("sample_")]
        assert len(sample_cols) == 200
        assert "time_period" in predictions.columns
        assert "location" in predictions.columns
        # 2 locations × 3 steps = 6 rows
        assert len(predictions) == 6


def test_train_returns_pickleable_model() -> None:
    """Test that the trained model dict is pickleable."""
    config = MultistepConfig(n_target_lags=4, n_samples=5)
    data = _make_training_data(n_times=12)

    model = train(config, data)

    # Verify pickling round-trip
    serialized = pickle.dumps(model)
    restored = pickle.loads(serialized)  # noqa: S301

    assert "multistep_model" in restored
    assert "locations" in restored
    assert "config" in restored
    assert set(restored["locations"]) == {"loc_A", "loc_B"}


def test_predict_with_exogenous() -> None:
    """Test train/predict with exogenous variables (rainfall, mean_temperature)."""
    config = MultistepConfig(
        n_target_lags=4,
        n_samples=10,
        exogenous_variables=["rainfall", "mean_temperature"],
    )
    train_data = _make_training_data_with_exog(n_times=12)
    historic_data = _make_training_data_with_exog(n_times=12)
    future_data = _make_future_data_with_exog(n_steps=3)

    model = train(config, train_data)
    result = predict(config, model, historic_data, future_data)

    assert "samples" in result.columns
    assert "time_period" in result.columns
    assert "location" in result.columns
    assert len(result) == 6  # 2 locations × 3 steps

    # Each row should have 10 samples
    for samples in result["samples"]:
        assert len(samples) == 10
        assert all(isinstance(s, float) for s in samples)
