"""CLI adapter for creating command-line interfaces from train/predict functions."""

import asyncio
import pickle
from pathlib import Path
from typing import Annotated, Any, Callable

import cyclopts
import pandas as pd  # type: ignore[import-untyped]


def create_cli_app(
    train_func: Callable[..., Any],
    predict_func: Callable[..., Any],
    config_class: type[Any],
    name: str = "chap-model",
) -> cyclopts.App:
    """Create a cyclopts CLI app with train and predict subcommands.

    Args:
        train_func: Training function with signature (config, data, ...) -> model
        predict_func: Prediction function with signature (config, model, historic, future, ...) -> predictions
        config_class: Configuration class with no-arg constructor
        name: Name of the CLI application

    Returns:
        Configured cyclopts.App with train-cmd and predict-cmd subcommands

    Example:
        >>> from my_model import train, predict, MyConfig
        >>> app = create_cli_app(train, predict, MyConfig, name="my-model")
        >>> app()  # Run CLI
    """
    app = cyclopts.App(
        name=name,
        help=f"{name} CHAP model CLI",
    )

    @app.command
    def train_cmd(  # pyright: ignore[reportUnusedFunction]
        train_data: Annotated[Path, cyclopts.Parameter(help="Path to training data CSV")],
        model_output: Annotated[Path, cyclopts.Parameter(help="Path to save trained model (pickle)")],
    ) -> None:
        """Train the model."""
        # Load data from CSV
        data = pd.read_csv(train_data)

        # Create config
        config = config_class()

        # Train model (handle async if needed)
        if asyncio.iscoroutinefunction(train_func):
            model = asyncio.run(train_func(config, data))
        else:
            model = train_func(config, data)

        # Save model
        with open(model_output, "wb") as f:
            pickle.dump(model, f)

        print(f"Model trained and saved to {model_output}")

    @app.command
    def predict_cmd(  # pyright: ignore[reportUnusedFunction]
        model_path: Annotated[Path, cyclopts.Parameter(help="Path to trained model (pickle)")],
        historic_data: Annotated[Path, cyclopts.Parameter(help="Path to historic data CSV")],
        future_data: Annotated[Path, cyclopts.Parameter(help="Path to future periods CSV")],
        output: Annotated[Path, cyclopts.Parameter(help="Path to save predictions CSV")],
    ) -> None:
        """Generate predictions."""
        # Load model
        with open(model_path, "rb") as f:
            model: Any = pickle.load(f)

        # Load data
        historic = pd.read_csv(historic_data)
        future = pd.read_csv(future_data)

        # Create config
        config = config_class()

        # Generate predictions (handle async if needed)
        if asyncio.iscoroutinefunction(predict_func):
            predictions = asyncio.run(predict_func(config, model, historic, future))
        else:
            predictions = predict_func(config, model, historic, future)

        # Convert nested samples column to wide format (sample_0, sample_1, ...)
        if "samples" in predictions.columns:
            samples_list = predictions["samples"].tolist()
            predictions = predictions.drop(columns=["samples"])
            if samples_list:
                n_samples = len(samples_list[0])
                for i in range(n_samples):
                    predictions[f"sample_{i}"] = [row[i] for row in samples_list]

        # Save predictions
        predictions.to_csv(output, index=False)

        print(f"Predictions saved to {output}")

    return app
