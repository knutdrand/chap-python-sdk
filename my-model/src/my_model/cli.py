"""CLI for my-model model."""

import asyncio
import pickle
from pathlib import Path
from typing import Annotated

import cyclopts
from chapkit.data import DataFrame

from my_model.model import MyModelConfig, predict, train

app = cyclopts.App(
    name="my-model",
    help="my-model CHAP model CLI",
)


@app.command
def train_cmd(
    train_data: Annotated[Path, cyclopts.Parameter(help="Path to training data CSV")],
    model_output: Annotated[Path, cyclopts.Parameter(help="Path to save trained model (pickle)")],
) -> None:
    """Train the model."""
    # Load data from CSV
    data = DataFrame.from_csv(str(train_data))

    # Create config
    config = MyModelConfig()

    # Train model
    model = asyncio.run(train(config, data))

    # Save model
    with open(model_output, "wb") as f:
        pickle.dump(model, f)

    print(f"Model trained and saved to {model_output}")


@app.command
def predict_cmd(
    model_path: Annotated[Path, cyclopts.Parameter(help="Path to trained model (pickle)")],
    historic_data: Annotated[Path, cyclopts.Parameter(help="Path to historic data CSV")],
    future_data: Annotated[Path, cyclopts.Parameter(help="Path to future periods CSV")],
    output: Annotated[Path, cyclopts.Parameter(help="Path to save predictions CSV")],
) -> None:
    """Generate predictions."""
    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Load data
    historic = DataFrame.from_csv(str(historic_data))
    future = DataFrame.from_csv(str(future_data))

    # Create config
    config = MyModelConfig()

    # Generate predictions
    predictions = asyncio.run(predict(config, model, historic, future))

    # Save predictions
    predictions.to_csv(str(output))

    print(f"Predictions saved to {output}")


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
