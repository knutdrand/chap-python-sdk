"""Probabilistic sample generation via bootstrap residuals."""

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]


def bootstrap_recursive_samples(
    forecaster: Any,
    residuals_by_location: dict[str, np.ndarray],
    n_steps: int,
    n_samples: int,
    exog_future: pd.DataFrame | None,
    locations: list[str],
) -> dict[str, pd.DataFrame]:
    """Generate bootstrap samples for multiple locations.

    For each sample trajectory:
    1. Start with last known values
    2. For each prediction step:
       - Predict point estimate
       - Sample residual from training residuals
       - Add residual to prediction (this becomes the actual sampled value)
       - Use sampled value as input for next recursive step
    3. Store trajectory

    Args:
        forecaster: Fitted ForecasterRecursiveMultiSeries instance
        residuals_by_location: Dict mapping location -> array of training residuals
        n_steps: Number of steps to predict
        n_samples: Number of sample trajectories to generate
        exog_future: Future exogenous variables (or None)
        locations: List of location names

    Returns:
        Dict mapping location -> DataFrame of shape (n_steps, n_samples)
    """
    # Get point predictions first
    point_preds = forecaster.predict(steps=n_steps, exog=exog_future)

    # skforecast returns predictions in long format with 'level' and 'pred' columns
    # Convert to dict of arrays for easier processing
    predictions_by_location = {}
    for location in locations:
        location_preds = point_preds[point_preds["level"] == location]["pred"].values
        predictions_by_location[location] = location_preds

    # Initialize result storage
    samples_by_location = {loc: pd.DataFrame() for loc in locations}

    # Generate samples
    for sample_idx in range(n_samples):
        # For each location, sample residuals and add to point predictions
        for location in locations:
            residuals = residuals_by_location.get(location, np.array([0.0]))

            # Sample residuals for all steps at once
            sampled_residuals = np.random.choice(residuals, size=n_steps, replace=True)

            # Add residuals to point predictions
            location_preds = predictions_by_location[location]
            sampled_trajectory = location_preds + sampled_residuals

            # Ensure non-negative predictions for disease cases
            sampled_trajectory = np.maximum(sampled_trajectory, 0)

            # Store sample
            samples_by_location[location][sample_idx] = sampled_trajectory

    return samples_by_location
