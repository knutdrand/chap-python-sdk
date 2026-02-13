"""Probabilistic sample generation via bootstrap residuals."""

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]


def bootstrap_recursive_samples(
    forecaster: Any,
    residuals_by_step: dict[str, dict[int, np.ndarray]],
    n_steps: int,
    n_samples: int,
    exog_future: pd.DataFrame | None,
    locations: list[str],
) -> dict[str, pd.DataFrame]:
    """Generate bootstrap samples for multiple locations.

    For each sample trajectory:
    1. Get point predictions for all steps
    2. For each step k, sample a residual from step-specific residuals
    3. Add residual to point prediction (clamped to >= 0)

    Args:
        forecaster: Fitted ForecasterRecursiveMultiSeries instance
        residuals_by_step: Dict mapping location -> step -> array of residuals
        n_steps: Number of steps to predict
        n_samples: Number of sample trajectories to generate
        exog_future: Future exogenous variables (or None)
        locations: List of location names

    Returns:
        Dict mapping location -> DataFrame of shape (n_steps, n_samples)
    """
    # Get point predictions
    point_preds = forecaster.predict(steps=n_steps, exog=exog_future)

    # skforecast returns predictions in long format with 'level' and 'pred' columns
    predictions_by_location: dict[str, np.ndarray] = {}
    for location in locations:
        location_preds = point_preds[point_preds["level"] == location]["pred"].values
        predictions_by_location[location] = location_preds

    # Initialize result storage
    samples_by_location: dict[str, pd.DataFrame] = {loc: pd.DataFrame() for loc in locations}

    for sample_idx in range(n_samples):
        for location in locations:
            loc_residuals = residuals_by_step.get(location, {})
            location_preds = predictions_by_location[location]
            trajectory = np.empty(n_steps)

            for step in range(n_steps):
                # Use step-specific residuals; fall back to max available step
                if step in loc_residuals:
                    res = loc_residuals[step]
                elif loc_residuals:
                    max_step = max(loc_residuals.keys())
                    res = loc_residuals[max_step]
                else:
                    res = np.array([0.0])

                # Guard against empty arrays
                if len(res) == 0:
                    res = np.array([0.0])

                residual = np.random.choice(res)
                trajectory[step] = max(location_preds[step] + residual, 0)

            samples_by_location[location][sample_idx] = trajectory

    return samples_by_location
