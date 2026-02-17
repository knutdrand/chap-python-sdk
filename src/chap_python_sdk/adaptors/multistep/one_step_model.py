"""Residual bootstrap one-step model wrapping any sklearn regressor."""

import importlib

import numpy as np


def _import_class(class_path: str) -> type:
    """Dynamically import a class from a string path."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)  # type: ignore[no-any-return]


class ResidualDistribution:
    """Point predictions plus resampled residuals."""

    def __init__(self, predictions: np.ndarray, residuals: np.ndarray) -> None:
        """Store predictions and training residuals.

        Args:
            predictions: Point predictions, shape (n_rows,).
            residuals: Training residuals for resampling, shape (n_train,).
        """
        self._predictions = predictions
        self._residuals = residuals

    def sample(self, n_samples: int) -> np.ndarray:
        """Draw samples by adding resampled residuals to predictions.

        Args:
            n_samples: Number of samples to draw.

        Returns:
            Shape (n_samples, n_rows), clamped to >= 0.
        """
        rng = np.random.default_rng()
        n_rows = len(self._predictions)
        drawn = rng.choice(self._residuals, size=(n_samples, n_rows), replace=True)
        samples = self._predictions[np.newaxis, :] + drawn
        result: np.ndarray = np.maximum(samples, 0.0)
        return result


class ResidualBootstrapModel:
    """One-step model wrapping any sklearn regressor with residual bootstrapping."""

    def __init__(self, model_class: str, model_params: dict[str, object]) -> None:
        """Create regressor from class path and params.

        Args:
            model_class: Dotted path to sklearn class (e.g. "sklearn.ensemble.GradientBoostingRegressor").
            model_params: Keyword arguments passed to the regressor constructor.
        """
        cls = _import_class(model_class)
        self._regressor = cls(**model_params)
        self._residuals: np.ndarray = np.array([0.0])

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit regressor and store training residuals.

        Args:
            X: Feature matrix, shape (n_samples, n_features).
            y: Target values, shape (n_samples,).
        """
        self._regressor.fit(X, y)
        predictions = self._regressor.predict(X)
        self._residuals = y - predictions

    def predict_proba(self, X: np.ndarray) -> ResidualDistribution:
        """Return a ResidualDistribution over next-step values.

        Args:
            X: Feature matrix, shape (n_rows, n_features).

        Returns:
            ResidualDistribution for sampling.
        """
        predictions = self._regressor.predict(X)
        return ResidualDistribution(predictions, self._residuals)
