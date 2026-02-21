"""Factory functions for sklearn transform pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore[import-untyped]
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import FunctionTransformer, StandardScaler  # type: ignore[import-untyped]

from .config import MultistepConfig


def build_target_pipeline(config: MultistepConfig) -> Pipeline:
    """Build sklearn Pipeline for target transforms (log1p -> standardize).

    Returns an identity pipeline if no target transforms are configured.
    """
    steps: list[tuple[str, FunctionTransformer | StandardScaler]] = []
    if config.log_transform_target:
        steps.append(("log", FunctionTransformer(func=np.log1p, inverse_func=np.expm1)))
    if config.standardize_target:
        steps.append(("scaler", StandardScaler()))
    if not steps:
        steps.append(("identity", FunctionTransformer()))
    return Pipeline(steps)


def build_feature_transformer(
    feature_cols: list[str],
    config: MultistepConfig,
) -> ColumnTransformer | FunctionTransformer:
    """Build sklearn transformer for covariate scaling.

    Returns an identity FunctionTransformer if standardize_covariates is False
    or no feature columns are provided.
    """
    if not config.standardize_covariates or not feature_cols:
        return FunctionTransformer()
    ct = ColumnTransformer(
        transformers=[("scaler", StandardScaler(), feature_cols)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    ct.set_output(transform="pandas")
    return ct


class FeatureLagger(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Sklearn-compatible transformer that adds lagged feature columns per location.

    For each feature column, adds ``{col}_lag1`` through ``{col}_lagN`` using
    ``groupby("location").shift(lag)``.  NaN rows produced by lagging are kept;
    the caller is responsible for masking them before fitting.
    """

    def __init__(self, n_lags: int, feature_cols: list[str]) -> None:
        """Initialize with lag count and feature column names."""
        self.n_lags = n_lags
        self.feature_cols = feature_cols

    def fit(self, X: pd.DataFrame, y: object = None) -> FeatureLagger:
        """Store the last ``n_lags`` rows per location as prediction context."""
        self.context_ = X.groupby("location").tail(self.n_lags).copy()
        return self

    def transform(self, X: pd.DataFrame, y: object = None) -> pd.DataFrame:
        """Add lagged columns.  NaN rows from insufficient history are kept."""
        result = X.copy()
        for col in self.feature_cols:
            for lag in range(1, self.n_lags + 1):
                result[f"{col}_lag{lag}"] = result.groupby("location")[col].shift(lag)
        return result

    @property
    def lag_columns(self) -> list[str]:
        """Return the list of lag column names produced by this transformer."""
        return [f"{col}_lag{lag}" for col in self.feature_cols for lag in range(1, self.n_lags + 1)]


def build_feature_lagger(
    feature_cols: list[str],
    config: MultistepConfig,
) -> FeatureLagger | FunctionTransformer:
    """Build a FeatureLagger or identity transformer.

    Returns an identity ``FunctionTransformer`` when ``n_feature_lags == 0``
    or no feature columns are provided.
    """
    if config.n_feature_lags <= 0 or not feature_cols:
        return FunctionTransformer()
    return FeatureLagger(n_lags=config.n_feature_lags, feature_cols=feature_cols)
