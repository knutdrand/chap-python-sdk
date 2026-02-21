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


class LocationEncoder(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Sklearn-compatible transformer that one-hot encodes the location column.

    Adds binary columns ``location_{name}`` for each location seen during fit,
    then drops the original ``location`` column.
    """

    def __init__(self, column: str = "location") -> None:
        """Initialize with the column name to encode."""
        self.column = column

    def fit(self, X: pd.DataFrame, y: object = None) -> LocationEncoder:
        """Learn unique location values."""
        self.categories_: list[str] = sorted(X[self.column].unique().tolist())
        return self

    def transform(self, X: pd.DataFrame, y: object = None) -> pd.DataFrame:
        """Add one-hot columns for each location, drop original column."""
        result = X.copy()
        for cat in self.categories_:
            result[f"{self.column}_{cat}"] = (result[self.column] == cat).astype(float)
        return result.drop(columns=[self.column])


class SeasonEncoder(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Sklearn-compatible transformer that adds seasonal features from time_period.

    Extracts the month from ``time_period`` and creates one-hot encoded month
    columns (``month_1`` through ``month_12``).  Optionally maps months to
    seasons using a provided mapping dict.
    """

    def __init__(
        self,
        column: str = "time_period",
        season_mapping: dict[int, str] | None = None,
    ) -> None:
        """Initialize with time column name and optional season mapping."""
        self.column = column
        self.season_mapping = season_mapping

    def fit(self, X: pd.DataFrame, y: object = None) -> SeasonEncoder:
        """Learn season categories (months or custom seasons)."""
        months = pd.to_datetime(X[self.column]).dt.month
        if self.season_mapping is not None:
            seasons = months.map(self.season_mapping)
            self.categories_: list[str] = sorted(seasons.unique().tolist())
            self.prefix_ = "season"
        else:
            self.categories_ = sorted(months.unique().tolist())
            self.prefix_ = "month"
        return self

    def transform(self, X: pd.DataFrame, y: object = None) -> pd.DataFrame:
        """Add one-hot season/month columns."""
        result = X.copy()
        months = pd.to_datetime(result[self.column]).dt.month
        if self.season_mapping is not None:
            values = months.map(self.season_mapping)
        else:
            values = months
        for cat in self.categories_:
            result[f"{self.prefix_}_{cat}"] = (values == cat).astype(float)
        return result


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
