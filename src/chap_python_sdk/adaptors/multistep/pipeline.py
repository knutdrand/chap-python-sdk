"""Factory functions for sklearn transform pipelines."""

from __future__ import annotations

import numpy as np
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
