"""Tests for pipeline factory functions."""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import FunctionTransformer  # type: ignore[import-untyped]

from chap_python_sdk.adaptors.multistep.config import MultistepConfig
from chap_python_sdk.adaptors.multistep.pipeline import (
    build_feature_transformer,
    build_target_pipeline,
)


def _make_df() -> pd.DataFrame:
    """Create a simple test DataFrame."""
    return pd.DataFrame(
        {
            "time_period": ["2020-01-01", "2020-02-01", "2020-03-01"] * 2,
            "location": ["A"] * 3 + ["B"] * 3,
            "disease_cases": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "rainfall": [100.0, 200.0, 300.0, 150.0, 250.0, 350.0],
            "temperature": [20.0, 25.0, 30.0, 22.0, 27.0, 32.0],
        }
    )


class TestBuildTargetPipeline:
    """Tests for build_target_pipeline."""

    def test_identity_when_no_transforms(self) -> None:
        """Default config produces identity pipeline."""
        config = MultistepConfig()
        pipeline = build_target_pipeline(config)
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0][0] == "identity"

    def test_identity_roundtrip(self) -> None:
        """Identity pipeline preserves values on fit_transform/inverse_transform."""
        config = MultistepConfig()
        pipeline = build_target_pipeline(config)
        values = np.array([10.0, 20.0, 30.0]).reshape(-1, 1)
        transformed = pipeline.fit_transform(values)
        restored = pipeline.inverse_transform(transformed)
        np.testing.assert_allclose(restored, values)

    def test_log_only(self) -> None:
        """Config with only log transform returns single-step pipeline."""
        config = MultistepConfig(log_transform_target=True)
        pipeline = build_target_pipeline(config)
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0][0] == "log"

    def test_standardize_only(self) -> None:
        """Config with only standardize returns single-step pipeline."""
        config = MultistepConfig(standardize_target=True)
        pipeline = build_target_pipeline(config)
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0][0] == "scaler"

    def test_log_and_standardize(self) -> None:
        """Config with both returns two-step pipeline (log first)."""
        config = MultistepConfig(log_transform_target=True, standardize_target=True)
        pipeline = build_target_pipeline(config)
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0][0] == "log"
        assert pipeline.steps[1][0] == "scaler"

    def test_fit_transform_inverse_roundtrip(self) -> None:
        """fit_transform then inverse_transform roundtrip preserves values."""
        config = MultistepConfig(log_transform_target=True, standardize_target=True)
        pipeline = build_target_pipeline(config)

        values = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]).reshape(-1, 1)
        transformed = pipeline.fit_transform(values)
        restored = pipeline.inverse_transform(transformed)
        np.testing.assert_allclose(restored, values, rtol=1e-10)

    def test_log_transform_values(self) -> None:
        """Log transform applies log1p."""
        config = MultistepConfig(log_transform_target=True)
        pipeline = build_target_pipeline(config)

        values = np.array([10.0, 20.0, 30.0]).reshape(-1, 1)
        transformed = pipeline.transform(values)
        np.testing.assert_allclose(transformed, np.log1p(values))

    def test_standardize_fits_on_log_transformed(self) -> None:
        """When both enabled, standardizer fits on log-transformed values."""
        config = MultistepConfig(log_transform_target=True, standardize_target=True)
        pipeline = build_target_pipeline(config)

        values = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]).reshape(-1, 1)
        pipeline.fit(values)

        scaler = pipeline.named_steps["scaler"]
        log_values = np.log1p(values)
        np.testing.assert_allclose(float(scaler.mean_[0]), float(np.mean(log_values)))
        np.testing.assert_allclose(float(scaler.scale_[0]), float(np.std(log_values)))


class TestBuildFeatureTransformer:
    """Tests for build_feature_transformer."""

    def test_identity_when_not_configured(self) -> None:
        """Returns identity FunctionTransformer when standardize_covariates is False."""
        config = MultistepConfig()
        result = build_feature_transformer(["rainfall"], config)
        assert isinstance(result, FunctionTransformer)

    def test_identity_when_no_feature_cols(self) -> None:
        """Returns identity FunctionTransformer when feature_cols is empty."""
        config = MultistepConfig(standardize_covariates=True)
        result = build_feature_transformer([], config)
        assert isinstance(result, FunctionTransformer)

    def test_identity_roundtrip(self) -> None:
        """Identity transformer preserves values."""
        config = MultistepConfig()
        transformer = build_feature_transformer(["rainfall"], config)
        df = _make_df()
        features = df[["rainfall"]]
        result = transformer.fit_transform(features)
        np.testing.assert_allclose(np.asarray(result), features.to_numpy())

    def test_returns_column_transformer(self) -> None:
        """Returns ColumnTransformer when configured with feature cols."""
        config = MultistepConfig(standardize_covariates=True)
        ct = build_feature_transformer(["rainfall", "temperature"], config)
        assert isinstance(ct, ColumnTransformer)

    def test_fit_transform_output_shape(self) -> None:
        """Transformer output has correct shape (feature columns only)."""
        config = MultistepConfig(standardize_covariates=True)
        ct = build_feature_transformer(["rainfall", "temperature"], config)

        df = _make_df()
        result = ct.fit_transform(df[["rainfall", "temperature"]])
        assert result.shape[0] == len(df)  # pyright: ignore[reportOptionalSubscript]
        assert result.shape[1] == 2  # pyright: ignore[reportOptionalSubscript]

    def test_fit_transform_scales_features(self) -> None:
        """Feature columns are standardized (approx zero mean)."""
        config = MultistepConfig(standardize_covariates=True)
        ct = build_feature_transformer(["rainfall", "temperature"], config)

        df = _make_df()
        result = ct.fit_transform(df[["rainfall", "temperature"]])

        scaled_vals = np.asarray(result)
        np.testing.assert_allclose(scaled_vals.mean(axis=0), 0.0, atol=1e-10)

    def test_fit_transform_roundtrip(self) -> None:
        """Underlying scaler fit_transform then inverse_transform roundtrip preserves values."""
        config = MultistepConfig(standardize_covariates=True)
        ct = build_feature_transformer(["rainfall"], config)
        assert isinstance(ct, ColumnTransformer)

        df = _make_df()
        features = df[["rainfall"]]
        ct.fit(features)
        scaler = ct.named_transformers_["scaler"]
        transformed = scaler.transform(features.to_numpy())
        restored = scaler.inverse_transform(transformed)
        np.testing.assert_allclose(restored, features.to_numpy(), rtol=1e-10)
