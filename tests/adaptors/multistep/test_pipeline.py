"""Tests for pipeline factory functions."""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import FunctionTransformer  # type: ignore[import-untyped]

from chap_python_sdk.adaptors.multistep.config import MultistepConfig
from chap_python_sdk.adaptors.multistep.pipeline import (
    FeatureLagger,
    InteractionTransformer,
    LocationEncoder,
    SeasonEncoder,
    build_feature_lagger,
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


class TestFeatureLagger:
    """Tests for FeatureLagger."""

    def _make_lagger_df(self) -> pd.DataFrame:
        """Create DataFrame with 5 time steps per location."""
        return pd.DataFrame(
            {
                "time_period": [f"2020-0{i}-01" for i in range(1, 6)] * 2,
                "location": ["A"] * 5 + ["B"] * 5,
                "rainfall": [10.0, 20.0, 30.0, 40.0, 50.0, 15.0, 25.0, 35.0, 45.0, 55.0],
                "temperature": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            }
        )

    def test_transform_adds_lag_columns(self) -> None:
        """Transform adds the expected lag columns."""
        df = self._make_lagger_df()
        lagger = FeatureLagger(n_lags=2, feature_cols=["rainfall", "temperature"])
        result = lagger.fit_transform(df)

        expected_cols = ["rainfall_lag1", "rainfall_lag2", "temperature_lag1", "temperature_lag2"]
        for col in expected_cols:
            assert col in result.columns

    def test_lag_columns_property(self) -> None:
        """lag_columns property returns correct names."""
        lagger = FeatureLagger(n_lags=2, feature_cols=["rainfall", "temperature"])
        assert lagger.lag_columns == [
            "rainfall_lag1",
            "rainfall_lag2",
            "temperature_lag1",
            "temperature_lag2",
        ]

    def test_lag_values_correct(self) -> None:
        """Lag values are correctly shifted per location group."""
        df = self._make_lagger_df()
        lagger = FeatureLagger(n_lags=1, feature_cols=["rainfall"])
        result = lagger.fit_transform(df)

        loc_a = result[result["location"] == "A"]
        assert np.isnan(loc_a["rainfall_lag1"].iloc[0])
        assert loc_a["rainfall_lag1"].iloc[1] == 10.0
        assert loc_a["rainfall_lag1"].iloc[2] == 20.0

        loc_b = result[result["location"] == "B"]
        assert np.isnan(loc_b["rainfall_lag1"].iloc[0])
        assert loc_b["rainfall_lag1"].iloc[1] == 15.0

    def test_nan_rows_preserved(self) -> None:
        """Transform does NOT drop NaN rows — they are kept for caller to handle."""
        df = self._make_lagger_df()
        lagger = FeatureLagger(n_lags=2, feature_cols=["rainfall"])
        result = lagger.fit_transform(df)
        assert len(result) == len(df)

    def test_fit_stores_context(self) -> None:
        """Fit stores the last n_lags rows per location as context_."""
        df = self._make_lagger_df()
        lagger = FeatureLagger(n_lags=2, feature_cols=["rainfall"])
        lagger.fit(df)

        # 2 locations * 2 lags = 4 context rows
        assert len(lagger.context_) == 4
        loc_a_ctx = lagger.context_[lagger.context_["location"] == "A"]
        assert list(loc_a_ctx["rainfall"]) == [40.0, 50.0]

    def test_original_columns_preserved(self) -> None:
        """Original columns are still present after transform."""
        df = self._make_lagger_df()
        lagger = FeatureLagger(n_lags=1, feature_cols=["rainfall"])
        result = lagger.fit_transform(df)
        for col in df.columns:
            assert col in result.columns


class TestBuildFeatureLagger:
    """Tests for build_feature_lagger factory."""

    def test_identity_when_zero_lags(self) -> None:
        """Returns identity FunctionTransformer when n_feature_lags is 0."""
        config = MultistepConfig(n_feature_lags=0)
        result = build_feature_lagger(["rainfall"], config)
        assert isinstance(result, FunctionTransformer)

    def test_identity_when_no_feature_cols(self) -> None:
        """Returns identity FunctionTransformer when feature_cols is empty."""
        config = MultistepConfig(n_feature_lags=2)
        result = build_feature_lagger([], config)
        assert isinstance(result, FunctionTransformer)

    def test_returns_feature_lagger(self) -> None:
        """Returns FeatureLagger when configured with lags and feature cols."""
        config = MultistepConfig(n_feature_lags=3)
        result = build_feature_lagger(["rainfall", "temperature"], config)
        assert isinstance(result, FeatureLagger)
        assert result.n_lags == 3
        assert result.feature_cols == ["rainfall", "temperature"]


class TestLocationEncoder:
    """Tests for LocationEncoder."""

    def test_adds_one_hot_columns(self) -> None:
        """Transform adds one-hot columns for each location."""
        df = _make_df()
        encoder = LocationEncoder()
        result = encoder.fit_transform(df)
        assert "location_A" in result.columns
        assert "location_B" in result.columns
        assert "location" not in result.columns

    def test_values_correct(self) -> None:
        """One-hot values are 1.0 for matching location, 0.0 otherwise."""
        df = _make_df()
        encoder = LocationEncoder()
        result = encoder.fit_transform(df)
        loc_a_rows = df["location"] == "A"
        assert all(result.loc[loc_a_rows, "location_A"] == 1.0)
        assert all(result.loc[loc_a_rows, "location_B"] == 0.0)

    def test_preserves_other_columns(self) -> None:
        """Non-location columns are preserved."""
        df = _make_df()
        encoder = LocationEncoder()
        result = encoder.fit_transform(df)
        assert "rainfall" in result.columns
        assert "temperature" in result.columns
        assert "time_period" in result.columns

    def test_row_count_preserved(self) -> None:
        """Number of rows is unchanged."""
        df = _make_df()
        encoder = LocationEncoder()
        result = encoder.fit_transform(df)
        assert len(result) == len(df)


class TestSeasonEncoder:
    """Tests for SeasonEncoder."""

    def test_adds_month_columns(self) -> None:
        """Default mode adds month columns."""
        df = _make_df()
        encoder = SeasonEncoder()
        result = encoder.fit_transform(df)
        assert "month_1" in result.columns
        assert "month_2" in result.columns
        assert "month_3" in result.columns

    def test_month_values_correct(self) -> None:
        """Month one-hot values are correct."""
        df = _make_df()
        encoder = SeasonEncoder()
        result = encoder.fit_transform(df)
        # First row is 2020-01-01, should be month 1
        assert result["month_1"].iloc[0] == 1.0
        assert result["month_2"].iloc[0] == 0.0

    def test_custom_season_mapping(self) -> None:
        """Custom season mapping creates season columns."""
        df = _make_df()
        mapping = {
            1: "dry",
            2: "dry",
            3: "wet",
            4: "wet",
            5: "wet",
            6: "wet",
            7: "wet",
            8: "wet",
            9: "wet",
            10: "dry",
            11: "dry",
            12: "dry",
        }
        encoder = SeasonEncoder(season_mapping=mapping)
        result = encoder.fit_transform(df)
        assert "season_dry" in result.columns
        assert "season_wet" in result.columns

    def test_preserves_time_period(self) -> None:
        """time_period column is preserved (not dropped)."""
        df = _make_df()
        encoder = SeasonEncoder()
        result = encoder.fit_transform(df)
        assert "time_period" in result.columns

    def test_row_count_preserved(self) -> None:
        """Number of rows is unchanged."""
        df = _make_df()
        encoder = SeasonEncoder()
        result = encoder.fit_transform(df)
        assert len(result) == len(df)


class TestInteractionTransformer:
    """Tests for InteractionTransformer."""

    def _make_encoded_df(self) -> pd.DataFrame:
        """Create DataFrame with pre-encoded location and season columns."""
        return pd.DataFrame(
            {
                "location_A": [1.0, 1.0, 0.0, 0.0],
                "location_B": [0.0, 0.0, 1.0, 1.0],
                "season_dry": [1.0, 0.0, 1.0, 0.0],
                "season_wet": [0.0, 1.0, 0.0, 1.0],
                "rainfall": [100.0, 200.0, 150.0, 250.0],
            }
        )

    def test_creates_interaction_columns(self) -> None:
        """Transform creates pairwise interaction columns."""
        df = self._make_encoded_df()
        transformer = InteractionTransformer()
        result = transformer.fit_transform(df)
        assert "location_A_x_season_dry" in result.columns
        assert "location_A_x_season_wet" in result.columns
        assert "location_B_x_season_dry" in result.columns
        assert "location_B_x_season_wet" in result.columns

    def test_interaction_values_correct(self) -> None:
        """Interaction values are products of the component columns."""
        df = self._make_encoded_df()
        transformer = InteractionTransformer()
        result = transformer.fit_transform(df)
        # Row 0: location_A=1, season_dry=1 -> interaction=1
        assert result["location_A_x_season_dry"].iloc[0] == 1.0
        # Row 0: location_A=1, season_wet=0 -> interaction=0
        assert result["location_A_x_season_wet"].iloc[0] == 0.0
        # Row 2: location_B=1, season_dry=1 -> interaction=1
        assert result["location_B_x_season_dry"].iloc[2] == 1.0

    def test_preserves_original_columns(self) -> None:
        """Original columns are preserved."""
        df = self._make_encoded_df()
        transformer = InteractionTransformer()
        result = transformer.fit_transform(df)
        for col in df.columns:
            assert col in result.columns

    def test_row_count_preserved(self) -> None:
        """Number of rows is unchanged."""
        df = self._make_encoded_df()
        transformer = InteractionTransformer()
        result = transformer.fit_transform(df)
        assert len(result) == len(df)

    def test_custom_prefixes(self) -> None:
        """Works with custom prefixes."""
        df = pd.DataFrame(
            {
                "cat_a": [1.0, 0.0],
                "cat_b": [0.0, 1.0],
                "feat_x": [1.0, 0.0],
                "feat_y": [0.0, 1.0],
            }
        )
        transformer = InteractionTransformer(left_prefix="cat_", right_prefix="feat_")
        result = transformer.fit_transform(df)
        assert "cat_a_x_feat_x" in result.columns
        assert "cat_b_x_feat_y" in result.columns
