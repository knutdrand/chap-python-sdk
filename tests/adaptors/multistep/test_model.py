"""Tests for DataFrameMultistepModel."""

import pickle

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import FunctionTransformer, StandardScaler  # type: ignore[import-untyped]

from chap_python_sdk.adaptors.multistep.model import DataFrameMultistepModel
from chap_python_sdk.adaptors.multistep.one_step_model import ResidualBootstrapModel


def _make_one_step() -> ResidualBootstrapModel:
    return ResidualBootstrapModel(
        "sklearn.ensemble.GradientBoostingRegressor",
        {"n_estimators": 10, "max_depth": 2, "random_state": 42},
    )


def _make_data(
    n_times: int = 24,
    locations: list[str] | None = None,
    include_exog: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (X, y) DataFrames."""
    if locations is None:
        locations = ["loc_A", "loc_B"]
    rng = np.random.default_rng(42)
    rows = []
    for loc in locations:
        for t in range(n_times):
            year = 2020 + t // 12
            month = (t % 12) + 1
            row: dict[str, object] = {
                "time_period": f"{year}-{month:02d}-01",
                "location": loc,
                "disease_cases": float(rng.poisson(50)),
            }
            if include_exog:
                row["rainfall"] = float(rng.uniform(0, 100))
                row["temperature"] = float(rng.uniform(15, 35))
            rows.append(row)
    df = pd.DataFrame(rows)
    index_cols = ["time_period", "location"]
    target = "disease_cases"
    feature_cols = [c for c in df.columns if c not in [*index_cols, target]]
    X: pd.DataFrame = df[index_cols + feature_cols]  # pyright: ignore[reportAssignmentType]
    y: pd.DataFrame = df[index_cols + [target]]  # pyright: ignore[reportAssignmentType]
    return X, y


def _make_future(
    n_steps: int = 3,
    locations: list[str] | None = None,
    include_exog: bool = False,
) -> pd.DataFrame:
    if locations is None:
        locations = ["loc_A", "loc_B"]
    rng = np.random.default_rng(456)
    rows = []
    for loc in locations:
        for t in range(n_steps):
            row: dict[str, object] = {
                "time_period": f"2022-{t + 1:02d}-01",
                "location": loc,
            }
            if include_exog:
                row["rainfall"] = float(rng.uniform(0, 100))
                row["temperature"] = float(rng.uniform(15, 35))
            rows.append(row)
    return pd.DataFrame(rows)


class TestDataFrameMultistepModel:
    """Tests for DataFrameMultistepModel."""

    def test_fit_predict_no_exog(self) -> None:
        """Fit and predict with no exogenous features."""
        X, y = _make_data()
        model = DataFrameMultistepModel(_make_one_step(), n_target_lags=4)
        model.fit(X, y)

        y_historic = y.copy()
        X_future = _make_future()
        preds = model.predict_xarray(y_historic, X_future, n_steps=3, n_samples=5)

        assert preds.dims == ("location", "trajectory", "step")
        assert preds.sizes["location"] == 2
        assert preds.sizes["trajectory"] == 5
        assert preds.sizes["step"] == 3

    def test_fit_predict_with_exog(self) -> None:
        """Fit and predict with exogenous features."""
        X, y = _make_data(include_exog=True)
        model = DataFrameMultistepModel(_make_one_step(), n_target_lags=4)
        model.fit(X, y)

        y_historic = y.copy()
        X_future = _make_future(include_exog=True)
        preds = model.predict_xarray(y_historic, X_future, n_steps=3, n_samples=5)

        assert preds.dims == ("location", "trajectory", "step")
        assert preds.sizes["location"] == 2
        assert preds.sizes["trajectory"] == 5
        assert preds.sizes["step"] == 3

    def test_fit_predict_no_exog_none_X(self) -> None:
        """Predict with X_future=None works."""
        X, y = _make_data()
        model = DataFrameMultistepModel(_make_one_step(), n_target_lags=4)
        model.fit(X, y)

        y_historic = y.copy()
        preds = model.predict_xarray(y_historic, None, n_steps=3, n_samples=5)

        assert preds.sizes["step"] == 3
        assert preds.sizes["trajectory"] == 5

    def test_target_pipeline_applied(self) -> None:
        """Target pipeline transforms during fit and inverse-transforms during predict."""
        X, y = _make_data()
        target_pipeline = Pipeline(
            [
                ("log", FunctionTransformer(func=np.log1p, inverse_func=np.expm1)),
                ("scaler", StandardScaler()),
            ]
        )
        model = DataFrameMultistepModel(_make_one_step(), n_target_lags=4, target_pipeline=target_pipeline)
        model.fit(X, y)

        y_historic = y.copy()
        preds = model.predict_xarray(y_historic, None, n_steps=3, n_samples=5)

        # Predictions should be in original scale (positive values for count data)
        assert preds.sizes["step"] == 3
        # Values should be reasonable (not in standardized scale)
        assert float(preds.mean()) > 1.0

    def test_no_target_pipeline_identity(self) -> None:
        """With target_pipeline=None, predictions are in raw model scale."""
        X, y = _make_data()
        model = DataFrameMultistepModel(_make_one_step(), n_target_lags=4, target_pipeline=None)
        model.fit(X, y)

        preds = model.predict_xarray(y.copy(), None, n_steps=3, n_samples=5)
        assert preds.sizes["step"] == 3

    def test_pickle_roundtrip(self) -> None:
        """Model survives pickle roundtrip."""
        X, y = _make_data()
        target_pipeline = Pipeline(
            [
                ("log", FunctionTransformer(func=np.log1p, inverse_func=np.expm1)),
            ]
        )
        model = DataFrameMultistepModel(_make_one_step(), n_target_lags=4, target_pipeline=target_pipeline)
        model.fit(X, y)

        serialized = pickle.dumps(model)
        restored: DataFrameMultistepModel = pickle.loads(serialized)  # noqa: S301

        preds_orig = model.predict_xarray(y.copy(), None, n_steps=3, n_samples=5)
        preds_restored = restored.predict_xarray(y.copy(), None, n_steps=3, n_samples=5)

        # Same shape (values differ due to stochastic sampling)
        assert preds_orig.shape == preds_restored.shape

    def test_single_location(self) -> None:
        """Works with a single location."""
        X, y = _make_data(locations=["only_loc"])
        model = DataFrameMultistepModel(_make_one_step(), n_target_lags=4)
        model.fit(X, y)

        preds = model.predict_xarray(y.copy(), None, n_steps=3, n_samples=5)
        assert preds.sizes["location"] == 1

    def test_predict(self) -> None:
        """Predict returns a wide-format DataFrame with one column per sample."""
        n_locations = 2
        n_steps = 3
        n_samples = 5
        X, y = _make_data(locations=["loc_A", "loc_B"])
        model = DataFrameMultistepModel(_make_one_step(), n_target_lags=4)
        model.fit(X, y)

        df = model.predict(y.copy(), _make_future(), n_steps=n_steps, n_samples=n_samples)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == n_locations * n_steps
        assert "location" in df.columns
        assert "time_step" in df.columns
        sample_cols = [c for c in df.columns if c.startswith("sample_")]
        assert len(sample_cols) == n_samples
