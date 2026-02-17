"""Tests for multistep data transformer."""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import xarray as xr
from chapkit.data import DataFrame

from chap_python_sdk.adaptors.multistep.data_transformer import (
    chapkit_future_to_xarray,
    chapkit_to_xarray,
    xarray_predictions_to_chapkit,
)


def _make_chapkit_df(
    n_locations: int = 2,
    n_times: int = 10,
    include_exog: bool = False,
) -> DataFrame:
    """Create a synthetic chapkit DataFrame for testing."""
    locations = [f"loc_{i}" for i in range(n_locations)]
    times = pd.date_range("2020-01-01", periods=n_times, freq="MS")
    rng = np.random.default_rng(42)

    rows: dict[str, list[object]] = {"time_period": [], "location": [], "disease_cases": []}
    if include_exog:
        rows["rainfall"] = []
        rows["mean_temperature"] = []

    for loc in locations:
        for t in times:
            rows["time_period"].append(str(t.isoformat()))
            rows["location"].append(loc)
            rows["disease_cases"].append(float(rng.poisson(100)))
            if include_exog:
                rows["rainfall"].append(float(rng.normal(50, 10)))
                rows["mean_temperature"].append(float(rng.normal(25, 5)))

    return DataFrame.from_dict(rows)


def _make_future_df(
    n_locations: int = 2,
    n_steps: int = 3,
    include_exog: bool = False,
) -> DataFrame:
    """Create a synthetic future chapkit DataFrame."""
    locations = [f"loc_{i}" for i in range(n_locations)]
    times = pd.date_range("2020-11-01", periods=n_steps, freq="MS")
    rng = np.random.default_rng(42)

    rows: dict[str, list[object]] = {"time_period": [], "location": []}
    if include_exog:
        rows["rainfall"] = []
        rows["mean_temperature"] = []

    for loc in locations:
        for t in times:
            rows["time_period"].append(str(t.isoformat()))
            rows["location"].append(loc)
            if include_exog:
                rows["rainfall"].append(float(rng.normal(50, 10)))
                rows["mean_temperature"].append(float(rng.normal(25, 5)))

    return DataFrame.from_dict(rows)


class TestChapkitToXarray:
    """Tests for chapkit_to_xarray."""

    def test_output_shape(self) -> None:
        """Output y has correct dims and shape."""
        data = _make_chapkit_df(n_locations=3, n_times=12)
        y, X = chapkit_to_xarray(data)

        assert y.dims == ("location", "time")
        assert y.sizes["location"] == 3
        assert y.sizes["time"] == 12
        assert X is None

    def test_with_exogenous(self) -> None:
        """X is returned when exogenous variables specified."""
        data = _make_chapkit_df(n_locations=2, n_times=10, include_exog=True)
        y, X = chapkit_to_xarray(data, exogenous_variables=["rainfall", "mean_temperature"])

        assert y.dims == ("location", "time")
        assert X is not None
        assert X.dims == ("location", "time", "feature")
        assert X.sizes["feature"] == 2
        assert X.sizes["location"] == 2
        assert X.sizes["time"] == 10

    def test_location_coords(self) -> None:
        """Location coordinates are preserved."""
        data = _make_chapkit_df(n_locations=2)
        y, _ = chapkit_to_xarray(data)
        locations = y.coords["location"].values.tolist()
        assert "loc_0" in locations
        assert "loc_1" in locations

    def test_without_exog_returns_none(self) -> None:
        """X is None when no exogenous variables."""
        data = _make_chapkit_df()
        _, X = chapkit_to_xarray(data)
        assert X is None


class TestChapkitFutureToXarray:
    """Tests for chapkit_future_to_xarray."""

    def test_basic(self) -> None:
        """Returns correct locations and time periods."""
        future = _make_future_df(n_locations=2, n_steps=3)
        locations, time_periods, X_future = chapkit_future_to_xarray(future)

        assert len(locations) == 2
        assert len(time_periods) == 3
        assert X_future is None

    def test_with_exogenous(self) -> None:
        """X_future has correct shape with exogenous variables."""
        future = _make_future_df(n_locations=2, n_steps=3, include_exog=True)
        _locations, _time_periods, X_future = chapkit_future_to_xarray(
            future, exogenous_variables=["rainfall", "mean_temperature"]
        )

        assert X_future is not None
        assert X_future.dims == ("location", "step", "feature")
        assert X_future.sizes["location"] == 2
        assert X_future.sizes["step"] == 3
        assert X_future.sizes["feature"] == 2


class TestXarrayPredictionsToChapkit:
    """Tests for xarray_predictions_to_chapkit."""

    def test_output_format(self) -> None:
        """Output has correct columns and row count."""
        future = _make_future_df(n_locations=2, n_steps=3)

        predictions = xr.DataArray(
            np.random.default_rng(42).normal(100, 10, size=(2, 50, 3)),
            dims=["location", "trajectory", "step"],
            coords={"location": ["loc_0", "loc_1"]},
        )

        result = xarray_predictions_to_chapkit(predictions, future)
        result_pd = result.to_pandas()

        assert "time_period" in result_pd.columns
        assert "location" in result_pd.columns
        assert "samples" in result_pd.columns
        # 2 locations * 3 steps = 6 rows
        assert len(result_pd) == 6

    def test_samples_are_lists(self) -> None:
        """Each samples entry is a list of floats."""
        future = _make_future_df(n_locations=1, n_steps=2)

        predictions = xr.DataArray(
            np.ones((1, 10, 2)),
            dims=["location", "trajectory", "step"],
            coords={"location": ["loc_0"]},
        )

        result = xarray_predictions_to_chapkit(predictions, future)
        result_pd = result.to_pandas()

        for samples in result_pd["samples"]:
            assert isinstance(samples, list)
            assert len(samples) == 10

    def test_roundtrip_shapes(self) -> None:
        """Roundtrip: chapkit → xarray → predictions → chapkit preserves structure."""
        data = _make_chapkit_df(n_locations=2, n_times=10)
        future = _make_future_df(n_locations=2, n_steps=3)

        y, _X = chapkit_to_xarray(data)

        predictions = xr.DataArray(
            np.random.default_rng(42).normal(100, 10, size=(2, 20, 3)),
            dims=["location", "trajectory", "step"],
            coords={"location": y.coords["location"].values.tolist()},
        )

        result = xarray_predictions_to_chapkit(predictions, future)
        result_pd = result.to_pandas()

        assert len(result_pd) == 6  # 2 locs * 3 steps
        assert all(len(s) == 20 for s in result_pd["samples"])
