"""Tests for multistep data transformer."""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import xarray as xr
from chapkit.data import DataFrame

from chap_python_sdk.adaptors.multistep.data_transformer import (
    xarray_predictions_to_chapkit,
)


def _make_future_df(
    n_locations: int = 2,
    n_steps: int = 3,
) -> DataFrame:
    """Create a synthetic future chapkit DataFrame."""
    locations = [f"loc_{i}" for i in range(n_locations)]
    times = pd.date_range("2020-11-01", periods=n_steps, freq="MS")

    rows: dict[str, list[object]] = {"time_period": [], "location": []}
    for loc in locations:
        for t in times:
            rows["time_period"].append(str(t.isoformat()))
            rows["location"].append(loc)

    return DataFrame.from_dict(rows)


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

    def test_preserves_row_count(self) -> None:
        """Output has locations * steps rows."""
        future = _make_future_df(n_locations=2, n_steps=3)

        predictions = xr.DataArray(
            np.random.default_rng(42).normal(100, 10, size=(2, 20, 3)),
            dims=["location", "trajectory", "step"],
            coords={"location": ["loc_0", "loc_1"]},
        )

        result = xarray_predictions_to_chapkit(predictions, future)
        result_pd = result.to_pandas()

        assert len(result_pd) == 6  # 2 locs * 3 steps
        assert all(len(s) == 20 for s in result_pd["samples"])
