"""Tests for data transformation between chapkit and skforecast formats."""

import pandas as pd  # type: ignore[import-untyped]
import pytest

pl = pytest.importorskip("polars", reason="polars not installed")
pytest.importorskip("skforecast", reason="skforecast not installed")

from chapkit.data import DataFrame as ChapkitDataFrame  # noqa: E402

from chap_python_sdk.adaptors.skforecast.data_transformer import chapkit_to_wide, wide_to_chapkit  # noqa: E402


class TestChapkitToWide:
    """Tests for chapkit_to_wide transformation."""

    def test_basic_transformation(self) -> None:
        """Test basic long to wide transformation."""
        data = pl.DataFrame(
            {
                "time_period": ["2023-01", "2023-02", "2023-03", "2023-01", "2023-02", "2023-03"],
                "location": ["A", "A", "A", "B", "B", "B"],
                "disease_cases": [10, 20, 30, 15, 25, 35],
            }
        )

        target_wide, exog_wide = chapkit_to_wide(data, target_variable="disease_cases")

        assert isinstance(target_wide, pd.DataFrame)
        assert target_wide.shape == (3, 2)
        assert list(target_wide.columns) == ["A", "B"]
        assert target_wide.loc["2023-01-01", "A"] == 10
        assert target_wide.loc["2023-03-01", "B"] == 35
        assert exog_wide is None

    def test_with_exogenous_variables(self) -> None:
        """Test transformation with exogenous variables."""
        data = pl.DataFrame(
            {
                "time_period": ["2023-01", "2023-02", "2023-03", "2023-01", "2023-02", "2023-03"],
                "location": ["A", "A", "A", "B", "B", "B"],
                "disease_cases": [10, 20, 30, 15, 25, 35],
                "rainfall": [100.0, 150.0, 120.0, 110.0, 160.0, 130.0],
                "temperature": [25.0, 27.0, 26.0, 26.0, 28.0, 27.0],
            }
        )

        target_wide, exog_wide = chapkit_to_wide(
            data,
            target_variable="disease_cases",
            exogenous_variables=["rainfall", "temperature"],
        )

        assert target_wide.shape == (3, 2)
        assert exog_wide is not None
        assert exog_wide.shape == (3, 4)
        assert "rainfall_A" in exog_wide.columns
        assert "rainfall_B" in exog_wide.columns
        assert "temperature_A" in exog_wide.columns
        assert "temperature_B" in exog_wide.columns

    def test_datetime_index(self) -> None:
        """Test that time_period is correctly converted to DatetimeIndex."""
        data = pl.DataFrame(
            {
                "time_period": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-01-01", "2023-02-01", "2023-03-01"],
                "location": ["A", "A", "A", "B", "B", "B"],
                "disease_cases": [10, 20, 30, 15, 25, 35],
            }
        )

        target_wide, _ = chapkit_to_wide(data, target_variable="disease_cases")

        assert isinstance(target_wide.index, pd.DatetimeIndex)
        assert target_wide.index[0] == pd.Timestamp("2023-01-01")

    def test_sorted_index(self) -> None:
        """Test that output is sorted by time."""
        data = pl.DataFrame(
            {
                "time_period": ["2023-02", "2023-01", "2023-03", "2023-02", "2023-01", "2023-03"],
                "location": ["A", "A", "A", "B", "B", "B"],
                "disease_cases": [20, 10, 30, 25, 15, 35],
            }
        )

        target_wide, _ = chapkit_to_wide(data, target_variable="disease_cases")

        assert list(target_wide.index) == [pd.Timestamp("2023-01"), pd.Timestamp("2023-02"), pd.Timestamp("2023-03")]

    def test_two_dates_no_freq_error(self) -> None:
        """Test that < 3 dates doesn't crash on infer_freq."""
        data = pl.DataFrame(
            {
                "time_period": ["2023-01", "2023-02"],
                "location": ["A", "A"],
                "disease_cases": [10, 20],
            }
        )

        target_wide, _ = chapkit_to_wide(data, target_variable="disease_cases")

        assert target_wide.shape == (2, 1)


class TestWideToChapkit:
    """Tests for wide_to_chapkit transformation."""

    def test_basic_transformation(self) -> None:
        """Test basic wide to long transformation."""
        predictions_wide = {
            "A": pd.DataFrame([[10.0, 12.0, 11.0], [15.0, 17.0, 16.0]]),
            "B": pd.DataFrame([[20.0, 22.0, 21.0], [25.0, 27.0, 26.0]]),
        }

        future = pl.DataFrame(
            {
                "time_period": ["2023-04", "2023-05", "2023-04", "2023-05"],
                "location": ["A", "A", "B", "B"],
            }
        )

        result = wide_to_chapkit(predictions_wide, future)

        assert isinstance(result, ChapkitDataFrame)
        assert len(result) == 4
        assert "time_period" in result.columns
        assert "location" in result.columns
        assert "samples" in result.columns

    def test_samples_column_format(self) -> None:
        """Test that samples column contains lists of floats."""
        predictions_wide = {
            "A": pd.DataFrame([[10.0, 12.0, 11.0]]),
        }

        future = pl.DataFrame(
            {
                "time_period": ["2023-04"],
                "location": ["A"],
            }
        )

        result = wide_to_chapkit(predictions_wide, future)

        samples = result["samples"][0]  # type: ignore[index]
        assert isinstance(samples, list)
        assert len(samples) == 3
        assert all(isinstance(s, float) for s in samples)

    def test_location_alignment(self) -> None:
        """Test that locations are correctly aligned with future DataFrame."""
        predictions_wide = {
            "A": pd.DataFrame([[10.0, 12.0]]),
            "B": pd.DataFrame([[20.0, 22.0]]),
        }

        future = pl.DataFrame(
            {
                "time_period": ["2023-04", "2023-04"],
                "location": ["A", "B"],
            }
        )

        result = wide_to_chapkit(predictions_wide, future)

        # Check samples by location using chapkit DataFrame indexing
        locations = result["location"]
        samples = result["samples"]
        for i, loc in enumerate(locations):
            if loc == "A":
                assert samples[i] == [10.0, 12.0]  # type: ignore[index]
            elif loc == "B":
                assert samples[i] == [20.0, 22.0]  # type: ignore[index]

    def test_time_period_alignment(self) -> None:
        """Test that time periods are correctly aligned."""
        predictions_wide = {
            "A": pd.DataFrame([[10.0, 12.0], [15.0, 17.0]]),
        }

        future = pl.DataFrame(
            {
                "time_period": ["2023-04", "2023-05"],
                "location": ["A", "A"],
            }
        )

        result = wide_to_chapkit(predictions_wide, future)

        result_sorted = result.sort("time_period")
        assert result_sorted["time_period"][0] == "2023-04-01T00:00:00"  # type: ignore[index]
        assert result_sorted["time_period"][1] == "2023-05-01T00:00:00"  # type: ignore[index]
        assert result_sorted["samples"][0] == [10.0, 12.0]  # type: ignore[index]
        assert result_sorted["samples"][1] == [15.0, 17.0]  # type: ignore[index]

    def test_mismatch_length_raises_error(self) -> None:
        """Test that mismatched prediction length raises an error."""
        predictions_wide = {
            "A": pd.DataFrame([[10.0, 12.0], [15.0, 17.0]]),
        }

        future = pl.DataFrame(
            {
                "time_period": ["2023-04"],
                "location": ["A"],
            }
        )

        with pytest.raises(ValueError, match="Mismatch in prediction length"):
            wide_to_chapkit(predictions_wide, future)
