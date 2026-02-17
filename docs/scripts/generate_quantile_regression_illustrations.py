"""Generate illustrations for quantile regression documentation using Altair."""

import numpy as np
import pandas as pd
import altair as alt
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "images"
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_ols_vs_quantile_comparison() -> None:
    """Generate comparison of OLS vs quantile regression on heteroscedastic data."""
    n = 200
    X = np.random.uniform(0, 10, n)
    # Heteroscedastic: variance increases with X
    y = 2 + 3 * X + np.random.normal(0, 0.3 + 0.5 * X, n)

    # OLS fit
    ols_slope = np.cov(X, y)[0, 1] / np.var(X)
    ols_intercept = np.mean(y) - ols_slope * np.mean(X)
    y_ols = ols_intercept + ols_slope * X

    # Sort for quantile regression lines
    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]

    # Approximate quantile lines (for visualization)
    # Use rolling quantiles
    window = 40
    quantiles = [0.1, 0.5, 0.9]
    y_quantiles = {tau: np.zeros(n) for tau in quantiles}

    for i in range(n):
        start = max(0, i - window // 2)
        end = min(n, i + window // 2)
        y_window = y_sorted[start:end]

        for tau in quantiles:
            y_quantiles[tau][i] = np.percentile(y_window, tau * 100)

    # Create DataFrame
    df_points = pd.DataFrame({"X": X, "Y": y})
    df_ols = pd.DataFrame({"X": X_sorted, "Y": y_ols[sort_idx], "Type": "OLS (Mean)"})

    df_quantiles = []
    for tau in quantiles:
        df_quantiles.append(
            pd.DataFrame(
                {"X": X_sorted, "Y": y_quantiles[tau], "Type": f"τ={tau}"}
            )
        )

    df_lines = pd.concat([df_ols] + df_quantiles, ignore_index=True)

    # Create chart
    points = (
        alt.Chart(df_points)
        .mark_circle(size=30, opacity=0.4, color="gray")
        .encode(x=alt.X("X:Q", title="X"), y=alt.Y("Y:Q", title="Y"))
    )

    lines = (
        alt.Chart(df_lines)
        .mark_line(strokeWidth=2.5)
        .encode(
            x="X:Q",
            y="Y:Q",
            color=alt.Color(
                "Type:N",
                scale=alt.Scale(
                    domain=["OLS (Mean)", "τ=0.1", "τ=0.5", "τ=0.9"],
                    range=["red", "steelblue", "orange", "purple"],
                ),
                legend=alt.Legend(title="Regression Type"),
            ),
            strokeDash=alt.condition(
                alt.datum.Type == "OLS (Mean)", alt.value([5, 5]), alt.value([0])
            ),
        )
    )

    chart = (points + lines).properties(
        width=600,
        height=400,
        title="OLS vs Quantile Regression: Heteroscedastic Data",
    )

    chart.save(str(OUTPUT_DIR / "ols_vs_quantile_comparison.png"))
    print("Generated: ols_vs_quantile_comparison.png")


def generate_check_loss_function() -> None:
    """Generate visualization of check loss function for different quantiles."""
    u = np.linspace(-3, 3, 300)
    quantiles = [0.1, 0.5, 0.9]

    data_list = []
    for tau in quantiles:
        loss = np.where(u >= 0, tau * u, (tau - 1) * u)
        for i in range(len(u)):
            data_list.append({"u": u[i], "Loss": loss[i], "τ": f"τ={tau}"})

    df = pd.DataFrame(data_list)

    chart = (
        alt.Chart(df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("u:Q", title="Residual (y - ŷ)"),
            y=alt.Y("Loss:Q", title="Loss ρτ(u)"),
            color=alt.Color(
                "τ:N",
                scale=alt.Scale(
                    domain=["τ=0.1", "τ=0.5", "τ=0.9"],
                    range=["steelblue", "orange", "purple"],
                ),
            ),
        )
        .properties(
            width=600, height=400, title="Check Loss Function (Pinball Loss)"
        )
    )

    chart.save(str(OUTPUT_DIR / "check_loss_function.png"))
    print("Generated: check_loss_function.png")


def generate_heteroscedastic_comparison() -> None:
    """Generate detailed comparison showing how quantiles capture heteroscedasticity."""
    n = 200
    X = np.random.uniform(0, 10, n)
    y = 2 + 3 * X + np.random.normal(0, 0.5 + 0.4 * X, n)

    # Sort for smooth lines
    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]

    # Rolling quantiles
    window = 50
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    y_quantiles = {tau: np.zeros(n) for tau in quantiles}

    for i in range(n):
        start = max(0, i - window // 2)
        end = min(n, i + window // 2)
        y_window = y_sorted[start:end]

        for tau in quantiles:
            y_quantiles[tau][i] = np.percentile(y_window, tau * 100)

    # Create DataFrame
    df_points = pd.DataFrame({"X": X, "Y": y})

    df_quantiles = []
    for tau in quantiles:
        df_quantiles.append(
            pd.DataFrame(
                {"X": X_sorted, "Y": y_quantiles[tau], "Quantile": f"τ={tau}"}
            )
        )

    df_lines = pd.concat(df_quantiles, ignore_index=True)

    # Create chart
    points = (
        alt.Chart(df_points)
        .mark_circle(size=25, opacity=0.3, color="lightgray")
        .encode(x=alt.X("X:Q", title="X"), y=alt.Y("Y:Q", title="Y"))
    )

    lines = (
        alt.Chart(df_lines)
        .mark_line(strokeWidth=2)
        .encode(
            x="X:Q",
            y="Y:Q",
            color=alt.Color(
                "Quantile:N",
                scale=alt.Scale(
                    domain=[f"τ={tau}" for tau in quantiles],
                    range=["#d7191c", "#fdae61", "#ffffbf", "#abd9e9", "#2c7bb6"],
                ),
            ),
        )
    )

    chart = (points + lines).properties(
        width=600,
        height=400,
        title="Quantile Regression Captures Heteroscedasticity (Increasing Spread)",
    )

    chart.save(str(OUTPUT_DIR / "heteroscedastic_comparison.png"))
    print("Generated: heteroscedastic_comparison.png")


def generate_quantile_prediction_intervals() -> None:
    """Generate illustration of prediction intervals from quantile regression."""
    n_train = 150
    X_train = np.random.uniform(0, 10, n_train)
    y_train = 2 + 3 * X_train + np.random.normal(0, 1 + 0.3 * X_train, n_train)

    # Test points
    X_test = np.linspace(0, 10, 100)

    # Sort for smooth quantiles
    sort_idx = np.argsort(X_train)
    X_sorted = X_train[sort_idx]
    y_sorted = y_train[sort_idx]

    # Rolling quantiles on training data
    window = 30
    quantiles = [0.025, 0.5, 0.975]

    # Interpolate to test points
    y_quantiles_test = {}
    for tau in quantiles:
        y_quant_train = np.zeros(n_train)
        for i in range(n_train):
            start = max(0, i - window // 2)
            end = min(n_train, i + window // 2)
            y_window = y_sorted[start:end]
            y_quant_train[i] = np.percentile(y_window, tau * 100)

        # Interpolate to test points
        y_quantiles_test[tau] = np.interp(X_test, X_sorted, y_quant_train)

    # Create DataFrame
    df_train = pd.DataFrame({"X": X_train, "Y": y_train, "Type": "Training Data"})

    df_median = pd.DataFrame(
        {
            "X": X_test,
            "Y": y_quantiles_test[0.5],
            "Type": "Median Prediction (τ=0.5)",
        }
    )

    df_interval = pd.DataFrame(
        {
            "X": X_test,
            "Lower": y_quantiles_test[0.025],
            "Upper": y_quantiles_test[0.975],
        }
    )

    # Create chart
    points = (
        alt.Chart(df_train)
        .mark_circle(size=40, opacity=0.4, color="gray")
        .encode(x=alt.X("X:Q", title="X"), y=alt.Y("Y:Q", title="Y"))
    )

    median_line = (
        alt.Chart(df_median)
        .mark_line(strokeWidth=3, color="red")
        .encode(x="X:Q", y="Y:Q")
    )

    area = (
        alt.Chart(df_interval)
        .mark_area(opacity=0.3, color="red")
        .encode(
            x=alt.X("X:Q"),
            y=alt.Y("Lower:Q", title="Y"),
            y2=alt.Y2("Upper:Q"),
        )
    )

    chart = (area + points + median_line).properties(
        width=600,
        height=400,
        title="95% Prediction Intervals from Quantile Regression",
    )

    chart.save(str(OUTPUT_DIR / "quantile_prediction_intervals.png"))
    print("Generated: quantile_prediction_intervals.png")


def generate_timeseries_quantile_regression() -> None:
    """Generate time series with quantile forecasts."""
    # Historical data
    n_hist = 100
    t_hist = np.arange(n_hist)
    trend = 0.05 * t_hist
    seasonal = 3 * np.sin(2 * np.pi * t_hist / 20)
    noise = np.random.normal(0, 1, n_hist)
    y_hist = 10 + trend + seasonal + noise

    # Forecast horizon
    n_forecast = 20
    t_forecast = np.arange(n_hist, n_hist + n_forecast)
    trend_forecast = 0.05 * t_forecast
    seasonal_forecast = 3 * np.sin(2 * np.pi * t_forecast / 20)

    # Quantile forecasts (simulate with varying uncertainty)
    quantiles = [0.1, 0.5, 0.9]
    y_forecasts = {}
    for tau in quantiles:
        offset = 1.5 * (tau - 0.5) * 2  # Spread increases
        y_forecasts[tau] = 10 + trend_forecast + seasonal_forecast + offset

    # Create DataFrames
    df_hist = pd.DataFrame(
        {"Time": t_hist, "Temperature": y_hist, "Type": "Observed"}
    )

    df_forecasts = []
    for tau in quantiles:
        df_forecasts.append(
            pd.DataFrame(
                {
                    "Time": t_forecast,
                    "Temperature": y_forecasts[tau],
                    "Quantile": f"τ={tau}",
                }
            )
        )

    df_forecast = pd.concat(df_forecasts, ignore_index=True)

    # Prediction band
    df_band = pd.DataFrame(
        {
            "Time": t_forecast,
            "Lower": y_forecasts[0.1],
            "Upper": y_forecasts[0.9],
        }
    )

    # Create chart
    hist_line = (
        alt.Chart(df_hist)
        .mark_line(strokeWidth=2, color="black")
        .encode(
            x=alt.X("Time:Q", title="Time"), y=alt.Y("Temperature:Q", title="Temperature (°C)")
        )
    )

    forecast_lines = (
        alt.Chart(df_forecast)
        .mark_line(strokeWidth=2, strokeDash=[5, 5])
        .encode(
            x="Time:Q",
            y="Temperature:Q",
            color=alt.Color(
                "Quantile:N",
                scale=alt.Scale(
                    domain=["τ=0.1", "τ=0.5", "τ=0.9"],
                    range=["steelblue", "orange", "purple"],
                ),
            ),
        )
    )

    band = (
        alt.Chart(df_band)
        .mark_area(opacity=0.2, color="orange")
        .encode(
            x="Time:Q", y=alt.Y("Lower:Q", title="Temperature (°C)"), y2="Upper:Q"
        )
    )

    # Vertical line at forecast start
    vline = (
        alt.Chart(pd.DataFrame({"x": [n_hist]}))
        .mark_rule(strokeWidth=2, color="red", strokeDash=[3, 3])
        .encode(x="x:Q")
    )

    chart = (hist_line + band + forecast_lines + vline).properties(
        width=600, height=400, title="Time Series Quantile Regression Forecast"
    )

    chart.save(str(OUTPUT_DIR / "timeseries_quantile_regression.png"))
    print("Generated: timeseries_quantile_regression.png")


def generate_bootstrap_quantile_coefficients() -> None:
    """Generate bootstrap distribution of quantile regression coefficients."""
    # Simulate bootstrap samples
    n_bootstrap = 1000
    beta_1_median = np.random.normal(3.0, 0.15, n_bootstrap)
    beta_1_q10 = np.random.normal(2.5, 0.20, n_bootstrap)
    beta_1_q90 = np.random.normal(3.5, 0.18, n_bootstrap)

    df = pd.DataFrame(
        {
            "Coefficient": np.concatenate(
                [beta_1_median, beta_1_q10, beta_1_q90]
            ),
            "Quantile": (
                ["τ=0.5 (Median)"] * n_bootstrap
                + ["τ=0.1"] * n_bootstrap
                + ["τ=0.9"] * n_bootstrap
            ),
        }
    )

    chart = (
        alt.Chart(df)
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X(
                "Coefficient:Q",
                bin=alt.Bin(maxbins=40),
                title="Slope Coefficient β₁",
            ),
            y=alt.Y("count():Q", title="Frequency"),
            color=alt.Color(
                "Quantile:N",
                scale=alt.Scale(
                    domain=["τ=0.1", "τ=0.5 (Median)", "τ=0.9"],
                    range=["steelblue", "orange", "purple"],
                ),
            ),
            column=alt.Column("Quantile:N", title=None),
        )
        .properties(width=180, height=300)
    )

    chart.save(str(OUTPUT_DIR / "bootstrap_quantile_coefficients.png"))
    print("Generated: bootstrap_quantile_coefficients.png")


def generate_crossing_quantiles_problem() -> None:
    """Generate illustration of crossing quantiles problem."""
    X = np.linspace(0, 10, 50)

    # Correct quantiles (non-crossing)
    y_q10_correct = 1 + 0.5 * X
    y_q50_correct = 2 + 0.5 * X
    y_q90_correct = 3 + 0.5 * X

    # Incorrect quantiles (with crossing)
    y_q10_wrong = 1 + 0.5 * X
    y_q50_wrong = 2 + 0.5 * X + 0.3 * np.sin(X)
    y_q90_wrong = 3 + 0.5 * X - 0.5 * np.sin(X)  # Crosses median

    df_correct = pd.DataFrame(
        {
            "X": np.tile(X, 3),
            "Y": np.concatenate([y_q10_correct, y_q50_correct, y_q90_correct]),
            "Quantile": ["τ=0.1"] * 50 + ["τ=0.5"] * 50 + ["τ=0.9"] * 50,
            "Type": "Correct (No Crossing)",
        }
    )

    df_wrong = pd.DataFrame(
        {
            "X": np.tile(X, 3),
            "Y": np.concatenate([y_q10_wrong, y_q50_wrong, y_q90_wrong]),
            "Quantile": ["τ=0.1"] * 50 + ["τ=0.5"] * 50 + ["τ=0.9"] * 50,
            "Type": "Incorrect (Quantiles Cross)",
        }
    )

    df = pd.concat([df_correct, df_wrong], ignore_index=True)

    chart = (
        alt.Chart(df)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("X:Q", title="X"),
            y=alt.Y("Y:Q", title="Y"),
            color=alt.Color(
                "Quantile:N",
                scale=alt.Scale(
                    domain=["τ=0.1", "τ=0.5", "τ=0.9"],
                    range=["steelblue", "orange", "purple"],
                ),
            ),
        )
        .properties(width=280, height=300)
        .facet(column=alt.Column("Type:N", title=None))
    )

    chart.save(str(OUTPUT_DIR / "crossing_quantiles_problem.png"))
    print("Generated: crossing_quantiles_problem.png")


def generate_quantile_climate_forecast() -> None:
    """Generate climate forecast with quantile predictions."""
    # Historical temperature
    n_hist = 150
    t_hist = np.arange(n_hist)
    seasonal = 10 * np.sin(2 * np.pi * t_hist / 365)
    temp_hist = 15 + seasonal + np.random.normal(0, 2, n_hist)

    # Forecast
    n_forecast = 60
    t_forecast = np.arange(n_hist, n_hist + n_forecast)
    seasonal_forecast = 10 * np.sin(2 * np.pi * t_forecast / 365)

    # Quantile forecasts with increasing uncertainty
    uncertainty_scale = np.linspace(1.0, 1.5, n_forecast)
    y_median = 15 + seasonal_forecast
    y_q10 = y_median - 2.5 * uncertainty_scale
    y_q90 = y_median + 2.5 * uncertainty_scale

    # Create DataFrames
    df_hist = pd.DataFrame({"Day": t_hist, "Temperature": temp_hist})

    df_forecast_median = pd.DataFrame(
        {"Day": t_forecast, "Temperature": y_median, "Type": "Median"}
    )

    df_band = pd.DataFrame(
        {"Day": t_forecast, "Lower": y_q10, "Upper": y_q90}
    )

    # Create chart
    hist_line = (
        alt.Chart(df_hist)
        .mark_line(strokeWidth=1.5, color="black", opacity=0.7)
        .encode(
            x=alt.X("Day:Q", title="Day of Year"),
            y=alt.Y("Temperature:Q", title="Temperature (°C)", scale=alt.Scale(domain=[0, 30])),
        )
    )

    forecast_line = (
        alt.Chart(df_forecast_median)
        .mark_line(strokeWidth=2.5, color="red")
        .encode(x="Day:Q", y="Temperature:Q")
    )

    band = (
        alt.Chart(df_band)
        .mark_area(opacity=0.3, color="red")
        .encode(x="Day:Q", y=alt.Y("Lower:Q", title="Temperature (°C)"), y2="Upper:Q")
    )

    # Vertical line
    vline = (
        alt.Chart(pd.DataFrame({"x": [n_hist]}))
        .mark_rule(strokeWidth=2, color="darkred", strokeDash=[5, 5])
        .encode(x="x:Q")
    )

    chart = (hist_line + band + forecast_line + vline).properties(
        width=600,
        height=400,
        title="Climate Temperature Forecast with 80% Prediction Band",
    )

    chart.save(str(OUTPUT_DIR / "quantile_climate_forecast.png"))
    print("Generated: quantile_climate_forecast.png")


def generate_energy_quantile_forecast() -> None:
    """Generate energy load forecast with high quantiles."""
    # Historical load with daily pattern
    n_hist = 168  # 1 week of hourly data
    t_hist = np.arange(n_hist)
    hour_of_day = t_hist % 24
    daily_pattern = 50 + 30 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
    load_hist = daily_pattern + np.random.normal(0, 5, n_hist)

    # Forecast
    n_forecast = 24  # 1 day ahead
    t_forecast = np.arange(n_hist, n_hist + n_forecast)
    hour_forecast = t_forecast % 24
    daily_pattern_forecast = 50 + 30 * np.sin(
        2 * np.pi * (hour_forecast - 6) / 24
    )

    # Quantiles: especially high quantiles for capacity planning
    y_median = daily_pattern_forecast
    y_q50 = y_median
    y_q90 = y_median + 10
    y_q95 = y_median + 15

    # Create DataFrames
    df_hist = pd.DataFrame({"Hour": t_hist, "Load": load_hist})

    df_forecast = pd.DataFrame(
        {
            "Hour": np.tile(t_forecast, 3),
            "Load": np.concatenate([y_q50, y_q90, y_q95]),
            "Quantile": (
                ["τ=0.5 (Median)"] * n_forecast
                + ["τ=0.9"] * n_forecast
                + ["τ=0.95 (Capacity)"] * n_forecast
            ),
        }
    )

    # Create chart
    hist_line = (
        alt.Chart(df_hist)
        .mark_line(strokeWidth=1.5, color="gray")
        .encode(
            x=alt.X("Hour:Q", title="Hour"),
            y=alt.Y("Load:Q", title="Energy Load (MW)"),
        )
    )

    forecast_lines = (
        alt.Chart(df_forecast)
        .mark_line(strokeWidth=2.5, strokeDash=[5, 5])
        .encode(
            x="Hour:Q",
            y="Load:Q",
            color=alt.Color(
                "Quantile:N",
                scale=alt.Scale(
                    domain=["τ=0.5 (Median)", "τ=0.9", "τ=0.95 (Capacity)"],
                    range=["orange", "purple", "red"],
                ),
            ),
        )
    )

    # Vertical line
    vline = (
        alt.Chart(pd.DataFrame({"x": [n_hist]}))
        .mark_rule(strokeWidth=2, color="darkred", strokeDash=[3, 3])
        .encode(x="x:Q")
    )

    chart = (hist_line + forecast_lines + vline).properties(
        width=600,
        height=400,
        title="Energy Load Forecast: High Quantiles for Capacity Planning",
    )

    chart.save(str(OUTPUT_DIR / "energy_quantile_forecast.png"))
    print("Generated: energy_quantile_forecast.png")


def generate_heteroscedasticity_detection() -> None:
    """Generate illustration showing how to detect heteroscedasticity with quantiles."""
    n = 200
    X = np.random.uniform(0, 10, n)

    # Two scenarios: homoscedastic vs heteroscedastic
    y_homo = 2 + 3 * X + np.random.normal(0, 2, n)  # Constant variance
    y_hetero = 2 + 3 * X + np.random.normal(0, 0.5 + 0.4 * X, n)  # Increasing variance

    # Sort
    sort_idx_homo = np.argsort(X)
    X_sorted = X[sort_idx_homo]
    y_homo_sorted = y_homo[sort_idx_homo]

    sort_idx_hetero = np.argsort(X)
    y_hetero_sorted = y_hetero[sort_idx_hetero]

    # Rolling quantiles
    window = 40
    quantiles = [0.1, 0.5, 0.9]

    y_quants_homo = {tau: np.zeros(n) for tau in quantiles}
    y_quants_hetero = {tau: np.zeros(n) for tau in quantiles}

    for i in range(n):
        start = max(0, i - window // 2)
        end = min(n, i + window // 2)

        window_homo = y_homo_sorted[start:end]
        window_hetero = y_hetero_sorted[start:end]

        for tau in quantiles:
            y_quants_homo[tau][i] = np.percentile(window_homo, tau * 100)
            y_quants_hetero[tau][i] = np.percentile(window_hetero, tau * 100)

    # Create DataFrames
    df_homo = []
    for tau in quantiles:
        df_homo.append(
            pd.DataFrame(
                {
                    "X": X_sorted,
                    "Y": y_quants_homo[tau],
                    "Quantile": f"τ={tau}",
                    "Type": "Homoscedastic (Parallel Quantiles)",
                }
            )
        )

    df_hetero = []
    for tau in quantiles:
        df_hetero.append(
            pd.DataFrame(
                {
                    "X": X_sorted,
                    "Y": y_quants_hetero[tau],
                    "Quantile": f"τ={tau}",
                    "Type": "Heteroscedastic (Diverging Quantiles)",
                }
            )
        )

    df = pd.concat(df_homo + df_hetero, ignore_index=True)

    chart = (
        alt.Chart(df)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("X:Q", title="X"),
            y=alt.Y("Y:Q", title="Y"),
            color=alt.Color(
                "Quantile:N",
                scale=alt.Scale(
                    domain=["τ=0.1", "τ=0.5", "τ=0.9"],
                    range=["steelblue", "orange", "purple"],
                ),
            ),
        )
        .properties(width=280, height=300)
        .facet(column=alt.Column("Type:N", title=None))
    )

    chart.save(str(OUTPUT_DIR / "heteroscedasticity_detection.png"))
    print("Generated: heteroscedasticity_detection.png")


def main() -> None:
    """Generate all illustrations for quantile regression documentation."""
    print("Generating illustrations for quantile regression documentation...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    generate_ols_vs_quantile_comparison()
    generate_check_loss_function()
    generate_heteroscedastic_comparison()
    generate_quantile_prediction_intervals()
    generate_timeseries_quantile_regression()
    generate_bootstrap_quantile_coefficients()
    generate_crossing_quantiles_problem()
    generate_quantile_climate_forecast()
    generate_energy_quantile_forecast()
    generate_heteroscedasticity_detection()

    print(f"\nAll illustrations generated successfully in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
