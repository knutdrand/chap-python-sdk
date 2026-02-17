"""Generate illustrations for residual bootstrapping documentation using Altair."""

import numpy as np
import pandas as pd
import altair as alt
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "images"
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_original_data_and_fit() -> None:
    """Generate illustration of original data with fitted model."""
    # Generate data
    x = np.linspace(1, 10, 10)
    true_slope = 2.0
    true_intercept = 0.5
    noise = np.array([-0.2, -0.4, -0.8, -0.3, -0.4, -0.7, 0.3, -0.6, -0.4, -0.7])
    y = true_intercept + true_slope * x + noise
    y_fitted = true_intercept + true_slope * x

    # Create DataFrame
    df = pd.DataFrame({"X": x, "Y": y, "Y_fitted": y_fitted})

    # Create chart
    points = alt.Chart(df).mark_circle(size=100, color="steelblue").encode(
        x=alt.X("X:Q", title="X (Predictor)", scale=alt.Scale(domain=[0, 11])),
        y=alt.Y("Y:Q", title="Y (Response)", scale=alt.Scale(domain=[0, 22])),
    )

    line = alt.Chart(df).mark_line(color="red", strokeWidth=2).encode(
        x="X:Q",
        y="Y_fitted:Q"
    )

    chart = (points + line).properties(
        width=600,
        height=400,
        title="Original Data with Fitted Model: Y = 0.5 + 2.0*X"
    )

    chart.save(str(OUTPUT_DIR / "original_data_fitted_model.png"))
    print("Generated: original_data_fitted_model.png")


def generate_residuals_plot() -> None:
    """Generate residuals over time plot."""
    # Generate residuals
    t = np.arange(1, 11)
    residuals = np.array([-0.2, -0.4, -0.8, -0.3, -0.4, -0.7, 0.3, -0.6, -0.4, -0.7])

    df = pd.DataFrame({"Time": t, "Residual": residuals})

    # Create chart
    points = alt.Chart(df).mark_circle(size=100, color="steelblue").encode(
        x=alt.X("Time:Q", title="Time (t)", scale=alt.Scale(domain=[0, 11])),
        y=alt.Y("Residual:Q", title="Residual (ε̂)", scale=alt.Scale(domain=[-1, 0.5])),
    )

    zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
        color="gray",
        strokeDash=[5, 5]
    ).encode(y="y:Q")

    chart = (points + zero_line).properties(
        width=600,
        height=300,
        title="Residuals from Fitted Model"
    )

    chart.save(str(OUTPUT_DIR / "residuals_plot.png"))
    print("Generated: residuals_plot.png")


def generate_bootstrap_sample_comparison() -> None:
    """Generate comparison showing preserved trend in bootstrap sample."""
    # Original data
    x = np.linspace(1, 10, 10)
    y_fitted = 0.5 + 2.0 * x
    original_residuals = np.array([-0.2, -0.4, -0.8, -0.3, -0.4, -0.7, 0.3, -0.6, -0.4, -0.7])
    y_original = y_fitted + original_residuals

    # Bootstrap sample (resampled residuals)
    bootstrap_residuals = np.array([-0.8, 0.3, -0.4, -0.7, -0.3, -0.4, -0.8, -0.2, 0.3, -0.7])
    y_bootstrap = y_fitted + bootstrap_residuals

    # Create DataFrame
    df = pd.DataFrame({
        "X": np.concatenate([x, x]),
        "Y": np.concatenate([y_original, y_bootstrap]),
        "Type": ["Original"] * 10 + ["Bootstrap Sample"] * 10,
        "Y_fitted": np.concatenate([y_fitted, y_fitted])
    })

    # Create base chart (layer points and line first)
    base = alt.Chart(df).encode(
        x=alt.X("X:Q", title="X (Predictor)", scale=alt.Scale(domain=[0, 11])),
        y=alt.Y("Y:Q", title="Y (Response)", scale=alt.Scale(domain=[0, 22]))
    )

    points = base.mark_circle(size=100).encode(
        color=alt.Color("Type:N", scale=alt.Scale(domain=["Original", "Bootstrap Sample"],
                                                    range=["steelblue", "orange"]))
    )

    line = base.mark_line(color="red", strokeWidth=2).encode(
        y="Y_fitted:Q"
    )

    # Layer first, set properties, then facet
    chart = (points + line).properties(
        width=280,
        height=300
    ).facet(
        column=alt.Column("Type:N", title=None)
    )

    chart.save(str(OUTPUT_DIR / "bootstrap_sample_comparison.png"))
    print("Generated: bootstrap_sample_comparison.png")


def generate_acf_plot() -> None:
    """Generate autocorrelation function (ACF) plot for residuals."""
    lags = np.arange(0, 8)
    acf_values = np.array([1.0, 0.65, 0.45, 0.35, 0.25, 0.15, 0.08, 0.05])
    significance = 0.4

    df = pd.DataFrame({"Lag": lags, "ACF": acf_values})

    # Create bars
    bars = alt.Chart(df).mark_bar(color="steelblue", width=30).encode(
        x=alt.X("Lag:Q", title="Lag", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("ACF:Q", title="Autocorrelation", scale=alt.Scale(domain=[-0.2, 1.1])),
    )

    # Significance threshold
    threshold = alt.Chart(pd.DataFrame({"y": [significance]})).mark_rule(
        color="red",
        strokeDash=[5, 5],
        strokeWidth=2
    ).encode(y="y:Q")

    # Add annotation
    threshold_text = alt.Chart(pd.DataFrame({
        "x": [6],
        "y": [significance + 0.05],
        "text": ["Significance threshold"]
    })).mark_text(
        align="right",
        color="red",
        fontSize=11
    ).encode(
        x="x:Q",
        y="y:Q",
        text="text:N"
    )

    chart = (bars + threshold + threshold_text).properties(
        width=600,
        height=300,
        title="Residual Autocorrelation Function (ACF) - Use Block Bootstrap"
    )

    chart.save(str(OUTPUT_DIR / "residual_acf_plot.png"))
    print("Generated: residual_acf_plot.png")


def generate_block_bootstrap_illustration() -> None:
    """Generate illustration of block bootstrap for residuals."""
    # Original residuals
    t = np.arange(1, 13)
    residuals = np.array([-0.2, -0.3, 0.1, 0.2, -0.1, -0.4, 0.3, 0.2, -0.2, 0.1, -0.3, 0.1])

    # Bootstrap sample using blocks
    bootstrap_residuals = np.array([-0.1, -0.4, 0.3, 0.2, -0.3, 0.1, 0.2, -0.1, 0.3, 0.2, -0.2, 0.1])

    df = pd.DataFrame({
        "Time": np.concatenate([t, t]),
        "Residual": np.concatenate([residuals, bootstrap_residuals]),
        "Type": ["Original Residuals"] * 12 + ["Bootstrap Residuals (Block Length=4)"] * 12
    })

    # Create base chart
    base = alt.Chart(df).encode(
        x=alt.X("Time:Q", title="Time", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("Residual:Q", title="Residual (ε)", scale=alt.Scale(domain=[-0.5, 0.4]))
    )

    line = base.mark_line(point=True, strokeWidth=2).encode(
        color=alt.Color("Type:N", scale=alt.Scale(domain=["Original Residuals",
                                                            "Bootstrap Residuals (Block Length=4)"],
                                                    range=["steelblue", "orange"]))
    )

    # Add zero line
    zero_line = base.mark_rule(color="gray", strokeDash=[3, 3]).encode(y=alt.datum(0))

    # Layer first, then facet
    chart = (line + zero_line).properties(
        width=600,
        height=150
    ).facet(
        row=alt.Row("Type:N", title=None)
    )

    chart.save(str(OUTPUT_DIR / "block_bootstrap_illustration.png"))
    print("Generated: block_bootstrap_illustration.png")


def generate_residual_diagnostics() -> None:
    """Generate residual diagnostic plots for checking assumptions."""
    np.random.seed(42)

    # Good residuals (homoscedastic)
    t_good = np.linspace(1, 50, 50)
    residuals_good = np.random.normal(0, 0.5, 50)

    # Bad residuals (heteroscedastic)
    t_bad = np.linspace(1, 50, 50)
    residuals_bad = np.random.normal(0, 0.1 + 0.02 * t_bad, 50)

    df = pd.DataFrame({
        "Time": np.concatenate([t_good, t_bad]),
        "Residual": np.concatenate([residuals_good, residuals_bad]),
        "Pattern": ["Good: Homoscedastic"] * 50 + ["Bad: Heteroscedastic"] * 50
    })

    # Create base chart
    base = alt.Chart(df).encode(
        x=alt.X("Time:Q", title="Time (t)", scale=alt.Scale(domain=[0, 51])),
        y=alt.Y("Residual:Q", title="Residual (ε̂)")
    )

    points = base.mark_circle(size=60, opacity=0.6).encode(
        color=alt.Color("Pattern:N", scale=alt.Scale(domain=["Good: Homoscedastic",
                                                               "Bad: Heteroscedastic"],
                                                       range=["steelblue", "red"]))
    )

    # Add zero line
    zero_line = base.mark_rule(color="gray", strokeDash=[3, 3]).encode(y=alt.datum(0))

    # Layer first, then facet
    chart = (points + zero_line).properties(
        width=280,
        height=250
    ).facet(
        column=alt.Column("Pattern:N", title=None)
    ).resolve_scale(y="independent")

    chart.save(str(OUTPUT_DIR / "residual_diagnostics.png"))
    print("Generated: residual_diagnostics.png")


def generate_qq_plot() -> None:
    """Generate QQ plot for checking normality of residuals."""
    np.random.seed(42)

    # Generate normally distributed residuals
    n = 100
    residuals = np.random.normal(0, 1, n)
    residuals.sort()

    # Theoretical quantiles
    theoretical = np.random.normal(0, 1, n)
    theoretical.sort()

    df = pd.DataFrame({
        "Theoretical": theoretical,
        "Sample": residuals
    })

    # Create scatter plot
    points = alt.Chart(df).mark_circle(size=60, color="steelblue", opacity=0.6).encode(
        x=alt.X("Theoretical:Q", title="Theoretical Quantiles (Normal)"),
        y=alt.Y("Sample:Q", title="Sample Quantiles (Residuals)"),
    )

    # Add reference line
    line_df = pd.DataFrame({
        "x": [-3, 3],
        "y": [-3, 3]
    })
    line = alt.Chart(line_df).mark_line(color="red", strokeWidth=2, strokeDash=[5, 5]).encode(
        x="x:Q",
        y="y:Q"
    )

    chart = (points + line).properties(
        width=400,
        height=400,
        title="QQ Plot - Residuals vs Normal Distribution"
    )

    chart.save(str(OUTPUT_DIR / "qq_plot.png"))
    print("Generated: qq_plot.png")


def generate_residuals_vs_fitted() -> None:
    """Generate residuals vs fitted values plot for model diagnostics."""
    np.random.seed(42)

    # Good pattern (random)
    fitted_good = np.linspace(5, 20, 50)
    residuals_good = np.random.normal(0, 1, 50)

    # Bad pattern (non-linear relationship)
    fitted_bad = np.linspace(5, 20, 50)
    residuals_bad = 0.3 * (fitted_bad - 12.5) ** 2 - 3 + np.random.normal(0, 0.5, 50)

    df = pd.DataFrame({
        "Fitted": np.concatenate([fitted_good, fitted_bad]),
        "Residual": np.concatenate([residuals_good, residuals_bad]),
        "Pattern": ["Good: Random scatter"] * 50 + ["Bad: Non-linear pattern"] * 50
    })

    # Create base chart
    base = alt.Chart(df).encode(
        x=alt.X("Fitted:Q", title="Fitted Values (Ŷ)"),
        y=alt.Y("Residual:Q", title="Residual (ε̂)")
    )

    points = base.mark_circle(size=60, opacity=0.6).encode(
        color=alt.Color("Pattern:N", scale=alt.Scale(domain=["Good: Random scatter",
                                                               "Bad: Non-linear pattern"],
                                                       range=["steelblue", "red"]))
    )

    # Add zero line
    zero_line = base.mark_rule(color="gray", strokeDash=[3, 3]).encode(y=alt.datum(0))

    # Layer first, then facet
    chart = (points + zero_line).properties(
        width=280,
        height=250
    ).facet(
        column=alt.Column("Pattern:N", title=None)
    ).resolve_scale(y="independent")

    chart.save(str(OUTPUT_DIR / "residuals_vs_fitted.png"))
    print("Generated: residuals_vs_fitted.png")


def generate_bootstrap_distributions() -> None:
    """Generate comparison of parametric vs non-parametric bootstrap distributions."""
    np.random.seed(42)

    # Non-parametric (empirical)
    empirical_data = np.array([-0.8, -0.7, -0.4, -0.3, -0.2, 0.1, 0.2, 0.3])
    empirical_samples = np.random.choice(empirical_data, size=1000, replace=True)

    # Parametric (fitted normal)
    parametric_samples = np.random.normal(np.mean(empirical_data), np.std(empirical_data), 1000)

    df = pd.DataFrame({
        "Value": np.concatenate([empirical_samples, parametric_samples]),
        "Method": ["Non-parametric (Resample)"] * 1000 + ["Parametric (Fitted Normal)"] * 1000
    })

    # Create histograms
    chart = alt.Chart(df).mark_bar(opacity=0.7, binSpacing=1).encode(
        x=alt.X("Value:Q", bin=alt.Bin(maxbins=30), title="Residual Value"),
        y=alt.Y("count():Q", title="Frequency"),
        color=alt.Color("Method:N", scale=alt.Scale(domain=["Non-parametric (Resample)",
                                                              "Parametric (Fitted Normal)"],
                                                      range=["steelblue", "orange"])),
        row=alt.Row("Method:N", title=None)
    ).properties(
        width=600,
        height=150
    )

    chart.save(str(OUTPUT_DIR / "bootstrap_distributions.png"))
    print("Generated: bootstrap_distributions.png")


def generate_forecast_intervals() -> None:
    """Generate forecast intervals illustration using bootstrap."""
    np.random.seed(42)

    # Historical data
    t_past = np.arange(1, 51)
    y_past = 10 + 0.5 * t_past + np.random.normal(0, 2, 50)

    # Forecasts
    t_future = np.arange(51, 61)
    y_forecast = 10 + 0.5 * t_future

    # Bootstrap prediction intervals (simulate multiple bootstrap forecasts)
    bootstrap_forecasts = []
    for _ in range(100):
        noise = np.random.normal(0, 2, 10)
        bootstrap_forecasts.append(y_forecast + noise)

    bootstrap_forecasts = np.array(bootstrap_forecasts)
    lower_bound = np.percentile(bootstrap_forecasts, 2.5, axis=0)
    upper_bound = np.percentile(bootstrap_forecasts, 97.5, axis=0)

    # Create DataFrames
    df_past = pd.DataFrame({"Time": t_past, "Value": y_past, "Type": "Observed"})
    df_forecast = pd.DataFrame({"Time": t_future, "Value": y_forecast, "Type": "Forecast"})
    df_interval = pd.DataFrame({
        "Time": np.concatenate([t_future, t_future[::-1]]),
        "Value": np.concatenate([lower_bound, upper_bound[::-1]])
    })

    # Create chart
    past_line = alt.Chart(df_past).mark_line(color="steelblue", strokeWidth=2).encode(
        x=alt.X("Time:Q", title="Time", scale=alt.Scale(domain=[0, 62])),
        y=alt.Y("Value:Q", title="Response Variable", scale=alt.Scale(domain=[0, 50]))
    )

    forecast_line = alt.Chart(df_forecast).mark_line(
        color="red",
        strokeWidth=2,
        strokeDash=[5, 5]
    ).encode(
        x="Time:Q",
        y="Value:Q"
    )

    interval_area = alt.Chart(df_interval).mark_area(
        opacity=0.3,
        color="orange"
    ).encode(
        x="Time:Q",
        y="Value:Q"
    )

    # Add vertical line at forecast start
    vline = alt.Chart(pd.DataFrame({"x": [50.5]})).mark_rule(
        color="gray",
        strokeDash=[3, 3]
    ).encode(x="x:Q")

    chart = (interval_area + past_line + forecast_line + vline).properties(
        width=600,
        height=400,
        title="Bootstrap Forecast Intervals (95% CI)"
    )

    chart.save(str(OUTPUT_DIR / "forecast_intervals.png"))
    print("Generated: forecast_intervals.png")


def main() -> None:
    """Generate all illustrations for residual bootstrapping documentation."""
    print("Generating illustrations for residual bootstrapping documentation...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    generate_original_data_and_fit()
    generate_residuals_plot()
    generate_bootstrap_sample_comparison()
    generate_acf_plot()
    generate_block_bootstrap_illustration()
    generate_residual_diagnostics()
    generate_qq_plot()
    generate_residuals_vs_fitted()
    generate_bootstrap_distributions()
    generate_forecast_intervals()

    print(f"\nAll illustrations generated successfully in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
