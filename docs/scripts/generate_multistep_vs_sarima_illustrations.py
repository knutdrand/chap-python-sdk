"""Generate illustrations comparing multistep recursive regression with (S)ARIMA(X)."""

import numpy as np
import pandas as pd
import altair as alt
from pathlib import Path

np.random.seed(42)

OUTPUT_DIR = Path(__file__).parent.parent / "images"
OUTPUT_DIR.mkdir(exist_ok=True)

CHART_WIDTH = 560
CHART_HEIGHT = 300


def generate_synthetic_disease_data(
    n: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic monthly disease case data with trend, seasonality, and noise.

    Returns time index, raw counts, and temperature covariate.
    """
    t = np.arange(n)
    seasonal = 15 * np.sin(2 * np.pi * t / 12)
    trend = 0.1 * t
    noise = np.random.normal(0, 3, n)
    cases = 50 + trend + seasonal + noise
    cases = np.maximum(cases, 0)
    temperature = 20 + 8 * np.sin(2 * np.pi * t / 12 + 0.5) + np.random.normal(0, 1, n)
    return t, cases, temperature


def plot_lag_feature_construction() -> None:
    """Show how the multistep model constructs lag features from a time series."""
    _, cases, _ = generate_synthetic_disease_data(24)
    cases = np.round(cases, 1)
    n_lags = 3

    # Show the table-like construction for a window
    rows = []
    for i in range(n_lags, min(n_lags + 8, len(cases))):
        row = {
            "time": f"t={i}",
            "time_idx": i,
            "y(t)": cases[i],
        }
        for lag in range(n_lags, 0, -1):
            row[f"y(t-{lag})"] = cases[i - lag]
        rows.append(row)

    # Visualize a time series with arrows showing which lags feed into the prediction
    t_show = np.arange(12)
    df = pd.DataFrame({"t": t_show, "y": cases[:12]})

    # Highlight a target time point and its lags
    target_t = 6
    lag_points = []
    for lag in range(1, n_lags + 1):
        lag_points.append(
            {"t": target_t - lag, "y": cases[target_t - lag], "role": f"Lag {lag} feature"}
        )
    lag_points.append({"t": target_t, "y": cases[target_t], "role": "Target y(t)"})
    lag_df = pd.DataFrame(lag_points)

    base = (
        alt.Chart(df)
        .mark_line(color="grey", strokeWidth=1.5, opacity=0.6)
        .encode(
            x=alt.X("t:Q", title="Time step", scale=alt.Scale(domain=[-0.5, 11.5])),
            y=alt.Y("y:Q", title="Disease cases"),
        )
    )

    all_pts = (
        alt.Chart(df)
        .mark_circle(size=40, color="grey", opacity=0.4)
        .encode(x="t:Q", y="y:Q")
    )

    highlight = (
        alt.Chart(lag_df)
        .mark_circle(size=120)
        .encode(
            x="t:Q",
            y="y:Q",
            color=alt.Color(
                "role:N",
                scale=alt.Scale(
                    domain=["Lag 3 feature", "Lag 2 feature", "Lag 1 feature", "Target y(t)"],
                    range=["#4c78a8", "#72b7b2", "#54a24b", "#e45756"],
                ),
                title="Role",
            ),
        )
    )

    # Draw arrows from lags to target
    arrow_data = []
    for lag in range(1, n_lags + 1):
        arrow_data.append(
            {
                "x": target_t - lag,
                "y": cases[target_t - lag],
                "x2": target_t,
                "y2": cases[target_t],
            }
        )
    arrow_df = pd.DataFrame(arrow_data)
    arrows = (
        alt.Chart(arrow_df)
        .mark_rule(color="#999", strokeDash=[4, 2], strokeWidth=1)
        .encode(x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q")
    )

    chart = (base + all_pts + arrows + highlight).properties(
        width=CHART_WIDTH,
        height=CHART_HEIGHT,
        title="Multistep Model: Lag Feature Construction",
    )
    chart.save(str(OUTPUT_DIR / "multistep-lag-features.png"), scale_factor=2)
    print("  Saved multistep-lag-features.png")


def plot_recursive_prediction() -> None:
    """Show the recursive prediction loop: each prediction feeds back as a lag."""
    np.random.seed(42)
    n_history = 8
    n_forecast = 5
    n_lags = 3

    # Simulate a simple AR-like process for illustration
    history = [50.0, 52.0, 48.0, 55.0, 53.0, 49.0, 56.0, 54.0]
    forecast_mean = [57.0, 55.5, 58.0, 56.5, 59.0]

    t_hist = list(range(n_history))
    t_fore = list(range(n_history, n_history + n_forecast))

    # Create trajectories showing feedback
    n_traj = 20
    trajectories = []
    for traj_i in range(n_traj):
        vals = list(history[-n_lags:])
        traj_vals = []
        for step in range(n_forecast):
            pred = forecast_mean[step] + np.random.normal(0, 2)
            traj_vals.append(pred)
            vals.append(pred)
            vals = vals[-n_lags:]
        for step, val in enumerate(traj_vals):
            trajectories.append(
                {"t": n_history + step, "y": val, "trajectory": traj_i}
            )

    hist_df = pd.DataFrame({"t": t_hist, "y": history})
    traj_df = pd.DataFrame(trajectories)

    hist_line = (
        alt.Chart(hist_df)
        .mark_line(color="steelblue", strokeWidth=2)
        .encode(
            x=alt.X("t:Q", title="Time step", scale=alt.Scale(domain=[-0.5, 13.5])),
            y=alt.Y("y:Q", title="Disease cases", scale=alt.Scale(domain=[40, 70])),
        )
    )
    hist_pts = (
        alt.Chart(hist_df)
        .mark_circle(size=50, color="steelblue")
        .encode(x="t:Q", y="y:Q")
    )

    traj_lines = (
        alt.Chart(traj_df)
        .mark_line(opacity=0.25, strokeWidth=1)
        .encode(
            x="t:Q",
            y="y:Q",
            color=alt.value("#e45756"),
            detail="trajectory:N",
        )
    )

    # Vertical line separating history from forecast
    rule_df = pd.DataFrame({"t": [n_history - 0.5]})
    rule = (
        alt.Chart(rule_df)
        .mark_rule(color="black", strokeDash=[4, 4], strokeWidth=1)
        .encode(x="t:Q")
    )

    # Labels
    label_df = pd.DataFrame(
        [
            {"t": 3.5, "y": 68, "label": "Observed history"},
            {"t": 10.5, "y": 68, "label": "Recursive forecasts"},
        ]
    )
    labels = (
        alt.Chart(label_df)
        .mark_text(fontSize=12, fontWeight="bold")
        .encode(x="t:Q", y="y:Q", text="label:N")
    )

    chart = (hist_line + hist_pts + traj_lines + rule + labels).properties(
        width=CHART_WIDTH,
        height=CHART_HEIGHT,
        title="Recursive Multi-Step Prediction: Each Forecast Feeds Back as a Lag",
    )
    chart.save(str(OUTPUT_DIR / "multistep-recursive-prediction.png"), scale_factor=2)
    print("  Saved multistep-recursive-prediction.png")


def plot_ar_vs_arima_differencing() -> None:
    """Contrast raw lag features (multistep) vs differenced series (ARIMA)."""
    np.random.seed(42)
    t = np.arange(60)
    trend = 0.3 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 12)
    noise = np.random.normal(0, 2, len(t))
    y = 50 + trend + seasonal + noise

    y_diff1 = np.diff(y, n=1)

    raw_df = pd.DataFrame({"t": t, "y": y, "series": "Original y(t)"})
    diff_df = pd.DataFrame(
        {"t": t[1:], "y": y_diff1, "series": "First difference Δy(t) = y(t) - y(t-1)"}
    )
    combined = pd.concat([raw_df, diff_df])

    chart = (
        alt.Chart(combined)
        .mark_line(strokeWidth=1.5)
        .encode(
            x=alt.X("t:Q", title="Time step"),
            y=alt.Y("y:Q", title="Value"),
            color=alt.Color("series:N", title="Series"),
            row=alt.Row("series:N", title=None, header=alt.Header(labelFontSize=12)),
        )
        .properties(width=CHART_WIDTH, height=180, title="")
        .resolve_scale(y="independent")
    )

    chart.save(
        str(OUTPUT_DIR / "multistep-vs-arima-differencing.png"), scale_factor=2
    )
    print("  Saved multistep-vs-arima-differencing.png")


def plot_feature_role_comparison() -> None:
    """Show how lags enter as generic ML features vs as constrained linear AR coefficients."""
    np.random.seed(42)
    n_lags = 4
    lags = [f"y(t-{i})" for i in range(1, n_lags + 1)]

    # Multistep: feature importances from a tree model (non-linear, flexible)
    multistep_importance = [0.45, 0.25, 0.18, 0.12]

    # ARIMA AR(4): linear coefficients (constrained to stationarity)
    arima_coefs = [0.60, -0.20, 0.15, -0.05]

    ms_df = pd.DataFrame(
        {"lag": lags, "value": multistep_importance, "model": "Multistep (feature importance)"}
    )
    ar_df = pd.DataFrame(
        {"lag": lags, "value": arima_coefs, "model": "ARIMA AR(4) coefficients"}
    )
    combined = pd.concat([ms_df, ar_df])

    chart = (
        alt.Chart(combined)
        .mark_bar()
        .encode(
            x=alt.X("lag:N", title="Lag feature", sort=lags),
            y=alt.Y("value:Q", title="Weight / Importance"),
            color=alt.Color(
                "model:N",
                scale=alt.Scale(range=["#4c78a8", "#e45756"]),
                title="Model",
            ),
            xOffset="model:N",
        )
        .properties(
            width=CHART_WIDTH,
            height=CHART_HEIGHT,
            title="Role of Lagged Values: Feature Importance vs Linear Coefficients",
        )
    )
    chart.save(str(OUTPUT_DIR / "multistep-vs-arima-lag-roles.png"), scale_factor=2)
    print("  Saved multistep-vs-arima-lag-roles.png")


def plot_ma_component_illustration() -> None:
    """Illustrate the MA component of ARIMA — modeling past forecast errors."""
    np.random.seed(42)
    n = 40
    t = np.arange(n)

    # Simulate an MA(1) process for illustration
    theta = 0.7
    errors = np.random.normal(0, 2, n + 1)
    y_ma = np.array([errors[i] + theta * errors[i - 1] for i in range(1, n + 1)])
    y_ma += 50  # shift up for realism

    # Pure AR model residuals (no MA correction) — will be autocorrelated
    y_ar_pred = np.zeros(n)
    y_ar_pred[0] = y_ma[0]
    for i in range(1, n):
        y_ar_pred[i] = 0.3 * y_ma[i - 1] + 0.7 * 50
    residuals = y_ma - y_ar_pred

    res_df = pd.DataFrame({"t": t, "residual": residuals})

    # Annotate autocorrelation
    res_chart = (
        alt.Chart(res_df)
        .mark_bar(width=4)
        .encode(
            x=alt.X("t:Q", title="Time step"),
            y=alt.Y("residual:Q", title="Residual (actual − predicted)"),
            color=alt.condition(
                alt.datum.residual > 0, alt.value("#4c78a8"), alt.value("#e45756")
            ),
        )
        .properties(
            width=CHART_WIDTH,
            height=CHART_HEIGHT,
            title="ARIMA's MA Component: Correcting Patterns in Forecast Errors",
        )
    )

    zero_line = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(color="black", strokeWidth=0.5)
        .encode(y="y:Q")
    )

    chart = res_chart + zero_line
    chart.save(str(OUTPUT_DIR / "arima-ma-residuals.png"), scale_factor=2)
    print("  Saved arima-ma-residuals.png")


def plot_seasonal_structure() -> None:
    """Contrast SARIMA's explicit seasonal terms with multistep's lag-based approach."""
    np.random.seed(42)
    t = np.arange(48)
    seasonal = 15 * np.sin(2 * np.pi * t / 12)
    noise = np.random.normal(0, 2, 48)
    y = 50 + seasonal + noise

    df = pd.DataFrame({"t": t, "y": y})

    # Highlight seasonal lag (lag 12) connections
    connections = []
    for base_t in [24, 25, 26, 27]:
        connections.append(
            {
                "t1": base_t - 12,
                "y1": y[base_t - 12],
                "t2": base_t,
                "y2": y[base_t],
                "label": "SARIMA: y(t-12)",
            }
        )
    for base_t in [24, 25, 26, 27]:
        connections.append(
            {
                "t1": base_t - 1,
                "y1": y[base_t - 1],
                "t2": base_t,
                "y2": y[base_t],
                "label": "Both: y(t-1)",
            }
        )

    conn_df = pd.DataFrame(connections)

    line = (
        alt.Chart(df)
        .mark_line(color="grey", strokeWidth=1.5, opacity=0.6)
        .encode(
            x=alt.X("t:Q", title="Month", scale=alt.Scale(domain=[-0.5, 48])),
            y=alt.Y("y:Q", title="Disease cases"),
        )
    )
    pts = (
        alt.Chart(df).mark_circle(size=30, color="grey", opacity=0.4).encode(x="t:Q", y="y:Q")
    )

    arrows = (
        alt.Chart(conn_df)
        .mark_rule(strokeWidth=1.5, opacity=0.7)
        .encode(
            x="t1:Q",
            y="y1:Q",
            x2="t2:Q",
            y2="y2:Q",
            color=alt.Color(
                "label:N",
                scale=alt.Scale(
                    domain=["SARIMA: y(t-12)", "Both: y(t-1)"],
                    range=["#e45756", "#4c78a8"],
                ),
                title="Seasonal Connection",
            ),
            strokeDash=alt.StrokeDash(
                "label:N",
                scale=alt.Scale(
                    domain=["SARIMA: y(t-12)", "Both: y(t-1)"],
                    range=[[1, 0], [4, 2]],
                ),
            ),
        )
    )

    chart = (line + pts + arrows).properties(
        width=CHART_WIDTH,
        height=CHART_HEIGHT,
        title="Seasonal Structure: SARIMA's Explicit Seasonal Lag vs Short Lags",
    )
    chart.save(str(OUTPUT_DIR / "sarima-seasonal-connections.png"), scale_factor=2)
    print("  Saved sarima-seasonal-connections.png")


def plot_uncertainty_comparison() -> None:
    """Compare residual bootstrap (multistep) vs parametric intervals (ARIMA)."""
    np.random.seed(42)
    n_hist = 20
    n_fore = 10

    t_hist = np.arange(n_hist)
    y_hist = 50 + 0.3 * t_hist + np.random.normal(0, 2, n_hist)

    t_fore = np.arange(n_hist, n_hist + n_fore)
    y_fore_mean = 50 + 0.3 * t_fore

    # Multistep: bootstrap trajectories (fan out, possibly non-symmetric)
    n_traj = 100
    trajectories = np.zeros((n_traj, n_fore))
    for i in range(n_traj):
        for step in range(n_fore):
            noise = np.random.normal(0, 2) + 0.3 * np.random.exponential(0.5)  # slight skew
            if step == 0:
                trajectories[i, step] = y_fore_mean[step] + noise
            else:
                trajectories[i, step] = trajectories[i, step - 1] + 0.3 + noise

    ms_lo = np.percentile(trajectories, 10, axis=0)
    ms_hi = np.percentile(trajectories, 90, axis=0)
    ms_median = np.median(trajectories, axis=0)

    # ARIMA: parametric Gaussian intervals (symmetric, growing)
    ar_std = np.array([2.0 * np.sqrt(s + 1) for s in range(n_fore)])
    ar_mean = y_fore_mean
    ar_lo = ar_mean - 1.645 * ar_std
    ar_hi = ar_mean + 1.645 * ar_std

    hist_df = pd.DataFrame({"t": t_hist, "y": y_hist})

    ms_band_df = pd.DataFrame({"t": t_fore, "lo": ms_lo, "hi": ms_hi, "median": ms_median})
    ar_band_df = pd.DataFrame({"t": t_fore, "lo": ar_lo, "hi": ar_hi, "mean": ar_mean})

    # --- Multistep panel ---
    ms_hist = (
        alt.Chart(hist_df)
        .mark_line(color="steelblue", strokeWidth=2)
        .encode(x=alt.X("t:Q", title="Time"), y=alt.Y("y:Q", title="Cases", scale=alt.Scale(domain=[35, 85])))
    )
    ms_band = (
        alt.Chart(ms_band_df)
        .mark_area(opacity=0.3, color="#e45756")
        .encode(x="t:Q", y="lo:Q", y2="hi:Q")
    )
    ms_line = (
        alt.Chart(ms_band_df)
        .mark_line(color="#e45756", strokeWidth=2)
        .encode(x="t:Q", y="median:Q")
    )
    ms_chart = (ms_hist + ms_band + ms_line).properties(
        width=CHART_WIDTH // 2, height=220, title="Multistep: Residual Bootstrap"
    )

    # --- ARIMA panel ---
    ar_hist = (
        alt.Chart(hist_df)
        .mark_line(color="steelblue", strokeWidth=2)
        .encode(x=alt.X("t:Q", title="Time"), y=alt.Y("y:Q", title="Cases", scale=alt.Scale(domain=[35, 85])))
    )
    ar_band = (
        alt.Chart(ar_band_df)
        .mark_area(opacity=0.3, color="#54a24b")
        .encode(x="t:Q", y="lo:Q", y2="hi:Q")
    )
    ar_line = (
        alt.Chart(ar_band_df)
        .mark_line(color="#54a24b", strokeWidth=2)
        .encode(x="t:Q", y="mean:Q")
    )
    ar_chart = (ar_hist + ar_band + ar_line).properties(
        width=CHART_WIDTH // 2, height=220, title="ARIMA: Gaussian Prediction Intervals"
    )

    chart = ms_chart | ar_chart
    chart.save(str(OUTPUT_DIR / "uncertainty-bootstrap-vs-parametric.png"), scale_factor=2)
    print("  Saved uncertainty-bootstrap-vs-parametric.png")


if __name__ == "__main__":
    print("Generating multistep vs SARIMA illustrations...")
    plot_lag_feature_construction()
    plot_recursive_prediction()
    plot_ar_vs_arima_differencing()
    plot_feature_role_comparison()
    plot_ma_component_illustration()
    plot_seasonal_structure()
    plot_uncertainty_comparison()
    print("Done.")
