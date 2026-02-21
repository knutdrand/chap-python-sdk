"""Generate illustrations for the SARIMAX guide."""

import numpy as np
import pandas as pd
import altair as alt
from pathlib import Path

np.random.seed(42)

OUTPUT_DIR = Path(__file__).parent.parent / "images"
OUTPUT_DIR.mkdir(exist_ok=True)

CHART_WIDTH = 560
CHART_HEIGHT = 280


def _acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute autocorrelation function up to max_lag."""
    x = x - x.mean()
    n = len(x)
    acf_vals = np.correlate(x, x, mode="full")[n - 1 :]
    acf_vals = acf_vals / acf_vals[0]
    return acf_vals[: max_lag + 1]


def _pacf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute partial autocorrelation via Durbin-Levinson."""
    acf_vals = _acf(x, max_lag)
    pacf_vals = np.zeros(max_lag + 1)
    pacf_vals[0] = 1.0
    if max_lag == 0:
        return pacf_vals
    pacf_vals[1] = acf_vals[1]
    phi = np.zeros((max_lag + 1, max_lag + 1))
    phi[1, 1] = acf_vals[1]
    for k in range(2, max_lag + 1):
        num = acf_vals[k] - sum(phi[k - 1, j] * acf_vals[k - j] for j in range(1, k))
        den = 1.0 - sum(phi[k - 1, j] * acf_vals[j] for j in range(1, k))
        if abs(den) < 1e-10:
            break
        phi[k, k] = num / den
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        pacf_vals[k] = phi[k, k]
    return pacf_vals


def plot_ar_process() -> None:
    """Show an AR(2) process and how each value depends on previous values."""
    np.random.seed(42)
    n = 80
    phi1, phi2 = 0.6, -0.2
    mu = 50.0
    y = np.zeros(n)
    y[0] = mu
    y[1] = mu
    for t in range(2, n):
        y[t] = mu * (1 - phi1 - phi2) + phi1 * y[t - 1] + phi2 * y[t - 2] + np.random.normal(0, 3)

    df = pd.DataFrame({"t": np.arange(n), "y": y})

    # Highlight a specific point and its dependencies
    target_t = 40
    dep_df = pd.DataFrame(
        [
            {"t": target_t - 2, "y": y[target_t - 2], "role": f"y(t-2), weight φ₂={phi2}"},
            {"t": target_t - 1, "y": y[target_t - 1], "role": f"y(t-1), weight φ₁={phi1}"},
            {"t": target_t, "y": y[target_t], "role": "y(t) = predicted value"},
        ]
    )

    line = (
        alt.Chart(df)
        .mark_line(color="grey", strokeWidth=1, opacity=0.5)
        .encode(
            x=alt.X("t:Q", title="Time"),
            y=alt.Y("y:Q", title="y(t)"),
        )
    )

    points = (
        alt.Chart(dep_df)
        .mark_circle(size=150)
        .encode(
            x="t:Q",
            y="y:Q",
            color=alt.Color(
                "role:N",
                scale=alt.Scale(range=["#4c78a8", "#54a24b", "#e45756"]),
                title="Role",
            ),
        )
    )

    arrows = (
        alt.Chart(
            pd.DataFrame(
                [
                    {"x": target_t - 2, "y": y[target_t - 2], "x2": target_t, "y2": y[target_t]},
                    {"x": target_t - 1, "y": y[target_t - 1], "x2": target_t, "y2": y[target_t]},
                ]
            )
        )
        .mark_rule(color="#999", strokeDash=[4, 2], strokeWidth=1)
        .encode(x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q")
    )

    # Equation annotation
    eq_df = pd.DataFrame(
        [{"t": 55, "y": max(y) + 4, "label": "y(t) = c + φ₁·y(t-1) + φ₂·y(t-2) + ε(t)"}]
    )
    eq = (
        alt.Chart(eq_df)
        .mark_text(fontSize=13, fontWeight="bold", align="left")
        .encode(x="t:Q", y="y:Q", text="label:N")
    )

    chart = (line + arrows + points + eq).properties(
        width=CHART_WIDTH, height=CHART_HEIGHT, title="AR(2) Process: Each Value Is a Linear Combination of the Two Previous"
    )
    chart.save(str(OUTPUT_DIR / "sarimax-ar-process.png"), scale_factor=2)
    print("  Saved sarimax-ar-process.png")


def plot_ma_process() -> None:
    """Show an MA(1) process — each value depends on current and previous error."""
    np.random.seed(42)
    n = 80
    theta = 0.7
    mu = 50.0
    errors = np.random.normal(0, 3, n + 1)
    y = np.array([mu + errors[t] + theta * errors[t - 1] for t in range(1, n + 1)])

    df = pd.DataFrame({"t": np.arange(n), "y": y})
    err_df = pd.DataFrame({"t": np.arange(n), "error": errors[1:]})

    y_chart = (
        alt.Chart(df)
        .mark_line(color="#4c78a8", strokeWidth=1.5)
        .encode(
            x=alt.X("t:Q", title="Time"),
            y=alt.Y("y:Q", title="Value"),
        )
    )

    err_chart = (
        alt.Chart(err_df)
        .mark_bar(width=3, opacity=0.5)
        .encode(
            x=alt.X("t:Q", title="Time"),
            y=alt.Y("error:Q", title="Error ε(t)"),
            color=alt.condition(alt.datum.error > 0, alt.value("#54a24b"), alt.value("#e45756")),
        )
    )

    zero = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(color="black", strokeWidth=0.5)
        .encode(y="y:Q")
    )

    top = y_chart.properties(width=CHART_WIDTH, height=180, title="MA(1) Process: y(t) = μ + ε(t) + θ·ε(t-1)")
    bottom = (err_chart + zero).properties(width=CHART_WIDTH, height=140, title="White noise errors ε(t)")

    chart = alt.vconcat(top, bottom).resolve_scale(x="shared")
    chart.save(str(OUTPUT_DIR / "sarimax-ma-process.png"), scale_factor=2)
    print("  Saved sarimax-ma-process.png")


def plot_differencing() -> None:
    """Show the effect of first and seasonal differencing."""
    np.random.seed(42)
    n = 72
    t = np.arange(n)
    trend = 0.4 * t
    seasonal = 15 * np.sin(2 * np.pi * t / 12)
    noise = np.random.normal(0, 3, n)
    y = 50 + trend + seasonal + noise

    y_diff1 = np.diff(y, n=1)
    # Seasonal difference: y(t) - y(t-12)
    y_seasonal = y[12:] - y[:-12]

    panels = []
    for series, label, vals, t_vals in [
        ("Original y(t)", "Original: trend + seasonality", y, t),
        ("Δy(t) = y(t) − y(t−1)", "First difference: removes trend, seasonality remains", y_diff1, t[1:]),
        ("y(t) − y(t−12)", "Seasonal difference (s=12): removes seasonality, trend remains", y_seasonal, t[12:]),
    ]:
        for i, v in enumerate(vals):
            panels.append({"t": int(t_vals[i]), "y": float(v), "series": series, "label": label})

    df = pd.DataFrame(panels)

    chart = (
        alt.Chart(df)
        .mark_line(strokeWidth=1.5)
        .encode(
            x=alt.X("t:Q", title="Month"),
            y=alt.Y("y:Q", title="Value"),
            color=alt.Color("series:N", legend=None),
            row=alt.Row(
                "label:N",
                title=None,
                header=alt.Header(labelFontSize=11, labelAlign="left"),
                sort=[
                    "Original: trend + seasonality",
                    "First difference: removes trend, seasonality remains",
                    "Seasonal difference (s=12): removes seasonality, trend remains",
                ],
            ),
        )
        .properties(width=CHART_WIDTH, height=150)
        .resolve_scale(y="independent")
    )
    chart.save(str(OUTPUT_DIR / "sarimax-differencing.png"), scale_factor=2)
    print("  Saved sarimax-differencing.png")


def plot_seasonal_decomposition() -> None:
    """Decompose a series into trend, seasonal, and residual components."""
    np.random.seed(42)
    n = 72
    t = np.arange(n)
    trend = 50 + 0.3 * t
    seasonal = 15 * np.sin(2 * np.pi * t / 12)
    residual = np.random.normal(0, 3, n)
    y = trend + seasonal + residual

    rows = []
    for i in range(n):
        rows.append({"t": i, "value": y[i], "component": "1. Observed y(t)"})
        rows.append({"t": i, "value": trend[i], "component": "2. Trend"})
        rows.append({"t": i, "value": seasonal[i], "component": "3. Seasonal"})
        rows.append({"t": i, "value": residual[i], "component": "4. Residual"})

    df = pd.DataFrame(rows)

    chart = (
        alt.Chart(df)
        .mark_line(strokeWidth=1.5)
        .encode(
            x=alt.X("t:Q", title="Month"),
            y=alt.Y("value:Q", title="Value"),
            color=alt.Color(
                "component:N",
                scale=alt.Scale(
                    domain=["1. Observed y(t)", "2. Trend", "3. Seasonal", "4. Residual"],
                    range=["#4c78a8", "#e45756", "#54a24b", "#f58518"],
                ),
                legend=None,
            ),
            row=alt.Row(
                "component:N",
                title=None,
                header=alt.Header(labelFontSize=11),
                sort=["1. Observed y(t)", "2. Trend", "3. Seasonal", "4. Residual"],
            ),
        )
        .properties(width=CHART_WIDTH, height=110)
        .resolve_scale(y="independent")
    )
    chart.save(str(OUTPUT_DIR / "sarimax-decomposition.png"), scale_factor=2)
    print("  Saved sarimax-decomposition.png")


def plot_acf_pacf_examples() -> None:
    """Show ACF and PACF patterns for AR, MA, and ARMA processes."""
    np.random.seed(42)
    n = 500
    max_lag = 20
    conf = 1.96 / np.sqrt(n)

    # AR(2) process
    ar2 = np.zeros(n)
    for t in range(2, n):
        ar2[t] = 0.6 * ar2[t - 1] - 0.2 * ar2[t - 2] + np.random.normal(0, 1)

    # MA(2) process
    errors = np.random.normal(0, 1, n + 2)
    ma2 = np.array([errors[t] + 0.7 * errors[t - 1] + 0.3 * errors[t - 2] for t in range(2, n + 2)])

    rows = []
    for label, series in [("AR(2)", ar2), ("MA(2)", ma2)]:
        acf_vals = _acf(series, max_lag)
        pacf_vals = _pacf(series, max_lag)
        for lag in range(1, max_lag + 1):
            rows.append({"lag": lag, "value": acf_vals[lag], "function": "ACF", "process": label})
            rows.append({"lag": lag, "value": pacf_vals[lag], "function": "PACF", "process": label})

    df = pd.DataFrame(rows)

    # Add confidence band boundaries into main df for faceting compatibility
    df["conf_hi"] = conf
    df["conf_lo"] = -conf
    df["zero"] = 0.0

    chart = (
        alt.Chart(df)
        .mark_bar(width=8)
        .encode(
            x=alt.X("lag:Q", title="Lag", scale=alt.Scale(domain=[0, max_lag + 1])),
            y=alt.Y("value:Q", title="Correlation"),
        )
        .properties(width=250, height=150)
        .facet(column=alt.Column("function:N", title=None), row=alt.Row("process:N", title=None))
        .properties(title="ACF and PACF Signatures Help Identify Model Order")
        .resolve_scale(y="independent")
    )
    chart.save(str(OUTPUT_DIR / "sarimax-acf-pacf.png"), scale_factor=2)
    print("  Saved sarimax-acf-pacf.png")


def plot_exogenous_effect() -> None:
    """Show how exogenous variables enter SARIMAX as a linear regression component."""
    np.random.seed(42)
    n = 60
    t = np.arange(n)

    temperature = 22 + 8 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 1, n)
    beta = 2.5
    arima_noise = np.zeros(n)
    for i in range(1, n):
        arima_noise[i] = 0.5 * arima_noise[i - 1] + np.random.normal(0, 3)

    y = 20 + beta * temperature + arima_noise

    rows = []
    for i in range(n):
        rows.append({"t": i, "value": y[i], "component": "Observed cases y(t)", "panel": "1. Full model"})
        rows.append({"t": i, "value": beta * temperature[i], "component": "Exogenous: β·temperature(t)", "panel": "2. Components"})
        rows.append({"t": i, "value": arima_noise[i], "component": "ARIMA errors: η(t)", "panel": "2. Components"})

    df = pd.DataFrame(rows)

    chart = (
        alt.Chart(df)
        .mark_line(strokeWidth=1.5)
        .encode(
            x=alt.X("t:Q", title="Month"),
            y=alt.Y("value:Q", title="Value"),
            color=alt.Color(
                "component:N",
                scale=alt.Scale(
                    domain=["Observed cases y(t)", "Exogenous: β·temperature(t)", "ARIMA errors: η(t)"],
                    range=["#4c78a8", "#e45756", "#54a24b"],
                ),
                title="Component",
            ),
            row=alt.Row(
                "panel:N",
                title=None,
                header=alt.Header(labelFontSize=11),
                sort=["1. Full model", "2. Components"],
            ),
        )
        .properties(width=CHART_WIDTH, height=180)
        .resolve_scale(y="independent")
    )
    chart.save(str(OUTPUT_DIR / "sarimax-exogenous.png"), scale_factor=2)
    print("  Saved sarimax-exogenous.png")


def plot_stationarity_examples() -> None:
    """Show stationary vs non-stationary series."""
    np.random.seed(42)
    n = 100
    t = np.arange(n)

    # Stationary: AR(1) with |φ|<1
    stat = np.zeros(n)
    for i in range(1, n):
        stat[i] = 0.5 * stat[i - 1] + np.random.normal(0, 2)
    stat += 50

    # Non-stationary: random walk
    walk = np.cumsum(np.random.normal(0, 2, n)) + 50

    # Non-stationary: trend
    trending = 50 + 0.5 * t + np.random.normal(0, 2, n)

    rows = []
    for vals, label in [
        (stat, "Stationary: constant mean & variance"),
        (walk, "Non-stationary: random walk (no fixed mean)"),
        (trending, "Non-stationary: deterministic trend"),
    ]:
        for i in range(n):
            rows.append({"t": i, "y": vals[i], "type": label})

    df = pd.DataFrame(rows)

    chart = (
        alt.Chart(df)
        .mark_line(strokeWidth=1.5)
        .encode(
            x=alt.X("t:Q", title="Time"),
            y=alt.Y("y:Q", title="Value"),
            color=alt.Color("type:N", legend=None),
            row=alt.Row(
                "type:N",
                title=None,
                header=alt.Header(labelFontSize=11, labelAlign="left"),
                sort=[
                    "Stationary: constant mean & variance",
                    "Non-stationary: random walk (no fixed mean)",
                    "Non-stationary: deterministic trend",
                ],
            ),
        )
        .properties(width=CHART_WIDTH, height=140)
        .resolve_scale(y="independent")
    )
    chart.save(str(OUTPUT_DIR / "sarimax-stationarity.png"), scale_factor=2)
    print("  Saved sarimax-stationarity.png")


def plot_residual_diagnostics() -> None:
    """Show what good vs bad residual diagnostics look like."""
    np.random.seed(42)
    n = 100

    # Good residuals: white noise
    good = np.random.normal(0, 2, n)

    # Bad residuals: autocorrelated
    bad = np.zeros(n)
    for i in range(1, n):
        bad[i] = 0.7 * bad[i - 1] + np.random.normal(0, 2)

    rows = []
    for vals, label in [
        (good, "Good: residuals look like white noise"),
        (bad, "Bad: residuals are autocorrelated (model is missing structure)"),
    ]:
        for i in range(n):
            rows.append({"t": i, "residual": vals[i], "quality": label})

    df = pd.DataFrame(rows)

    chart = (
        alt.Chart(df)
        .mark_bar(width=3)
        .encode(
            x=alt.X("t:Q", title="Time"),
            y=alt.Y("residual:Q", title="Residual"),
            color=alt.condition(alt.datum.residual > 0, alt.value("#4c78a8"), alt.value("#e45756")),
        )
        .properties(width=CHART_WIDTH, height=150)
        .facet(
            row=alt.Row(
                "quality:N",
                title=None,
                header=alt.Header(labelFontSize=11, labelAlign="left"),
                sort=[
                    "Good: residuals look like white noise",
                    "Bad: residuals are autocorrelated (model is missing structure)",
                ],
            ),
        )
        .properties(title="Residual Diagnostics: What to Look For")
        .resolve_scale(y="independent")
    )
    chart.save(str(OUTPUT_DIR / "sarimax-residual-diagnostics.png"), scale_factor=2)
    print("  Saved sarimax-residual-diagnostics.png")


def plot_prediction_intervals() -> None:
    """Show SARIMAX prediction intervals widening over the forecast horizon."""
    np.random.seed(42)
    n_hist = 48
    n_fore = 18
    t_hist = np.arange(n_hist)
    t_fore = np.arange(n_hist, n_hist + n_fore)

    seasonal = 12 * np.sin(2 * np.pi * t_hist / 12)
    y_hist = 50 + 0.2 * t_hist + seasonal + np.random.normal(0, 3, n_hist)

    seasonal_fore = 12 * np.sin(2 * np.pi * t_fore / 12)
    y_fore_mean = 50 + 0.2 * t_fore + seasonal_fore

    sigma = 3.0
    # Intervals widen with sqrt of horizon
    steps = np.arange(1, n_fore + 1)
    widths = sigma * np.sqrt(steps) * 1.645

    rows_hist = [{"t": int(t_hist[i]), "y": y_hist[i]} for i in range(n_hist)]
    rows_fore = [
        {"t": int(t_fore[i]), "mean": y_fore_mean[i], "lo95": y_fore_mean[i] - widths[i], "hi95": y_fore_mean[i] + widths[i]}
        for i in range(n_fore)
    ]

    hist_df = pd.DataFrame(rows_hist)
    fore_df = pd.DataFrame(rows_fore)

    hist_line = (
        alt.Chart(hist_df)
        .mark_line(color="steelblue", strokeWidth=1.5)
        .encode(x=alt.X("t:Q", title="Month"), y=alt.Y("y:Q", title="Disease cases"))
    )

    fore_band = (
        alt.Chart(fore_df)
        .mark_area(opacity=0.25, color="#e45756")
        .encode(x="t:Q", y="lo95:Q", y2="hi95:Q")
    )
    fore_line = (
        alt.Chart(fore_df)
        .mark_line(color="#e45756", strokeWidth=2, strokeDash=[4, 2])
        .encode(x="t:Q", y="mean:Q")
    )

    rule = (
        alt.Chart(pd.DataFrame({"x": [n_hist - 0.5]}))
        .mark_rule(color="black", strokeDash=[4, 4])
        .encode(x="x:Q")
    )

    label_df = pd.DataFrame(
        [
            {"t": 20, "y": 82, "label": "Observed"},
            {"t": 55, "y": 82, "label": "Forecast ± 90% interval"},
        ]
    )
    labels = alt.Chart(label_df).mark_text(fontSize=12, fontWeight="bold").encode(x="t:Q", y="y:Q", text="label:N")

    chart = (hist_line + fore_band + fore_line + rule + labels).properties(
        width=CHART_WIDTH, height=CHART_HEIGHT, title="SARIMAX Prediction Intervals Widen Over the Forecast Horizon"
    )
    chart.save(str(OUTPUT_DIR / "sarimax-prediction-intervals.png"), scale_factor=2)
    print("  Saved sarimax-prediction-intervals.png")


def plot_full_model_diagram() -> None:
    """Create a visual summary of all SARIMAX components and how they combine."""
    components = [
        {"component": "AR(p)", "description": "Autoregressive: linear dependence on recent values", "order": 1},
        {"component": "I(d)", "description": "Integrated: differencing to remove trend", "order": 2},
        {"component": "MA(q)", "description": "Moving average: linear dependence on recent errors", "order": 3},
        {"component": "SAR(P)ₛ", "description": "Seasonal AR: dependence on values s periods ago", "order": 4},
        {"component": "SI(D)ₛ", "description": "Seasonal differencing: y(t) − y(t−s)", "order": 5},
        {"component": "SMA(Q)ₛ", "description": "Seasonal MA: dependence on errors s periods ago", "order": 6},
        {"component": "X", "description": "Exogenous: linear effect of external covariates", "order": 7},
    ]
    df = pd.DataFrame(components)

    bars = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            y=alt.Y("component:N", title=None, sort=alt.EncodingSortField(field="order"), axis=alt.Axis(labelFontSize=13, labelFontWeight="bold")),
            x=alt.X("order:Q", title=None, axis=None, scale=alt.Scale(domain=[0, 8])),
            color=alt.Color(
                "component:N",
                scale=alt.Scale(
                    domain=["AR(p)", "I(d)", "MA(q)", "SAR(P)ₛ", "SI(D)ₛ", "SMA(Q)ₛ", "X"],
                    range=["#4c78a8", "#72b7b2", "#54a24b", "#e45756", "#f58518", "#eeca3b", "#b279a2"],
                ),
                legend=None,
            ),
        )
    )

    text = (
        alt.Chart(df)
        .mark_text(align="left", dx=5, fontSize=12)
        .encode(
            y=alt.Y("component:N", sort=alt.EncodingSortField(field="order")),
            x=alt.X("order:Q"),
            text="description:N",
        )
    )

    chart = (bars + text).properties(
        width=CHART_WIDTH,
        height=250,
        title="SARIMAX(p,d,q)(P,D,Q)ₛ — The Seven Components",
    )
    chart.save(str(OUTPUT_DIR / "sarimax-components-diagram.png"), scale_factor=2)
    print("  Saved sarimax-components-diagram.png")


if __name__ == "__main__":
    print("Generating SARIMAX guide illustrations...")
    plot_ar_process()
    plot_ma_process()
    plot_differencing()
    plot_seasonal_decomposition()
    plot_acf_pacf_examples()
    plot_exogenous_effect()
    plot_stationarity_examples()
    plot_residual_diagnostics()
    plot_prediction_intervals()
    plot_full_model_diagram()
    print("Done.")
