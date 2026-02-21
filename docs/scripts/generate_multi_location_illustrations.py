"""Generate illustrations for multi-location time series modeling approaches."""

import numpy as np
import pandas as pd
import altair as alt
from pathlib import Path

np.random.seed(42)

OUTPUT_DIR = Path(__file__).parent.parent / "images"
OUTPUT_DIR.mkdir(exist_ok=True)

CHART_WIDTH = 560
CHART_HEIGHT = 280

# Province names for examples
PROVINCES = ["Province A", "Province B", "Province C", "Province D"]
COLORS = ["#4c78a8", "#e45756", "#54a24b", "#f58518"]
N_MONTHS = 60


def generate_province_data() -> pd.DataFrame:
    """Generate synthetic disease case data for multiple provinces.

    Provinces share a common seasonal pattern but differ in level, trend, and noise.
    """
    t = np.arange(N_MONTHS)
    rows = []

    configs = [
        {"level": 80, "trend": 0.3, "seasonal_amp": 20, "noise_std": 4, "phase": 0},
        {"level": 50, "trend": 0.1, "seasonal_amp": 12, "noise_std": 3, "phase": 0.3},
        {"level": 120, "trend": -0.1, "seasonal_amp": 25, "noise_std": 5, "phase": -0.2},
        {"level": 30, "trend": 0.05, "seasonal_amp": 8, "noise_std": 2, "phase": 0.1},
    ]

    for province, cfg in zip(PROVINCES, configs):
        seasonal = cfg["seasonal_amp"] * np.sin(2 * np.pi * t / 12 + cfg["phase"])
        trend = cfg["trend"] * t
        noise = np.random.normal(0, cfg["noise_std"], N_MONTHS)
        y = cfg["level"] + trend + seasonal + noise
        y = np.maximum(y, 0)
        for i in range(N_MONTHS):
            rows.append({"month": i, "province": province, "cases": y[i]})

    return pd.DataFrame(rows)


def plot_multi_province_overview() -> None:
    """Show the raw time series for all provinces — similar patterns, different scales."""
    df = generate_province_data()

    chart = (
        alt.Chart(df)
        .mark_line(strokeWidth=1.5)
        .encode(
            x=alt.X("month:Q", title="Month"),
            y=alt.Y("cases:Q", title="Disease cases"),
            color=alt.Color(
                "province:N",
                scale=alt.Scale(domain=PROVINCES, range=COLORS),
                title="Province",
            ),
            row=alt.Row("province:N", title=None, header=alt.Header(labelFontSize=12)),
        )
        .properties(width=CHART_WIDTH, height=130)
        .resolve_scale(y="independent")
    )

    chart.save(str(OUTPUT_DIR / "multi-location-overview.png"), scale_factor=2)
    print("  Saved multi-location-overview.png")


def plot_per_series_approach() -> None:
    """Illustrate the per-series approach: one model per province, isolated."""
    df = generate_province_data()

    # Show each province with its own "model box"
    # Use facet with annotation-like titles
    chart = (
        alt.Chart(df)
        .mark_line(strokeWidth=1.5)
        .encode(
            x=alt.X("month:Q", title="Month"),
            y=alt.Y("cases:Q", title="Cases"),
            color=alt.Color(
                "province:N",
                scale=alt.Scale(domain=PROVINCES, range=COLORS),
                legend=None,
            ),
        )
        .properties(width=CHART_WIDTH // 2 - 20, height=140)
        .facet(
            facet=alt.Facet("province:N", title=None, header=alt.Header(labelFontSize=11)),
            columns=2,
        )
        .resolve_scale(y="independent")
        .properties(title="Per-Series: One ARIMA Model Per Province")
    )

    chart.save(str(OUTPUT_DIR / "multi-location-per-series.png"), scale_factor=2)
    print("  Saved multi-location-per-series.png")


def plot_pooled_approach() -> None:
    """Illustrate the pooled / global approach: all data stacked into one model."""
    df = generate_province_data()

    line = (
        alt.Chart(df)
        .mark_line(strokeWidth=1.2, opacity=0.7)
        .encode(
            x=alt.X("month:Q", title="Month"),
            y=alt.Y("cases:Q", title="Disease cases"),
            color=alt.Color(
                "province:N",
                scale=alt.Scale(domain=PROVINCES, range=COLORS),
                title="Province",
            ),
        )
    )

    # Add a big bracket-like annotation
    label_df = pd.DataFrame(
        [{"month": 30, "cases": 165, "label": "Single model trained on all provinces"}]
    )
    label = (
        alt.Chart(label_df)
        .mark_text(fontSize=13, fontWeight="bold", color="black")
        .encode(x="month:Q", y="cases:Q", text="label:N")
    )

    chart = (line + label).properties(
        width=CHART_WIDTH,
        height=CHART_HEIGHT,
        title="Global Pooled: One Model Trained on All Locations",
    )
    chart.save(str(OUTPUT_DIR / "multi-location-pooled.png"), scale_factor=2)
    print("  Saved multi-location-pooled.png")


def plot_bias_variance_tradeoff() -> None:
    """Show the bias-variance tradeoff across approaches with a schematic."""
    approaches = [
        {"approach": "Per-series ARIMA", "bias": 0.2, "variance": 0.9, "x_pos": 0},
        {"approach": "Clustered ARIMA", "bias": 0.35, "variance": 0.5, "x_pos": 1},
        {"approach": "Global pooled", "bias": 0.7, "variance": 0.15, "x_pos": 2},
        {"approach": "Hierarchical", "bias": 0.3, "variance": 0.35, "x_pos": 3},
    ]
    df = pd.DataFrame(approaches)

    # Melt for grouped bar
    melted = df.melt(
        id_vars=["approach", "x_pos"],
        value_vars=["bias", "variance"],
        var_name="component",
        value_name="value",
    )

    chart = (
        alt.Chart(melted)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X(
                "approach:N",
                title=None,
                sort=["Per-series ARIMA", "Clustered ARIMA", "Global pooled", "Hierarchical"],
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("value:Q", title="Relative magnitude (schematic)", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "component:N",
                scale=alt.Scale(domain=["bias", "variance"], range=["#e45756", "#4c78a8"]),
                title="Component",
            ),
            xOffset="component:N",
        )
        .properties(
            width=CHART_WIDTH,
            height=CHART_HEIGHT,
            title="Bias–Variance Tradeoff Across Multi-Location Strategies (Schematic)",
        )
    )
    chart.save(str(OUTPUT_DIR / "multi-location-bias-variance.png"), scale_factor=2)
    print("  Saved multi-location-bias-variance.png")


def plot_sample_size_illustration() -> None:
    """Show effective training sample size per approach."""
    data = [
        {"approach": "Per-series\nARIMA", "n_samples": N_MONTHS, "note": f"{N_MONTHS} obs"},
        {"approach": "Clustered\n(2 provinces)", "n_samples": N_MONTHS * 2, "note": f"{N_MONTHS * 2} obs"},
        {"approach": "Global pooled\n(4 provinces)", "n_samples": N_MONTHS * 4, "note": f"{N_MONTHS * 4} obs"},
    ]
    df = pd.DataFrame(data)

    bars = (
        alt.Chart(df)
        .mark_bar(
            cornerRadiusTopLeft=3,
            cornerRadiusTopRight=3,
            color="#4c78a8",
        )
        .encode(
            x=alt.X(
                "approach:N",
                title=None,
                sort=["Per-series\nARIMA", "Clustered\n(2 provinces)", "Global pooled\n(4 provinces)"],
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("n_samples:Q", title="Training observations"),
        )
    )

    text = (
        alt.Chart(df)
        .mark_text(dy=-10, fontSize=12, fontWeight="bold")
        .encode(
            x=alt.X(
                "approach:N",
                sort=["Per-series\nARIMA", "Clustered\n(2 provinces)", "Global pooled\n(4 provinces)"],
            ),
            y="n_samples:Q",
            text="note:N",
        )
    )

    chart = (bars + text).properties(
        width=CHART_WIDTH,
        height=CHART_HEIGHT,
        title="Effective Training Set Size by Approach",
    )
    chart.save(str(OUTPUT_DIR / "multi-location-sample-size.png"), scale_factor=2)
    print("  Saved multi-location-sample-size.png")


def plot_heterogeneity_problem() -> None:
    """Show what goes wrong when you pool very different series naively."""
    np.random.seed(42)
    t = np.arange(36)
    rows = []

    # Province A: high level, strong seasonality
    y_a = 120 + 30 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 4, 36)
    # Province B: low level, weak seasonality
    y_b = 20 + 5 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 2, 36)
    # Pooled model would predict something in between
    y_pooled_a = np.full(12, np.mean(np.concatenate([y_a, y_b])))
    y_pooled_b = y_pooled_a.copy()

    for i in range(36):
        rows.append({"month": i, "province": "Province A (high)", "cases": y_a[i], "type": "Observed"})
        rows.append({"month": i, "province": "Province B (low)", "cases": y_b[i], "type": "Observed"})

    for i in range(12):
        rows.append(
            {"month": 36 + i, "province": "Province A (high)", "cases": y_pooled_a[i], "type": "Pooled prediction"}
        )
        rows.append(
            {"month": 36 + i, "province": "Province B (low)", "cases": y_pooled_b[i], "type": "Pooled prediction"}
        )

    df = pd.DataFrame(rows)

    chart = (
        alt.Chart(df)
        .mark_line(strokeWidth=1.5)
        .encode(
            x=alt.X("month:Q", title="Month"),
            y=alt.Y("cases:Q", title="Cases"),
            color=alt.Color(
                "province:N",
                scale=alt.Scale(
                    domain=["Province A (high)", "Province B (low)"],
                    range=["#4c78a8", "#e45756"],
                ),
            ),
            strokeDash=alt.StrokeDash(
                "type:N",
                scale=alt.Scale(
                    domain=["Observed", "Pooled prediction"],
                    range=[[1, 0], [6, 3]],
                ),
                title="Type",
            ),
        )
        .properties(
            width=CHART_WIDTH,
            height=CHART_HEIGHT,
            title="The Heterogeneity Problem: Naive Pooling Averages Away Differences",
        )
    )

    # Vertical divider
    rule = (
        alt.Chart(pd.DataFrame({"x": [35.5]}))
        .mark_rule(color="black", strokeDash=[4, 4])
        .encode(x="x:Q")
    )

    full = (chart + rule)
    full.save(str(OUTPUT_DIR / "multi-location-heterogeneity.png"), scale_factor=2)
    print("  Saved multi-location-heterogeneity.png")


def plot_clustering_approach() -> None:
    """Show how clustering groups similar provinces before fitting."""
    df = generate_province_data()

    # Assign clusters: A+C are "high case" cluster, B+D are "low case" cluster
    cluster_map = {
        "Province A": "Cluster 1 (medium-high)",
        "Province B": "Cluster 2 (low-medium)",
        "Province C": "Cluster 1 (medium-high)",
        "Province D": "Cluster 2 (low-medium)",
    }
    df["cluster"] = df["province"].map(cluster_map)

    chart = (
        alt.Chart(df)
        .mark_line(strokeWidth=1.2, opacity=0.8)
        .encode(
            x=alt.X("month:Q", title="Month"),
            y=alt.Y("cases:Q", title="Cases"),
            color=alt.Color(
                "province:N",
                scale=alt.Scale(domain=PROVINCES, range=COLORS),
                title="Province",
            ),
            row=alt.Row(
                "cluster:N",
                title=None,
                header=alt.Header(labelFontSize=12, labelFontWeight="bold"),
            ),
        )
        .properties(
            width=CHART_WIDTH,
            height=180,
            title="Clustered Approach: Group Similar Provinces, Fit One Model Per Cluster",
        )
        .resolve_scale(y="independent")
    )

    chart.save(str(OUTPUT_DIR / "multi-location-clustered.png"), scale_factor=2)
    print("  Saved multi-location-clustered.png")


def plot_hierarchical_illustration() -> None:
    """Illustrate hierarchical / mixed-effects concept with shared + local components."""
    np.random.seed(42)
    t = np.arange(36)

    # Shared component (what a hierarchical model learns as "global")
    shared = 10 * np.sin(2 * np.pi * t / 12)

    rows = []
    offsets = [80, 50, 120, 30]
    local_amps = [5, 3, 8, 2]
    for prov, offset, amp in zip(PROVINCES, offsets, local_amps):
        local = amp * np.sin(2 * np.pi * t / 12 + np.random.uniform(-0.5, 0.5))
        noise = np.random.normal(0, 3, 36)
        y = offset + shared + local + noise
        for i in range(36):
            rows.append({"month": i, "province": prov, "cases": y[i]})

    df = pd.DataFrame(rows)

    # Also show the shared component
    shared_df = pd.DataFrame({"month": t, "cases": 70 + shared, "province": "Shared pattern"})

    prov_chart = (
        alt.Chart(df)
        .mark_line(strokeWidth=1.2, opacity=0.6)
        .encode(
            x=alt.X("month:Q", title="Month"),
            y=alt.Y("cases:Q", title="Cases", scale=alt.Scale(domain=[0, 160])),
            color=alt.Color(
                "province:N",
                scale=alt.Scale(domain=PROVINCES, range=COLORS),
                title="Province",
            ),
        )
    )

    shared_chart = (
        alt.Chart(shared_df)
        .mark_line(strokeWidth=3, color="black", strokeDash=[1, 0])
        .encode(x="month:Q", y="cases:Q")
    )

    shared_label = (
        alt.Chart(pd.DataFrame([{"month": 18, "cases": 155, "label": "Shared seasonal pattern (global)"}]))
        .mark_text(fontSize=12, fontWeight="bold")
        .encode(x="month:Q", y="cases:Q", text="label:N")
    )

    local_label = (
        alt.Chart(pd.DataFrame([{"month": 18, "cases": 145, "label": "Colored = province-specific deviations"}]))
        .mark_text(fontSize=11, fontStyle="italic", color="grey")
        .encode(x="month:Q", y="cases:Q", text="label:N")
    )

    chart = (prov_chart + shared_chart + shared_label + local_label).properties(
        width=CHART_WIDTH,
        height=CHART_HEIGHT + 40,
        title="Hierarchical: Shared Global Structure + Province-Specific Deviations",
    )
    chart.save(str(OUTPUT_DIR / "multi-location-hierarchical.png"), scale_factor=2)
    print("  Saved multi-location-hierarchical.png")


if __name__ == "__main__":
    print("Generating multi-location modeling illustrations...")
    plot_multi_province_overview()
    plot_per_series_approach()
    plot_pooled_approach()
    plot_bias_variance_tradeoff()
    plot_sample_size_illustration()
    plot_heterogeneity_problem()
    plot_clustering_approach()
    plot_hierarchical_illustration()
    print("Done.")
