"""Generate illustrations for general bootstrapping documentation using Altair."""

import numpy as np
import pandas as pd
import altair as alt
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "images"
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_classical_bootstrap_process() -> None:
    """Generate illustration showing classical bootstrap resampling process."""
    # Original sample
    original = ["A", "B", "C", "D", "E", "F", "G", "H"]

    # Three bootstrap samples (pre-computed for reproducibility)
    samples = [
        ["A", "C", "C", "B", "H", "A", "E", "G"],
        ["D", "D", "F", "A", "B", "C", "E", "H"],
        ["B", "A", "E", "E", "C", "F", "F", "A"]
    ]

    # Create visualization data
    data_list = []
    for i, sample in enumerate(samples, 1):
        for j, value in enumerate(sample, 1):
            data_list.append({
                "Sample": f"Bootstrap {i}",
                "Position": j,
                "Value": value
            })

    # Add original
    for j, value in enumerate(original, 1):
        data_list.append({
            "Sample": "Original",
            "Position": j,
            "Value": value
        })

    df = pd.DataFrame(data_list)

    # Create chart
    chart = alt.Chart(df).mark_text(fontSize=16, fontWeight="bold").encode(
        x=alt.X("Position:O", title="Position", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Sample:N", title="", sort=["Original", "Bootstrap 1", "Bootstrap 2", "Bootstrap 3"]),
        text="Value:N",
        color=alt.condition(
            alt.datum.Sample == "Original",
            alt.value("steelblue"),
            alt.value("orange")
        )
    ).properties(
        width=600,
        height=200,
        title="Classical Bootstrap: Resampling with Replacement"
    )

    chart.save(str(OUTPUT_DIR / "classical_bootstrap_process.png"))
    print("Generated: classical_bootstrap_process.png")


def generate_iid_vs_timeseries() -> None:
    """Generate comparison of i.i.d. data vs time series data."""
    np.random.seed(42)

    # i.i.d. data (random)
    t_iid = np.arange(1, 21)
    y_iid = np.random.normal(10, 2, 20)

    # Time series data (with autocorrelation)
    t_ts = np.arange(1, 21)
    y_ts = [10]
    for i in range(1, 20):
        y_ts.append(0.7 * y_ts[-1] + np.random.normal(3, 1))
    y_ts = np.array(y_ts)

    df = pd.DataFrame({
        "Time": np.concatenate([t_iid, t_ts]),
        "Value": np.concatenate([y_iid, y_ts]),
        "Type": ["i.i.d. Data (Independent)"] * 20 + ["Time Series (Autocorrelated)"] * 20
    })

    # Create chart
    chart = alt.Chart(df).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("Time:Q", title="Time", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("Value:Q", title="Value"),
        color=alt.Color("Type:N", scale=alt.Scale(domain=["i.i.d. Data (Independent)",
                                                            "Time Series (Autocorrelated)"],
                                                    range=["steelblue", "orange"])),
        row=alt.Row("Type:N", title=None)
    ).properties(
        width=600,
        height=150
    )

    chart.save(str(OUTPUT_DIR / "iid_vs_timeseries.png"))
    print("Generated: iid_vs_timeseries.png")


def generate_broken_temporal_structure() -> None:
    """Generate illustration showing how classical bootstrap breaks temporal structure."""
    # Original time series with trend
    t_original = np.arange(1, 9)
    values_original = np.array([5, 6, 7, 4, 5, 8, 9, 8])

    # Classical bootstrap sample (random order)
    t_bootstrap = np.arange(1, 9)
    values_bootstrap = np.array([9, 6, 6, 5, 8, 5, 7, 8])  # Time order destroyed

    df = pd.DataFrame({
        "Time": np.concatenate([t_original, t_bootstrap]),
        "Value": np.concatenate([values_original, values_bootstrap]),
        "Type": ["Original (Temporal Structure Intact)"] * 8 +
                ["Classical Bootstrap (Structure Destroyed)"] * 8
    })

    # Create chart
    chart = alt.Chart(df).mark_line(point=alt.OverlayMarkDef(size=100), strokeWidth=2).encode(
        x=alt.X("Time:Q", title="Time", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("Value:Q", title="Value", scale=alt.Scale(domain=[3, 10])),
        color=alt.Color("Type:N", scale=alt.Scale(
            domain=["Original (Temporal Structure Intact)", "Classical Bootstrap (Structure Destroyed)"],
            range=["steelblue", "red"]
        )),
        row=alt.Row("Type:N", title=None)
    ).properties(
        width=600,
        height=150
    )

    chart.save(str(OUTPUT_DIR / "broken_temporal_structure.png"))
    print("Generated: broken_temporal_structure.png")


def generate_moving_block_bootstrap() -> None:
    """Generate illustration of moving block bootstrap."""
    # Original time series
    t = np.arange(1, 13)
    values = np.array([2.1, 2.3, 2.5, 2.4, 2.6, 2.8, 2.7, 2.9, 3.1, 3.0, 3.2, 3.4])

    df = pd.DataFrame({"Time": t, "Value": values})

    # Create chart with blocks highlighted
    line = alt.Chart(df).mark_line(point=True, strokeWidth=2, color="steelblue").encode(
        x=alt.X("Time:Q", title="Time", scale=alt.Scale(domain=[0, 13]), axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("Value:Q", title="Value", scale=alt.Scale(domain=[1.5, 3.5]))
    )

    # Highlight blocks
    blocks = []
    block_positions = [(3, 6), (7, 10), (2, 5)]  # Example blocks
    colors = ["red", "green", "purple"]

    for i, (start, end) in enumerate(block_positions):
        block_df = pd.DataFrame({
            "Time": [start, end],
            "y": [1.7, 1.7],
            "label": [f"Block {i+1}", ""]
        })
        blocks.append(
            alt.Chart(block_df).mark_line(strokeWidth=8, opacity=0.3, color=colors[i]).encode(
                x="Time:Q",
                y=alt.Y("y:Q")
            )
        )

    chart = line
    for block in blocks:
        chart = chart + block

    chart = chart.properties(
        width=600,
        height=300,
        title="Moving Block Bootstrap: Preserving Temporal Structure (Block Length = 4)"
    )

    chart.save(str(OUTPUT_DIR / "moving_block_bootstrap.png"))
    print("Generated: moving_block_bootstrap.png")


def generate_block_length_tradeoff() -> None:
    """Generate illustration showing block length trade-off."""
    block_lengths = ["ℓ=1\n(Classical)", "ℓ=n/4\n(Small)", "ℓ=n/2\n(Medium)", "ℓ=n\n(Full)"]
    bias = [1.0, 0.6, 0.3, 0.1]
    variance = [0.2, 0.4, 0.5, 0.9]
    diversity = [1.0, 0.8, 0.5, 0.1]

    df = pd.DataFrame({
        "Block Length": block_lengths * 3,
        "Metric": ["Bias"] * 4 + ["Variance"] * 4 + ["Diversity"] * 4,
        "Value": bias + variance + diversity
    })

    # Create chart
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("Block Length:N", title="Block Length", sort=block_lengths),
        y=alt.Y("Value:Q", title="Level (0-1)"),
        color=alt.Color("Metric:N", scale=alt.Scale(
            domain=["Bias", "Variance", "Diversity"],
            range=["red", "orange", "steelblue"]
        )),
        column=alt.Column("Metric:N", title=None)
    ).properties(
        width=180,
        height=300
    )

    chart.save(str(OUTPUT_DIR / "block_length_tradeoff.png"))
    print("Generated: block_length_tradeoff.png")


def generate_seasonal_block_bootstrap() -> None:
    """Generate illustration of seasonal block bootstrap."""
    # Create 3 years of monthly data
    months = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"] * 3
    t = np.arange(1, 37)
    seasonal_pattern = np.tile([5, 5.5, 7, 9, 12, 15, 18, 17, 14, 10, 7, 5.5], 3)
    noise = np.random.normal(0, 0.5, 36)
    values = seasonal_pattern + noise

    df = pd.DataFrame({
        "Time": t,
        "Value": values,
        "Year": ["Year 1"] * 12 + ["Year 2"] * 12 + ["Year 3"] * 12
    })

    # Create chart
    chart = alt.Chart(df).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("Time:Q", title="Month", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("Value:Q", title="Value (e.g., Temperature)", scale=alt.Scale(domain=[0, 20])),
        color=alt.Color("Year:N", scale=alt.Scale(
            domain=["Year 1", "Year 2", "Year 3"],
            range=["steelblue", "orange", "green"]
        ))
    ).properties(
        width=600,
        height=300,
        title="Seasonal Data: Use Block Length = 12 (Annual Period)"
    )

    chart.save(str(OUTPUT_DIR / "seasonal_block_bootstrap.png"))
    print("Generated: seasonal_block_bootstrap.png")


def generate_acf_for_block_selection() -> None:
    """Generate ACF plot showing how to select block length."""
    lags = np.arange(0, 13)
    acf = np.array([1.0, 0.85, 0.72, 0.61, 0.52, 0.38, 0.28, 0.20, 0.15, 0.12, 0.08, 0.05, 0.03])
    threshold = 0.4

    df = pd.DataFrame({"Lag": lags, "ACF": acf})

    # Create bars
    bars = alt.Chart(df).mark_bar(width=25).encode(
        x=alt.X("Lag:Q", title="Lag", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("ACF:Q", title="Autocorrelation Function (ACF)", scale=alt.Scale(domain=[-0.1, 1.1])),
        color=alt.condition(
            alt.datum.ACF > threshold,
            alt.value("red"),
            alt.value("steelblue")
        )
    )

    # Significance threshold
    threshold_line = alt.Chart(pd.DataFrame({"y": [threshold]})).mark_rule(
        color="red",
        strokeDash=[5, 5],
        strokeWidth=2
    ).encode(y="y:Q")

    # Add text annotation
    annotation = alt.Chart(pd.DataFrame({
        "x": [10],
        "y": [0.5],
        "text": ["Significant up to lag 5\n→ Choose block length ℓ = 8-10"]
    })).mark_text(
        align="left",
        fontSize=12,
        fontWeight="bold",
        color="darkred"
    ).encode(
        x="x:Q",
        y="y:Q",
        text="text:N"
    )

    chart = (bars + threshold_line + annotation).properties(
        width=600,
        height=300,
        title="Using ACF to Select Block Length"
    )

    chart.save(str(OUTPUT_DIR / "acf_block_selection.png"))
    print("Generated: acf_block_selection.png")


def generate_bootstrap_ci_distribution() -> None:
    """Generate illustration of bootstrap distribution and confidence intervals."""
    np.random.seed(42)

    # Simulate bootstrap estimates
    bootstrap_estimates = np.random.normal(2.5, 0.3, 5000)
    true_value = 2.5

    df = pd.DataFrame({"Estimate": bootstrap_estimates})

    # Calculate confidence interval
    ci_lower = np.percentile(bootstrap_estimates, 2.5)
    ci_upper = np.percentile(bootstrap_estimates, 97.5)

    # Create histogram
    hist = alt.Chart(df).mark_bar(opacity=0.7, color="steelblue").encode(
        x=alt.X("Estimate:Q", bin=alt.Bin(maxbins=50), title="Parameter Estimate"),
        y=alt.Y("count():Q", title="Frequency")
    )

    # Add true value line
    true_line = alt.Chart(pd.DataFrame({"x": [true_value]})).mark_rule(
        color="red",
        strokeWidth=3
    ).encode(x="x:Q")

    # Add CI lines
    ci_lower_line = alt.Chart(pd.DataFrame({"x": [ci_lower]})).mark_rule(
        color="orange",
        strokeWidth=2,
        strokeDash=[5, 5]
    ).encode(x="x:Q")

    ci_upper_line = alt.Chart(pd.DataFrame({"x": [ci_upper]})).mark_rule(
        color="orange",
        strokeWidth=2,
        strokeDash=[5, 5]
    ).encode(x="x:Q")

    chart = (hist + true_line + ci_lower_line + ci_upper_line).properties(
        width=600,
        height=300,
        title="Bootstrap Distribution with 95% Confidence Interval"
    )

    chart.save(str(OUTPUT_DIR / "bootstrap_ci_distribution.png"))
    print("Generated: bootstrap_ci_distribution.png")


def main() -> None:
    """Generate all illustrations for general bootstrapping documentation."""
    print("Generating illustrations for bootstrapping documentation...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    generate_classical_bootstrap_process()
    generate_iid_vs_timeseries()
    generate_broken_temporal_structure()
    generate_moving_block_bootstrap()
    generate_block_length_tradeoff()
    generate_seasonal_block_bootstrap()
    generate_acf_for_block_selection()
    generate_bootstrap_ci_distribution()

    print(f"\nAll illustrations generated successfully in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
