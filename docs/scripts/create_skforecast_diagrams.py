"""Create visualizations for skforecast documentation."""

import altair as alt
import pandas as pd
import numpy as np

# Set Altair to save as PNG
alt.data_transformers.disable_max_rows()


def create_data_transformation_diagram():
    """Create diagram showing long to wide format transformation."""
    # Sample long format data
    long_data = pd.DataFrame({
        "time_period": ["2024-01", "2024-01", "2024-02", "2024-02", "2024-03", "2024-03"],
        "location": ["loc_A", "loc_B", "loc_A", "loc_B", "loc_A", "loc_B"],
        "disease_cases": [120, 85, 135, 92, 145, 98],
        "row": [0, 1, 2, 3, 4, 5],
    })

    # Create visualization of long format
    long_chart = (
        alt.Chart(long_data)
        .mark_rect(stroke="black", strokeWidth=1)
        .encode(
            x=alt.X("column:N", axis=alt.Axis(title="", labelAngle=0)),
            y=alt.Y("row:O", axis=alt.Axis(title="Row")),
            color=alt.Color("row:O", legend=None, scale=alt.Scale(scheme="blues")),
        )
        .properties(width=400, height=200, title="Long Format (Input)")
    )

    # Add text labels
    long_text = (
        alt.Chart(long_data)
        .mark_text(fontSize=10)
        .encode(
            x="column:N",
            y="row:O",
            text="value:N",
        )
    )

    # Prepare data for text
    long_text_data = []
    for _, row in long_data.iterrows():
        long_text_data.append({"column": "time_period", "row": row["row"], "value": row["time_period"]})
        long_text_data.append({"column": "location", "row": row["row"], "value": row["location"]})
        long_text_data.append({"column": "disease_cases", "row": row["row"], "value": int(row["disease_cases"])})

    long_text = alt.Chart(pd.DataFrame(long_text_data)).mark_text(fontSize=10).encode(x="column:N", y="row:O", text="value:N")

    # Create wide format visualization
    wide_data = pd.DataFrame({
        "time_period": ["2024-01", "2024-02", "2024-03"],
        "loc_A": [120, 135, 145],
        "loc_B": [85, 92, 98],
        "row": [0, 1, 2],
    })

    wide_text_data = []
    for _, row in wide_data.iterrows():
        wide_text_data.append({"column": "time_period", "row": row["row"], "value": row["time_period"]})
        wide_text_data.append({"column": "loc_A", "row": row["row"], "value": int(row["loc_A"])})
        wide_text_data.append({"column": "loc_B", "row": row["row"], "value": int(row["loc_B"])})

    wide_chart = (
        alt.Chart(pd.DataFrame(wide_text_data))
        .mark_rect(stroke="black", strokeWidth=1)
        .encode(
            x=alt.X("column:N", axis=alt.Axis(title="", labelAngle=0)),
            y=alt.Y("row:O", axis=alt.Axis(title="Row")),
            color=alt.Color("row:O", legend=None, scale=alt.Scale(scheme="greens")),
        )
        .properties(width=300, height=120, title="Wide Format (For Skforecast)")
    )

    wide_text = alt.Chart(pd.DataFrame(wide_text_data)).mark_text(fontSize=10).encode(x="column:N", y="row:O", text="value:N")

    # Combine
    final_chart = alt.vconcat(long_chart + long_text, wide_chart + wide_text).configure_view(strokeWidth=0)

    return final_chart


def create_lag_features_diagram():
    """Create diagram showing how lag features are created."""
    # Time series data
    time_series = pd.DataFrame({
        "time": list(range(0, 7)),
        "value": [20, 22, 21, 23, 25, 24, 26],
    })

    # Original series chart
    series_chart = (
        alt.Chart(time_series)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("time:Q", title="Time", scale=alt.Scale(domain=[-0.5, 6.5])),
            y=alt.Y("value:Q", title="Temperature", scale=alt.Scale(domain=[18, 28])),
        )
        .properties(width=500, height=200, title="Original Time Series")
    )

    # Add value labels
    series_text = (
        alt.Chart(time_series)
        .mark_text(dy=-10, fontSize=12, fontWeight="bold")
        .encode(x="time:Q", y="value:Q", text="value:Q")
    )

    # Lag features table
    lag_data = pd.DataFrame({
        "t": [3, 4, 5, 6],
        "lag_3": [20, 22, 21, 23],
        "lag_2": [22, 21, 23, 25],
        "lag_1": [21, 23, 25, 24],
        "target": [23, 25, 24, 26],
    })

    # Create table visualization
    lag_long = []
    columns = ["t", "lag_3", "lag_2", "lag_1", "target"]
    for idx, row in lag_data.iterrows():
        for col_idx, col in enumerate(columns):
            lag_long.append({"row": idx, "column": col_idx, "col_name": col, "value": int(row[col])})

    lag_df = pd.DataFrame(lag_long)

    table_chart = (
        alt.Chart(lag_df)
        .mark_rect(stroke="black", strokeWidth=1)
        .encode(
            x=alt.X("column:O", axis=alt.Axis(title="", labels=False)),
            y=alt.Y("row:O", axis=alt.Axis(title="Row", tickMinStep=1)),
            color=alt.condition(
                alt.datum.col_name == "target",
                alt.value("#90EE90"),  # Light green for target
                alt.value("#ADD8E6"),  # Light blue for features
            ),
        )
        .properties(width=400, height=150, title="Lagged Features (Training Data)")
    )

    # Add column headers
    headers = pd.DataFrame([{"column": i, "col_name": col, "row": -1} for i, col in enumerate(columns)])

    header_chart = alt.Chart(headers).mark_text(fontSize=12, fontWeight="bold", dy=0).encode(x="column:O", text="col_name:N", y=alt.value(10))

    # Add values
    table_text = alt.Chart(lag_df).mark_text(fontSize=11).encode(x="column:O", y="row:O", text="value:Q")

    # Combine
    combined = alt.vconcat(series_chart + series_text, table_chart + table_text + header_chart).configure_view(strokeWidth=0)

    return combined


def create_bootstrap_samples_diagram():
    """Create diagram showing bootstrap sampling uncertainty."""
    np.random.seed(42)

    # Generate deterministic prediction
    time_points = np.arange(0, 10)
    deterministic = 20 + 0.5 * time_points

    # Generate bootstrap samples
    n_samples = 50
    samples_data = []

    for sample_id in range(n_samples):
        noise = np.random.normal(0, 0.3, size=10).cumsum()  # Cumulative to show spreading
        trajectory = deterministic + noise
        for t, val in zip(time_points, trajectory):
            samples_data.append({"time": t, "value": val, "sample_id": sample_id, "type": "sample"})

    # Add deterministic line
    for t, val in zip(time_points, deterministic):
        samples_data.append({"time": t, "value": val, "sample_id": -1, "type": "mean"})

    samples_df = pd.DataFrame(samples_data)

    # Plot samples
    sample_lines = (
        alt.Chart(samples_df[samples_df["type"] == "sample"])
        .mark_line(opacity=0.1, strokeWidth=1)
        .encode(
            x=alt.X("time:Q", title="Time Steps", scale=alt.Scale(domain=[0, 9])),
            y=alt.Y("value:Q", title="Predicted Cases", scale=alt.Scale(domain=[18, 28])),
            detail="sample_id:N",
            color=alt.value("steelblue"),
        )
    )

    # Plot mean
    mean_line = (
        alt.Chart(samples_df[samples_df["type"] == "mean"])
        .mark_line(strokeWidth=3, color="red")
        .encode(x="time:Q", y="value:Q")
    )

    # Add shaded region for uncertainty
    quantiles = samples_df[samples_df["type"] == "sample"].groupby("time")["value"].quantile([0.1, 0.5, 0.9]).reset_index()
    quantiles_wide = quantiles.pivot(index="time", columns="level_1", values="value").reset_index()
    quantiles_wide.columns = ["time", "q10", "q50", "q90"]

    area_chart = (
        alt.Chart(quantiles_wide)
        .mark_area(opacity=0.3, color="steelblue")
        .encode(x="time:Q", y="q10:Q", y2="q90:Q")
    )

    # Combine
    combined = (
        (area_chart + sample_lines + mean_line)
        .properties(width=600, height=300, title="Bootstrap Samples: Uncertainty Propagation (50 trajectories)")
        .configure_view(strokeWidth=0)
    )

    return combined


def create_recursive_prediction_diagram():
    """Create diagram showing recursive prediction steps."""
    # Data for each prediction step
    steps_data = []

    # Step 0: History
    history = [18, 19, 20, 21, 22, 21, 20, 22, 23, 21, 20, 22]
    for i, val in enumerate(history):
        steps_data.append({"time": i - 12, "value": val, "type": "history", "step": "history"})

    # Step 1: Predict using last 12
    pred_1 = 24
    steps_data.append({"time": 0, "value": pred_1, "type": "prediction", "step": "step_1"})

    # Step 2: Predict using last 11 history + pred_1
    pred_2 = 25
    steps_data.append({"time": 1, "value": pred_2, "type": "prediction", "step": "step_2"})

    # Step 3: Predict using last 10 history + pred_1 + pred_2
    pred_3 = 26
    steps_data.append({"time": 2, "value": pred_3, "type": "prediction", "step": "step_3"})

    df = pd.DataFrame(steps_data)

    # Base chart
    base = alt.Chart(df).encode(x=alt.X("time:Q", title="Time (relative)", scale=alt.Scale(domain=[-12, 3])))

    # History line
    history_line = base.transform_filter(alt.datum.type == "history").mark_line(strokeWidth=2, color="gray").encode(y=alt.Y("value:Q", title="Value"))

    history_points = base.transform_filter(alt.datum.type == "history").mark_circle(size=50, color="gray").encode(y="value:Q")

    # Predictions
    pred_points = (
        base.transform_filter(alt.datum.type == "prediction")
        .mark_circle(size=100)
        .encode(
            y="value:Q",
            color=alt.Color(
                "step:N",
                scale=alt.Scale(domain=["step_1", "step_2", "step_3"], range=["#FF6B6B", "#4ECDC4", "#45B7D1"]),
                legend=alt.Legend(title="Prediction Step"),
            ),
        )
    )

    # Add vertical line at t=0
    rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[5, 5], strokeWidth=2).encode(x="x:Q")

    # Combine
    combined = (
        (history_line + history_points + pred_points + rule)
        .properties(width=600, height=250, title="Recursive Prediction: Each Step Uses Previous Predictions")
        .configure_view(strokeWidth=0)
    )

    return combined


if __name__ == "__main__":
    # Create output directory
    import os
    import sys

    # Get absolute path to images directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "images")
    output_dir = os.path.abspath(output_dir)

    print(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Generate diagrams with error handling
    try:
        print("Creating data transformation diagram...")
        chart1 = create_data_transformation_diagram()
        output_path = os.path.join(output_dir, "skforecast-data-transformation.png")
        chart1.save(output_path, scale_factor=2.0)
        print(f"  Saved to: {output_path}")

        print("Creating lag features diagram...")
        chart2 = create_lag_features_diagram()
        output_path = os.path.join(output_dir, "skforecast-lag-features.png")
        chart2.save(output_path, scale_factor=2.0)
        print(f"  Saved to: {output_path}")

        print("Creating bootstrap samples diagram...")
        chart3 = create_bootstrap_samples_diagram()
        output_path = os.path.join(output_dir, "skforecast-bootstrap-samples.png")
        chart3.save(output_path, scale_factor=2.0)
        print(f"  Saved to: {output_path}")

        print("Creating recursive prediction diagram...")
        chart4 = create_recursive_prediction_diagram()
        output_path = os.path.join(output_dir, "skforecast-recursive-prediction.png")
        chart4.save(output_path, scale_factor=2.0)
        print(f"  Saved to: {output_path}")

        print("\nAll diagrams created successfully!")
    except Exception as e:
        print(f"\nError creating diagrams: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
