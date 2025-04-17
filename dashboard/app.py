import os
import dash
import pandas as pd
import numpy as np
from scipy.special import softmax
import plotly.graph_objects as go
from dash import dcc, html, Input, Output
from figures import (
    plot_embedding_init_pred_proba,
    plot_embedding_top_pred_proba,
    plot_prediction_difference,
    plot_first_change_iteration,
    delta_over_iterations,
    delta_vs_pixel_diff,
    plot_embedding_all_class_prob,
    plot_conf_matrix_orig_embed,
)

# ------------- Constants & Config ---------------------
RES_DIR = "../res"
PATCH_MAP = {"one": 1, "q1": 2, "q2": 3, "q3": 4, "all": 5}
INVERSE_PATCH_MAP = {v: k for k, v in PATCH_MAP.items()}
COLOR_MAP = {"simec": "rgb(229, 134, 6)", "simexp": "rgb(47, 138, 196)"}
FILL_COLOR_MAP = {"simec": "rgba(229, 134, 6, 0.2)", "simexp": "rgba(47, 138, 196, 0.2)"}

# ------------- Data Loading --------------------------
def load_data():
    df = pd.read_parquet(os.path.join(RES_DIR, "all_experiments_results.parquet"))
    npz_data = np.load(os.path.join(RES_DIR, "all_experiments_embeddings.npz"), allow_pickle=True)

    for key in ["distance", "original_image_pred_proba", "embedding_pred_proba", "modified_image_pred_proba", "modified_image"]:
        df[key] = list(npz_data[key])

    df.dropna(axis=1, inplace=True)
    df["input_name"] = df["input_name"].astype(str)

    for col in df.columns:
        if "proba" in col:
            df[col] = df[col].apply(lambda x: softmax(np.stack(x).flatten()))
        elif "pred" in col:
            df[col] = df[col].astype(str)

    df["embedding_pred_init_proba"] = df.apply(
        lambda row: row["embedding_pred_proba"][int(row["original_image_pred"])], axis=1
    )
    df["patch_option"] = df["patch_option"].apply(lambda x: PATCH_MAP[x])
    df.sort_values("iteration", inplace=True)

    return df

# ------------- Aggregation ---------------------------
def aggregate_data(df):
    def aggregate_cell(cell):
        if isinstance(cell[0], (float, int)):
            return np.mean(cell)
        if isinstance(cell[0], str):
            unique = list(set(cell))
            return unique[0] if len(unique) == 1 else cell
        if isinstance(cell[0], np.ndarray):
            return np.mean(np.stack(cell), axis=0)
        return None

    grouped = df.groupby(["dataset", "iteration", "repetition", "patch_option", "delta_multiplier", "algorithm"]).agg(list)
    grouped.drop("input_name", axis=1, inplace=True)
    grouped = grouped.map(aggregate_cell).reset_index()

    grouped["input_name"] = grouped["dataset"].apply(lambda s: f"agg_{s}")
    grouped["distance"] = grouped["distance"].apply(np.mean)

    return pd.concat([df, grouped], ignore_index=True).sort_values(["input_name", "iteration"])

# ------------- Dashboard Layout -----------------------
def create_layout(input_names, delta_mults):
    return html.Div([
        html.H1("SIMEC/SIMEXP Result Dashboard"),
        html.Div([
            html.Label("Select Input Name:"),
            dcc.Dropdown(id="input-name", options=input_names, value=input_names[0])
        ], style={"width": "50%", "float": "left"}),
        html.Div([
            html.Label("Select Delta Mult:"),
            dcc.Dropdown(id="delta-mult", options=delta_mults, value=delta_mults[0])
        ], style={"width": "50%", "float": "left"}),
        html.Label("Select Fraction of Explored Patches:"),
        dcc.Slider(
            id="explore-patches", min=1, max=5, step=1, value=1,
            marks={i: INVERSE_PATCH_MAP[i] for i in range(1, 6)}
        ),
        *[dcc.Graph(id=fig_id, style={"width": "50%", "float": "left"}) for fig_id in [
            "lineplot-embedding", "lineplot-delta", "boxplot-difference", "confusion-matrix",
            "lineplot-simec", "lineplot-simexp", "all-class-pred-proba-simec", "all-class-pred-proba-simexp",
            "conf-matrix-original-modified"
        ]]
    ])

# ------------- App & Callbacks ------------------------
app = dash.Dash(__name__)
df = load_data()
df = aggregate_data(df)

INPUT_NAMES = df["input_name"].unique().tolist()
DELTA_MULTS = df["delta_multiplier"].unique().tolist()
N_CLASSES = df["original_image_pred_proba"].apply(len).unique()[0]

app.layout = create_layout(INPUT_NAMES, DELTA_MULTS)

@app.callback(
    [Output(fig, "figure") for fig in [
        "lineplot-embedding", "lineplot-delta", "boxplot-difference", "confusion-matrix",
        "lineplot-simec", "lineplot-simexp", "all-class-pred-proba-simec", "all-class-pred-proba-simexp",
        "conf-matrix-original-modified"
    ]],
    [Input("input-name", "value"), Input("delta-mult", "value"), Input("explore-patches", "value")]
)
def update_all_plots(input_name, delta_mult, explore_patches_len):
    filtered = df[
        (df["input_name"] == input_name) &
        (df["delta_multiplier"] == delta_mult) &
        (df["patch_option"] == explore_patches_len)
    ].sort_values(by="iteration")

    if filtered.empty:
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text="No data found for selection", showarrow=False,
            font=dict(size=20, family="Arial", color="black")
        )
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), plot_bgcolor="white")
        return [fig] * 9
    
    return generate_all_figures(filtered, input_name)

# ------------- Plotting Utilities -------------------
def generate_all_figures(filtered_df, input_name):
    fig1 = plot_embedding_init_pred_proba(filtered_df)
    fig2 = plot_embedding_top_pred_proba(filtered_df)
    fig3 = plot_prediction_difference(filtered_df)
    fig4 = plot_first_change_iteration(filtered_df)
    fig5 = delta_over_iterations(filtered_df)
    fig6 = delta_vs_pixel_diff(filtered_df)
    fig7 = plot_embedding_all_class_prob(filtered_df, "simec")
    fig8 = plot_embedding_all_class_prob(filtered_df, "simexp")
    fig9 = plot_conf_matrix_orig_embed(filtered_df, N_CLASSES, input_name)

    return [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9]

# ------------- Run the App ---------------------------
if __name__ == "__main__":
    app.run(debug=True) 