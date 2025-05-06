import os
import argparse
import dash
import pandas as pd
import numpy as np
from scipy.special import softmax
from scipy import stats
import plotly.graph_objects as go
from dash import dcc, html, Input, Output
from transformers import BertTokenizerFast
from figures import *

# ------------- Constants & Config ---------------------
RES_DIR = "../res"
PATCH_MAP = {"one": 1, "q1": 2, "q2": 3, "q3": 4, "all": 5, "target-word":6}
INVERSE_PATCH_MAP = {v: k for k, v in PATCH_MAP.items()}
COLOR_MAP = {"simec": "rgb(229, 134, 6)", "simexp": "rgb(47, 138, 196)"}
FILL_COLOR_MAP = {"simec": "rgba(229, 134, 6, 0.2)", "simexp": "rgba(47, 138, 196, 0.2)"}

# ------------- Data Loading --------------------------
def load_data(out_name):
    df = pd.read_parquet(os.path.join(RES_DIR, f"{out_name}_results.parquet"))
    npz_data = np.load(os.path.join(RES_DIR, f"{out_name}_embeddings.npz"), allow_pickle=True)

    for key in ["distance", "original_image_pred_proba", "embedding_pred_proba", "modified_image_pred_proba", "modified_image", "evaluated_tokens", "embedding_proba_diff", "modified_image_diff", "embedding_pred_init_proba"]:
        df[key] = list(npz_data[key])

    df["input_name"] = df["input_name"].astype(str) + "_" + df["dataset"].astype(str)

    for col in df.columns:
        if "pred" in col and "proba" not in col:
            df[col] = df[col].astype(str)
    df["embedding_pred_proba_max"] = df["embedding_pred_proba"].apply(np.nanmax)
    df["patch_option"] = df["patch_option"].apply(lambda x: PATCH_MAP[x])
    df.sort_values("iteration", inplace=True)

    return df

# ------------- Aggregation ---------------------------
def aggregate_data(df):
    # --- Grouping --- 
    grouped = df.groupby(
        ["dataset", "iteration", "repetition", "patch_option", "delta_multiplier", "algorithm"]
    )
    # Drop problematic columns before mapping
    #grouped = grouped.drop(columns=["input_name"], errors="ignore")
    # --- Apply aggregation to all other columns ---
    new_agg_rows = []

    for (dataset, iteration, repetition, patch_option, delta_multiplier, algorithm), group in grouped:         
        input_name = f"agg_{dataset}"
        explored_patches = set([c.item() for c in group.explored_patches.explode()]) 
        delta = group.delta.mean()
        distance = np.nanmean(np.concatenate(group.distance.values))
        embedding_pred = group.embedding_pred.values
        embedding_pred_init_proba = np.nanmean(group.embedding_pred_init_proba)
        embedding_pred_proba = np.nanmean(np.stack(group.embedding_pred_proba), axis = 0).squeeze()
        embedding_pred_proba_max = np.nanmean(np.stack(group.embedding_pred_proba_max), axis = 0).squeeze()
        embedding_proba_diff = None if group.embedding_proba_diff.iloc[0] is None else np.nanmean(np.abs(np.stack(group.embedding_proba_diff)), axis = 0)
        evaluated_tokens = None if group.evaluated_tokens.iloc[0] == [-1] else group.evaluated_tokens.iloc[0]
        input_embedding_max = np.mean(group.input_embedding_max)
        input_embedding_norm = np.mean(group.input_embedding_norm)
        modified_image = None  # You can replace with actual aggregation logic if needed
        if group.modified_image_diff.iloc[0] is None or np.all(np.isnan(group.modified_image_diff.iloc[0])):
            # either first iteration (diffs starts from second iteration) or the only explored patch is CLS, so no real diffs in patches
            modified_image_diff = None
        else:
            modified_image_diff = np.nanmean(np.abs(np.stack(group.modified_image_diff)), axis = 0, keepdims=True)
        modified_image_pred = group.modified_image_pred.values
        modified_image_pred_proba = np.nanmean(np.stack(group.modified_image_pred_proba), axis = 0)
        number_explored_patches = np.nanmean(group.number_explored_patches)
        original_image_pred = group.original_image_pred.values
        original_image_pred_proba = np.nanmean(np.stack(group.original_image_pred_proba), axis = 0)
        patch_attribution = None if group.patch_attribution.iloc[0] is None else np.nanmean(np.concatenate(group.patch_attribution.values))

        new_agg_rows.append({
            "dataset": dataset,
            "iteration": iteration,
            "repetition": repetition,
            "patch_option": patch_option,
            "delta_multiplier": delta_multiplier,
            "algorithm": algorithm,
            "input_name": input_name,
            "explored_patches": explored_patches,
            "delta": delta,
            "distance": distance,
            "embedding_pred": embedding_pred,
            "embedding_pred_init_proba": embedding_pred_init_proba,
            "embedding_pred_proba": embedding_pred_proba,
            "embedding_pred_proba_max":embedding_pred_proba_max,
            "embedding_proba_diff": embedding_proba_diff,
            "evaluated_tokens": evaluated_tokens,
            "input_embedding_max": input_embedding_max,
            "input_embedding_norm": input_embedding_norm,
            "modified_image": modified_image,
            "modified_image_diff": modified_image_diff,
            "modified_image_pred": modified_image_pred,
            "modified_image_pred_proba": modified_image_pred_proba,
            "number_explored_patches": number_explored_patches,
            "original_image_pred": original_image_pred,
            "original_image_pred_proba": original_image_pred_proba,
            "patch_attribution": patch_attribution
        })

    # Convert to DataFrame and merge
    agg_df = pd.DataFrame(new_agg_rows)
    df = pd.concat([df, agg_df], ignore_index=True).sort_values(["input_name", "iteration"])
    return df

def align_probs_to_vocab(tokens, probs, vocab):
    if probs is None:
        return None # first diff is not existing
    token_to_prob = dict(zip(tokens, probs))
    return np.array([token_to_prob.get(tok, np.nan) for tok in vocab])

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
            id="explore-patches", min=1, max=6, step=1, value=1,
            marks={i: INVERSE_PATCH_MAP[i] for i in range(1, 7)}
        ),
        *[dcc.Graph(id=fig_id, style={"width": "50%", "float": "left"}) for fig_id in [
            "lineplot-embedding",  "confusion-matrix",
            "lineplot-simec", "lineplot-simexp", "all-class-pred-proba-simec", "all-class-pred-proba-simexp",
            "conf-matrix-original-modified", "conf-matrix-original-embedding","scatter-norm_vs_pixel_diff", "scatter-norm_vs_proba_diff", "scatter-max_vs_pixel_diff", "scatter-max_vs_proba_diff", 
        ]]
    ])

# ------------- App & Callbacks ------------------------
app = dash.Dash(__name__)
df = load_data("all_experiments")
vocab =  sorted(set(token for tokens in df[df["dataset"] == "winobias"]['evaluated_tokens'] for token in tokens))
mask = df["dataset"] == "winobias"
df.loc[mask, "embedding_pred_proba"] = df.loc[mask].apply(
    lambda row: align_probs_to_vocab(row['evaluated_tokens'], row['embedding_pred_proba'], vocab), 
    axis=1
)
df.loc[mask, "modified_image_pred_proba"] = df.loc[mask].apply(
    lambda row: align_probs_to_vocab(row['evaluated_tokens'], row['modified_image_pred_proba'], vocab), 
    axis=1
)
df.loc[mask, "embedding_proba_diff"] = df.loc[mask].apply(
    lambda row: align_probs_to_vocab(row['evaluated_tokens'], row['embedding_proba_diff'], vocab), 
    axis=1
)
df.loc[mask,"evaluated_tokens"] = pd.Series([list(vocab)] * mask.sum(), index=df.index[mask])
df = aggregate_data(df)

INPUT_NAMES = df["input_name"].unique().tolist()
DELTA_MULTS = df["delta_multiplier"].unique().tolist()
N_CLASSES = df["original_image_pred_proba"].apply(len).unique()[0]
CLASS_LABELS = {"cifar":["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"], "mnist":list(map(str, range(10))), "winobias":BertTokenizerFast.from_pretrained("gaunernst/bert-small-uncased"), "hatespeech": ["positive", "neutral", "hatespeech"]}

app.layout = create_layout(INPUT_NAMES, DELTA_MULTS)

@app.callback(
    [Output(fig, "figure") for fig in [
        "lineplot-embedding", "confusion-matrix",
        "lineplot-simec", "lineplot-simexp", "all-class-pred-proba-simec", "all-class-pred-proba-simexp",
        "conf-matrix-original-modified","conf-matrix-original-embedding", "scatter-norm_vs_pixel_diff", "scatter-norm_vs_proba_diff", "scatter-max_vs_pixel_diff", "scatter-max_vs_proba_diff", 
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
        return [fig] * 12
    
    return generate_all_figures(filtered, input_name)

# ------------- Plotting Utilities -------------------
def generate_all_figures(filtered_df, input_name):
    class_labels = CLASS_LABELS[filtered_df["dataset"].values[0]]
    fig1 = plot_embedding_init_pred_proba(filtered_df)
    fig2 = plot_embedding_top_pred_proba(filtered_df)
    fig3 = plot_prediction_difference(filtered_df)
    #fig4 = plot_first_change_iteration(filtered_df)
    #fig5 = delta_over_iterations(filtered_df)
    fig6 = delta_vs_pixel_diff(filtered_df)
    fig7 = plot_embedding_all_class_prob(filtered_df, "simec", class_labels)
    fig8 = plot_embedding_all_class_prob(filtered_df, "simexp", class_labels)
    fig9 = plot_conf_matrix_orig_modified(filtered_df, class_labels, input_name)
    fig9a = plot_conf_matrix_orig_embed(filtered_df, class_labels, input_name)
    fig10 = norm_vs_pixel_diff(filtered_df)
    fig11 = norm_vs_proba_diff(filtered_df)
    fig12 = max_vs_pixel_diff(filtered_df)
    fig13 = max_vs_proba_diff(filtered_df)

    return [fig1, fig2, fig3, fig6, fig7, fig8, fig9, fig9a, fig10, fig11, fig12, fig13]

# ------------- Run the App ---------------------------

if __name__ == "__main__":   
    app.run(debug=True) 