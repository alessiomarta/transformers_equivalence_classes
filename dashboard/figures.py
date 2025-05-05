from typing import List
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cityblock
import torch


COLOR_MAP = {"simec": "rgb(229, 134, 6)", "simexp": "rgb(47, 138, 196)"}
FILL_COLOR_MAP = {"simec": "rgba(229, 134, 6, 0.2)", "simexp": "rgba(47, 138, 196, 0.2)"}

def plot_embedding_init_pred_proba(filtered):
    grouped_df = (
        filtered.groupby(["iteration", "algorithm"])
        .agg(list)
        .reset_index()
    )
    grouped_df["embedding_pred_init_proba_mean"] = grouped_df[
        "embedding_pred_init_proba"
    ].apply(lambda x: np.nanmean(np.stack(x)))
    
    grouped_df["embedding_pred_init_proba_sd"] = grouped_df[
        "embedding_pred_init_proba"
    ].apply(lambda x: np.nanstd(np.stack(x)))
    
    fig = go.Figure()
    for alg in grouped_df["algorithm"].unique():
        subset = grouped_df[grouped_df["algorithm"] == alg].sort_values(by="iteration")
        x = subset["iteration"]
        y = subset["embedding_pred_init_proba_mean"]
        y_sd = subset["embedding_pred_init_proba_sd"]

        # Fill area between mean + std and mean - std
        fig.add_trace(go.Scatter(
            x=list(x) + list(x[::-1]),
            y=list(y + y_sd) + list((y - y_sd)[::-1]),
            fill='toself',
            fillcolor=FILL_COLOR_MAP[alg],
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=False
        ))

        # Mean line
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name=alg,
            line=dict(color=COLOR_MAP[alg]),
        ))

    fig.update_layout(
        title="Probability of the Original Class over Iterations, evaluated on embedding",
        xaxis_title="Iteration",
        yaxis_title="Probability",
        template="plotly_white"
    )
    return fig

def plot_embedding_top_pred_proba(filtered):
    grouped_df = (
        filtered.groupby(["iteration", "algorithm"])
        .agg(list)
        .reset_index()
    )
    grouped_df["embedding_pred_max_proba_mean"] = grouped_df[ 
        "embedding_pred_proba"
    ].apply(lambda x: np.nanmean(np.nanmax(x, axis=-1)))
    grouped_df["embedding_pred_max_proba_sd"] = grouped_df[
        "embedding_pred_proba"
    ].apply(lambda x: np.nanstd(np.nanmax(x, axis=-1)))
    fig = go.Figure()
    for alg in grouped_df["algorithm"].unique():
        subset = grouped_df[grouped_df["algorithm"] == alg].sort_values(by="iteration")
        x = subset["iteration"]
        y = subset["embedding_pred_max_proba_mean"]
        y_sd = subset["embedding_pred_max_proba_sd"]
        fig.add_trace(go.Scatter(
            x=list(x) + list(x[::-1]),
            y=list(y + y_sd) + list((y - y_sd)[::-1]),
            fill='toself',
            fillcolor=FILL_COLOR_MAP[alg],
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=False
        ))

        # Mean line
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name=alg,
            line=dict(color=COLOR_MAP[alg]),
        ))
        
    fig.update_layout(
        title="Probability of the Top Class over Iterations, evaluated on embedding",
        xaxis_title="Iteration",
        yaxis_title="Probability",
        template="plotly_white"
    )
    return fig

def safe_cityblock(a, b):
    a = np.array(a)
    b = np.array(b)
    mask = ~np.isnan(a) & ~np.isnan(b)
    if not np.any(mask):
        return np.nan  # or 0, depending on how you want to treat missing comparisons
    return cityblock(a[mask], b[mask])

def plot_prediction_difference(filtered):
    grouped_df = (
        filtered.groupby(["iteration", "algorithm"])
        .agg(list)
        .reset_index()
    )
    grouped_df["modified_image_pred_proba_mean"] = grouped_df[
        "modified_image_pred_proba"
    ].apply(lambda x: np.nanmean(x, axis=0))
    
    grouped_df["embedding_pred_proba_mean"] = grouped_df[
        "embedding_pred_proba"
    ].apply(lambda x: np.nanmean(x, axis=0))
    
    grouped_df["decoding_difference"] = grouped_df.apply(
        lambda row: safe_cityblock(
            row["embedding_pred_proba_mean"], row["modified_image_pred_proba_mean"]
        ),
        axis=1,
    )

    fig = px.box(
        grouped_df,
        x="algorithm",
        y="decoding_difference",
        color="algorithm",
        color_discrete_map=COLOR_MAP,
        title="Prediction Probability Difference (L1 distance) between Decoded Image and Embedding",
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Algorithm",
        yaxis_title="L1 Distance",
        showlegend=False,
    )
    
    return fig
    
def plot_first_change_iteration(filtered):
    filtered = filtered.copy()
    filtered["element_id"] = filtered["original_image_pred"].apply(
        lambda L: list(range(len(L))) if isinstance(L, list) else [0]
    )

    if isinstance(filtered["embedding_pred"].iloc[0], list):
        exploded_df = filtered.explode(
            ["original_image_pred", "embedding_pred", "modified_image_pred", "element_id"]
        ).reset_index(drop=True)
    else:
        exploded_df = filtered

    # Create mismatch flags
    exploded_df["embedding_class_mismatch"] = (
        exploded_df["embedding_pred"] != exploded_df["original_image_pred"]
    )
    exploded_df["decoded_class_mismatch"] = (
        exploded_df["modified_image_pred"] != exploded_df["original_image_pred"]
    )

    # Get first mismatch per element
    def get_mismatches(df, colname, label):
        try:
            mismatches = (
                df[df[colname]]
                .groupby(["algorithm", "repetition", "element_id"])
                .first()
                .reset_index()
            )
            mismatches["Prediction Type"] = label
        except TypeError:
            df_temp = df[df[colname]].copy()
            df_temp["element_id"] = df_temp["element_id"].apply(lambda x: x[0])
            mismatches = df_temp.groupby(["algorithm", "repetition", "element_id"]).first().reset_index()
            mismatches["Prediction Type"] = label
        return mismatches

    embedding_mismatches = get_mismatches(exploded_df, "embedding_class_mismatch", "Embedding")
    decoded_mismatches = get_mismatches(exploded_df, "decoded_class_mismatch", "Decoded")

    class_mismatches = pd.concat([embedding_mismatches, decoded_mismatches], ignore_index=True)

    # Aggregate stats
    summary = (
        class_mismatches.groupby(["algorithm", "Prediction Type"])["iteration"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "Mean Iteration", "std": "Std Dev"})
    )

    # Plot as grouped bar chart
    fig = px.bar(
        summary,
        x="algorithm",
        y="Mean Iteration",
        color="Prediction Type",
        barmode="group",
        error_y="Std Dev",
        title="Average Iteration of First Class Change",
        labels={"algorithm": "Algorithm"}
    )

    fig.update_layout(
        template="plotly_white",
        yaxis_title="Mean Iteration",
        legend_title="Prediction Type"
    )

    return fig
    
def delta_over_iterations(filtered):
    grouped_df = (
        filtered.groupby(["iteration", "algorithm"])
        .agg(list)
        .reset_index()
    )

    # Compute stats
    grouped_df["delta_mean"] = grouped_df["delta"].apply(np.mean)
    grouped_df["delta_sd"] = grouped_df["delta"].apply(np.std)

    fig = go.Figure()

    for alg in grouped_df["algorithm"].unique():
        df_alg = grouped_df[grouped_df["algorithm"] == alg].sort_values(by="iteration")

        x = df_alg["iteration"]
        y = df_alg["delta_mean"]
        y_upper = y + df_alg["delta_sd"]
        y_lower = y - df_alg["delta_sd"]

        # Fill area between mean ± std
        fig.add_trace(go.Scatter(
            x=list(x) + list(x[::-1]),
            y=list(y_upper) + list(y_lower[::-1]),
            fill='toself',
            fillcolor=FILL_COLOR_MAP[alg],
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=False,
        ))

        # Mean line
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name=alg,
            line=dict(color=COLOR_MAP[alg]),
        ))

    fig.update_layout(
        title="Delta over Iterations",
        xaxis_title="Iteration",
        yaxis_title="Delta",
        template="plotly_white"
    )

    return fig

def delta_vs_pixel_diff(filtered):
    def custom_diff(current, previous):
        if pd.isna(current).all():
            return None
        if previous is None:
            return None
        # Example: Return True if different, else False
        return current != previous
    
    grouped_df = (
        filtered.groupby(["iteration", "algorithm"])
        .agg(list)
        .reset_index() 
    )
    grouped_df["delta_mean"] = grouped_df["delta"].apply(np.mean)
    grouped_df["delta_sd"] = grouped_df["delta"].apply(np.std) 
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True, 
        vertical_spacing=0.1,
        subplot_titles=(
            "Pixel difference",
            "Delta",
        )
    )
    
    for alg in filtered['algorithm'].unique():
        fig6df = filtered[filtered['algorithm'] == alg].sort_values(
            ["dataset", "input_name", "delta_multiplier", "patch_option", "iteration", "repetition"]
        )
        fig6df["tot_patch_diff"] = fig6df["modified_image_diff"].apply(
                lambda x: np.mean(np.abs(x)) if isinstance(x, (np.ndarray, torch.Tensor)) else np.nan
            ) 

        # Step 3: Group and aggregate mean of patch_diff
        mean_patch_diff_df = (
            fig6df
            .groupby(["dataset", "input_name", "delta_multiplier", "patch_option", "iteration"])
        )
        x = mean_patch_diff_df.agg("count").reset_index()["iteration"]
        mean_patch_diff_df = (
            mean_patch_diff_df["tot_patch_diff"]
            .agg(mean_patch_diff="mean", std_patch_diff="std")
        )
        
        # === First plot: tot_patch_diff
        y1 = mean_patch_diff_df['mean_patch_diff'].values
        y1_upper = y1 + mean_patch_diff_df['std_patch_diff'].values
        y1_lower = y1 - mean_patch_diff_df['std_patch_diff'].values

        fig.add_trace(go.Scatter(x=x, y=y1_upper, mode='lines', line=dict(width=0),
                                showlegend=False, hoverinfo='skip'), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=y1_lower, mode='lines', fill='tonexty',
                                fillcolor=FILL_COLOR_MAP[alg], line=dict(width=0),
                                showlegend=False, hoverinfo='skip'), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name=f'{alg}',
                                line=dict(color=COLOR_MAP[alg])), row=1, col=1)

        # === Second plot: delta mean
        y2 = grouped_df[grouped_df['algorithm'] == alg]["delta_mean"].values
        y2_upper = y2 + grouped_df[grouped_df['algorithm'] == alg]["delta_sd"].values
        y2_lower = y2 - grouped_df[grouped_df['algorithm'] == alg]["delta_sd"].values

        fig.add_trace(go.Scatter(x=x, y=y2_upper, mode='lines', line=dict(width=0),
                                showlegend=False, hoverinfo='skip'), row=2, col=1)
        fig.add_trace(go.Scatter(x=x, y=y2_lower, mode='lines', fill='tonexty',
                                fillcolor=FILL_COLOR_MAP[alg], line=dict(width=0),
                                showlegend=False, hoverinfo='skip'), row=2, col=1)
        fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name=f'{alg}',
                                line=dict(color=COLOR_MAP[alg]), showlegend=False), row=2, col=1)

    fig.update_layout(
        height=700,
        title_text="Pixel differences compared to delta, over iterations",
        xaxis2_title="Iteration",
        template="plotly_white"
    )
    return fig

def plot_embedding_all_class_prob(filtered, algo, class_labels):
    algo_df = filtered.loc[
        filtered["algorithm"] == algo,
        ["iteration", "repetition", "embedding_pred_proba"]
    ].sort_values(by="iteration")
    # Group by iteration and compute mean class probabilities
    grouped = (
        algo_df.groupby("iteration")["embedding_pred_proba"]
        .apply(lambda probs: np.nanmean(np.stack(probs), axis=0))
        .reset_index()
    )
    # Convert list of probs to DataFrame with one column per class
    class_probs_df = grouped["embedding_pred_proba"].apply(pd.Series)
    class_probs_df["iteration"] = grouped["iteration"]

    # Melt to long format for plotting
    melted = pd.melt(
        class_probs_df,
        id_vars="iteration",
        var_name="class",
        value_name="probability"
    )
    if filtered["dataset"].values[0] != "winobias":
        melted["class_label"] = melted["class"].astype(int).map(lambda x: class_labels[int(x)])
    else:
        evaluated_tokens = list(filtered["evaluated_tokens"].values[0])
        melted["class_label"] = melted["class"].map(lambda x: class_labels.convert_ids_to_tokens(evaluated_tokens[x]))

    # Create line plot
    fig = px.line(
        melted,
        x="iteration",
        y="probability",
        color="class_label",
        title=f"Avg Probabilities of All Classes over Iterations ({algo.upper()})"
    )
    
    fig.update_layout(
        xaxis_title="Iteration",
        yaxis_title="Probability",
        template="plotly_white"
    )

    return fig

def plot_conf_matrix_orig_modified(filtered, class_labels, input_name=None, normalize=False):
    
    if filtered["dataset"].values[0] != "winobias":
        class_labels_numeric = list(map(str, range(len(class_labels))))
    else:
        class_labels_numeric = list(map(str, filtered["evaluated_tokens"].values[0]))
        class_labels = [class_labels.convert_ids_to_tokens(t) for t in filtered["evaluated_tokens"].values[0]]
    # Extract true and predicted labels depending on input_name
    if input_name and input_name.startswith("agg_"):
        y_true = filtered["original_image_pred"].explode()
        y_pred = filtered["modified_image_pred"].explode()
    else:
        y_true = filtered["original_image_pred"]
        y_pred = filtered["modified_image_pred"] 

    # Convert to string for consistent labeling
    y_true = y_true.astype(str)
    y_pred = y_pred.astype(str)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_labels_numeric)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)  # Avoid NaNs from division by 0

    # Create heatmap
    fig = px.imshow(
        cm,
        x=class_labels,
        y=class_labels,
        color_continuous_scale="Blues",
        text_auto=".2f" if normalize else True,
        labels=dict(x="Predicted", y="True", color="Count" if not normalize else "Proportion"),
        title="Confusion Matrix: Original vs Modified Predictions"
    )

    fig.update_layout(
        xaxis_title="Modified Image Class",
        yaxis_title="Original Image Class",
        template="plotly_white"
    )

    return fig

def plot_conf_matrix_orig_embed(filtered, class_labels, input_name=None, normalize=False):
    if filtered["dataset"].values[0] != "winobias":
        class_labels_numeric = list(map(str, range(len(class_labels))))
    else:
        class_labels_numeric = list(map(str, filtered["evaluated_tokens"].values[0]))
        class_labels = [class_labels.convert_ids_to_tokens(t) for t in filtered["evaluated_tokens"].values[0]]
    # Extract true and predicted labels depending on input_name
    if input_name and input_name.startswith("agg_"):
        y_true = filtered["original_image_pred"].explode()
        y_pred = filtered["embedding_pred"].explode()
    else:
        y_true = filtered["original_image_pred"]
        y_pred = filtered["embedding_pred"]

    # Convert to string for consistent labeling
    y_true = y_true.astype(str)
    y_pred = y_pred.astype(str)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_labels_numeric)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)  # Avoid NaNs from division by 0

    # Create heatmap
    fig = px.imshow(
        cm,
        x=class_labels,
        y=class_labels,
        color_continuous_scale="Blues",
        text_auto=".2f" if normalize else True,
        labels=dict(x="Predicted", y="True", color="Count" if not normalize else "Proportion"),
        title="Confusion Matrix: Original vs Embedding Predictions"
    )

    fig.update_layout(
        xaxis_title="Embedding Class",
        yaxis_title="Original Image Class",
        template="plotly_white"
    )

    return fig

def norm_vs_pixel_diff(filtered):
    # Step 1: Sort so diffs are meaningful
    df = filtered.sort_values(
        ["algorithm", "dataset", "input_name", "delta_multiplier", "patch_option", "iteration", "repetition"]
    )

    # Step 2: Compute difference with a lag of 3
    df["patch_diff"] = df["modified_image_diff"].apply(
                lambda x: np.mean(np.abs(x)) if isinstance(x, (np.ndarray, torch.Tensor)) else np.nan
            )
    #df = df.dropna()
    fig = px.scatter(df, x="input_embedding_norm", y="patch_diff", color="algorithm", hover_data=["input_embedding_norm", "patch_diff", "algorithm", "iteration"], color_discrete_map=COLOR_MAP,)
    fig.update_layout(
        title="Embedding norm vs Pixel diff",
        xaxis_title="Input Embedding norm",
        yaxis_title="Pixel difference",
        template="plotly_white"
    )
    return fig

def norm_vs_proba_diff(filtered):
    # Step 1: Sort so diffs are meaningful
    df = filtered.sort_values(
        ["algorithm", "dataset", "input_name", "delta_multiplier", "patch_option", "iteration", "repetition"]
    )

    # Step 2: Compute difference with a lag of 3
    df["proba_diff"] = df["embedding_proba_diff"].apply(
                lambda x: np.nanmean(np.abs(x)) if isinstance(x, (np.ndarray, torch.Tensor)) else np.nan
            )
    #df = df.dropna()
    fig = px.scatter(df, x="input_embedding_norm", y="proba_diff", color="algorithm", hover_data=["input_embedding_norm", "proba_diff", "algorithm", "iteration"], color_discrete_map=COLOR_MAP,)
    fig.update_layout(
        title="Embedding norm vs Embedding probabilities diff",
        xaxis_title="Input Embedding norm",
        yaxis_title="Probabilities difference",
        template="plotly_white"
    )
    return fig

def max_vs_pixel_diff(filtered):
    # Step 1: Sort so diffs are meaningful
    df = filtered.sort_values(
        ["algorithm", "dataset", "input_name", "delta_multiplier", "patch_option", "iteration", "repetition"]
    )

    # Step 2: Compute difference with a lag of 3
    df["patch_diff"] = df["modified_image_diff"].apply(
                lambda x: np.mean(np.abs(x)) if isinstance(x, (np.ndarray, torch.Tensor)) else np.nan
            )
    #df = df.dropna()
    fig = px.scatter(df, x="input_embedding_max", y="patch_diff", color="algorithm", hover_data=["input_embedding_max", "patch_diff", "algorithm", "iteration"], color_discrete_map=COLOR_MAP,)
    fig.update_layout(
        title="Embedding max vs Pixel diff",
        xaxis_title="Input Embedding max",
        yaxis_title="Pixel difference",
        template="plotly_white"
    )
    return fig

def max_vs_proba_diff(filtered):
    # Step 1: Sort so diffs are meaningful
    df = filtered.sort_values(
        ["algorithm", "dataset", "input_name", "delta_multiplier", "patch_option", "iteration", "repetition"]
    )

    # Step 2: Compute difference with a lag of 3
    df["proba_diff"] = df["embedding_proba_diff"].apply(
                lambda x: np.nanmean(np.abs(x)) if isinstance(x, (np.ndarray, torch.Tensor)) else np.nan
            )
    #df = df.dropna()
    fig = px.scatter(df, x="input_embedding_max", y="proba_diff", color="algorithm", hover_data=["input_embedding_max", "proba_diff", "algorithm", "iteration"], color_discrete_map=COLOR_MAP,)
    fig.update_layout(
        title="Embedding max vs Embedding probabilities diff",
        xaxis_title="Input Embedding max",
        yaxis_title="Probabilities difference",
        template="plotly_white"
    )
    return fig