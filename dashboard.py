import dash
from dash import dcc, html, Input, Output
import pandas as pd
import numpy as np
from scipy.special import softmax
from scipy.spatial.distance import cityblock as manhattan
import plotly.express as px
from typing import Iterable, List

# Load data
df = pd.read_parquet("all_experiments_results.parquet") 

df.dropna(axis = 1, inplace=True)
df.drop(['input_embedding', 'output_embedding', 'capped_embedding_pred', 'modified_original_pred_proba', 'metadata_path', 'model_path', 'algo', 'objective', 'orig_data_dir'], axis = 1,  inplace = True)
df['input_name'] = df['input_name'].astype(str)

for col in df.columns:
    if "proba" in col:
        df[col] = df[col].apply(lambda x: softmax(np.stack(x).flatten()))
    elif "pred" in col:
        df[col] = df[col].astype(str)
df['embedding_pred_init_proba'] = df.apply(lambda row: row['embedding_pred_proba'][int(row['original_image_pred'])], axis = 1).values.flatten()
df.sort_values("iteration", inplace=True)

patch_map = {
    "one": 1,
    "q1": 2,
    "q2": 3,
    "q3": 4,
    "all": 5
}
inverse_map = {v:k for k,v in patch_map.items()}
df['patches'] = df['patches'].apply(lambda x: patch_map[x])

def aggregation_function(cell):

    if isinstance(cell[0], float):
        return np.mean(cell)
    if isinstance(cell[0], int):
        return int(np.round(np.mean(cell)))
    if isinstance(cell[0], str):
        unique = list(set(cell))
        if len(unique) > 1:
            return cell
        else:
            return unique[0]
    if isinstance(cell[0], np.ndarray):
        return np.mean(np.stack(cell), axis = 0)
    
    return None

aggregated_data = df.groupby(["exp_name", "iteration", "repetition", "patches", "delta_mult", "algorithm"]).agg(list)
aggregated_data.drop("input_name", axis = 1, inplace = True)
aggregated_data = aggregated_data.map(aggregation_function).reset_index()
aggregated_data['input_name'] = aggregated_data['exp_name'].apply(lambda s: "agg_" + s)
df = pd.concat([df, aggregated_data], axis = 0, ignore_index=True)
df.sort_values(['input_name', 'iteration'], inplace=True)

input_names = df['input_name'].unique().tolist()
delta_mults = df['delta_mult'].unique().tolist()
N_classes = df['original_image_pred_proba'].apply(len).unique()[0]

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("SIMEC/SIMEXP Result Dashboard"),
    
    html.Div([
        html.Label("Select Input Name:"),
        dcc.Dropdown(id='input-name', options=input_names, multi=False, value = input_names[0]),
    ], style = {"width": "50%", "float": "left"}),
    
    html.Div([
        html.Label("Select Delta Mult:"),
        dcc.Dropdown(id='delta-mult', options=delta_mults, multi=False, value = delta_mults[0]),
    ], style = {"width": "50%", "float": "left"}),
    
    html.Label("Select Fraction of Explored Patches:"),
    dcc.Slider(id='explore-patches', min=1, max=5, step=1, value=1, 
               marks={i: inverse_map[i] for i in range(1,6)}),
    
    dcc.Graph(id='lineplot-embedding', style = {"width":"50%", "float": "left"}),
    dcc.Graph(id='lineplot-delta', style = {"width":"50%", "float": "left"}),
    dcc.Graph(id='boxplot-difference', style = {"width":"50%", "float": "left"}),
    dcc.Graph(id='confusion-matrix', style = {"width":"50%", "float": "left"}),
    dcc.Graph(id='lineplot-simec', style = {"width":"50%", "float": "left"}),
    dcc.Graph(id='lineplot-simexp', style = {"width":"50%", "float": "left"})
])

@app.callback(
    [Output('lineplot-embedding', 'figure'),
     Output('lineplot-delta', 'figure'),
     Output('boxplot-difference', 'figure'),
     Output('confusion-matrix', 'figure'),
     Output('lineplot-simec', 'figure'),
     Output('lineplot-simexp', 'figure')],
    [Input('input-name', 'value'),
     Input('delta-mult', 'value'),
     Input('explore-patches', 'value')]
)
def update_plots(input_name, delta_mult, explore_patches_len):
    filtered_df = df[(df['input_name'] == input_name) &
                     (df['delta_mult'] == delta_mult) &
                     (df['patches'] == explore_patches_len)]
    
    if filtered_df.empty:
        # Create an empty figure
        fig = px.scatter()

        # Add annotation with the text at the center
        fig.add_annotation(
            x=0.5, y=0.5,
            xref='paper', yref='paper',
            text="Chosen combination is absent",
            showarrow=False,
            font=dict(size=20, family="Arial", color="black")
        )

        # Update layout to remove axes
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor="white"
        )

        return fig, fig, fig, fig, fig, fig

    grouped_df = filtered_df.groupby(['iteration', 'algorithm']).agg(list)
    grouped_df = grouped_df.reset_index(names = ['it', 'alg'])
    grouped_df['embedding_pred_init_proba_mean'] = grouped_df['embedding_pred_init_proba'].apply(np.mean)
    grouped_df['embedding_pred_init_proba_sd'] = grouped_df['embedding_pred_init_proba'].apply(np.std)
    
    # Probability of the original class over iterations
    fig1 = px.line(grouped_df, x='it', y='embedding_pred_init_proba_mean', color='alg', error_y="embedding_pred_init_proba_sd",
                   title='Probability of the Original Class over Iterations')
    
    # Delta over iterations
    grouped_df['delta_mean'] = grouped_df['delta'].apply(np.mean)
    grouped_df['delta_sd'] = grouped_df['delta'].apply(np.std)
    
    fig2 = px.line(grouped_df, x='it', y='delta_mean', color='alg', error_y="delta_sd", 
                   title='Delta over Iterations')
    
    # Prediction probability difference (KLDiv) between decoded image and embedding
    grouped_df['modified_image_pred_proba_mean'] = grouped_df['modified_image_pred_proba'].apply(lambda x: np.mean(x, axis = 0))
    grouped_df['embedding_pred_proba_mean'] = grouped_df['embedding_pred_proba'].apply(lambda x: np.mean(x, axis = 0))
    grouped_df['decoding_difference'] = grouped_df.apply(lambda row: manhattan(row['embedding_pred_proba_mean'], row['modified_image_pred_proba_mean']), axis = 1)
    fig3 = px.box(grouped_df, x='alg', y='decoding_difference', 
                  title='Prediction Probability Difference (L1 distance) between Decoded Image and Embedding')
    
    # Histogram of class mismatch (embedding and decode), i.e. how long the prediction remains in the same class
    filtered_df['element_id'] = filtered_df['original_image_pred'].apply(lambda L: list(range(len(L))) if isinstance(L, List) else [0])
    if isinstance(filtered_df['embedding_pred'].iloc[0], List):
        exploded_df = filtered_df.explode(['original_image_pred','embedding_pred','modified_image_pred', 'element_id'])
    else:
        exploded_df = filtered_df

    exploded_df['embedding_class_mismatch'] = (exploded_df['embedding_pred'] != exploded_df['original_image_pred'])
    exploded_df['decoded_class_mismatch'] = (exploded_df['modified_image_pred'] != exploded_df['original_image_pred'])

    embedding_pred_mismatches = exploded_df[exploded_df['embedding_class_mismatch']].groupby(["algorithm", "repetition", "element_id"]).first().reset_index()
    embedding_pred_mismatches['is_decoded'] = [0]*embedding_pred_mismatches.shape[0]
    decoded_pred_mismatches = exploded_df[exploded_df['decoded_class_mismatch']].groupby(["algorithm", "repetition", "element_id"]).first().reset_index()
    decoded_pred_mismatches['is_decoded'] = [1]*decoded_pred_mismatches.shape[0]
    class_mismatches = pd.concat([embedding_pred_mismatches, decoded_pred_mismatches], axis = 0, ignore_index=True)
    class_mismatches = class_mismatches.groupby(['algorithm', 'is_decoded']).agg(list).reset_index(names = ['alg', 'decoded'])
    class_mismatches['mean_iteration'] = class_mismatches['iteration'].apply(np.mean)
    class_mismatches['sd_iteration'] = class_mismatches['iteration'].apply(np.std)

    fig4 = px.bar(class_mismatches, x = "alg", y = 'mean_iteration', color = "decoded", error_y='sd_iteration',
                  title = "Barchart of First Change wrt the Original Class")

    # Prediction probabilities (from embedding) of all classes (SIMEC only)
    simec_df = filtered_df.loc[filtered_df['algorithm'] == "simec", ['iteration', 'repetition', 'embedding_pred_proba']]
    simec_proba_means = simec_df.groupby("iteration").agg(list).map(lambda x: np.mean(x, axis = 0).tolist())
    simec_proba_means = pd.DataFrame(simec_proba_means['embedding_pred_proba'].tolist(), index = simec_proba_means.index)
    simec_proba_means = pd.melt(simec_proba_means.reset_index().rename({"iteration":"it"}, axis = 1), id_vars = "it", value_vars=list(range(simec_proba_means.shape[1])))
    fig5 = px.line(simec_proba_means, x = "it", y = "value", color ="variable",
                   title="Avg Probabilities of All Classes over Iterations (SIMEC)")

    # Prediction probabilities (from embedding) of all classes (SIMExp only)
    simexp_df = filtered_df.loc[filtered_df['algorithm'] == "simexp", ['iteration', 'repetition', 'embedding_pred_proba']]
    simexp_proba_means = simexp_df.groupby("iteration").agg(list).map(lambda x: np.mean(x, axis = 0).tolist())
    simexp_proba_means = pd.DataFrame(simexp_proba_means['embedding_pred_proba'].tolist(), index = simexp_proba_means.index)
    simexp_proba_means = pd.melt(simexp_proba_means.reset_index().rename({"iteration":"it"}, axis = 1), id_vars = "it", value_vars=list(range(simexp_proba_means.shape[1])))
    fig6 = px.line(simexp_proba_means, x = "it", y = "value", color ="variable",
                   title="Avg Probabilities of All Classes over Iterations (SIMExp)")
    
    return fig1, fig2, fig3, fig4, fig5, fig6

if __name__ == '__main__':
    app.run_server(debug=True)
