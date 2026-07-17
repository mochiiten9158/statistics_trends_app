import os
from pathlib import Path
from pprint import pp
import subprocess
import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch

from generate_plots_rensink import generate_harrison_scatter
from generate_pc_rensink import generate_harrison_pcp
from generate_ol_rensink import generate_harrison_ordered_line


method = "positive scatterplots"
# method = "positive parallel coordinates"
# method = "positive ordered lines"


def generate_synthetic_data_training(method):
    seed = 0
    corrs_positive = [round(i * 0.0025, 4) for i in range(0, 401)]
    if method == "positive scatterplots":
        count = 1
        for corr in corrs_positive:
            generate_harrison_scatter(corr=corr, seed=seed, out_dir=f"assets/statistics_evolution_corr/positive_scatterplots", count=count)
            count += 1
            seed += 1

    elif method == "positive parallel coordinates":
        count = 1
        for corr in corrs_positive:
            generate_harrison_pcp(corr=corr, seed=seed, out_dir=f"assets/statistics_evolution_corr/positive_pcp", count=count)
            count += 1
            seed += 1

    elif method == "positive ordered lines":
        count = 1
        for corr in corrs_positive:
            generate_harrison_ordered_line(corr=corr, seed=seed, out_dir=f"assets/statistics_evolution_corr/positive_ol", count=count)
            count += 1
            seed += 1


def extract_ps_stats_from_images(input_dir="assets/training", stats_bin="./extract_stats", output_base_dir="summary_stats_corr"):
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return
        
    if not os.path.exists(stats_bin):
        print(f"Error: C++ binary '{stats_bin}' not found.")
        return

    rbase_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])
    if not rbase_dirs:
        print(f"No subfolders found inside {input_dir}")
        return
    print(f"Found {len(rbase_dirs)} correlation base directories in {input_dir}.")


    tasks = []

    def collect_tasks(image_folder_path, output_folder_path):
        images = sorted(list(image_folder_path.glob("*.png")))
        if not images:
            return
        output_folder_path.mkdir(parents=True, exist_ok=True)
        for img_path in images:
            out_stats_path = output_folder_path / f"{img_path.stem}.csv"
            tasks.append((img_path, out_stats_path))

    for rbase_dir in rbase_dirs:
        current_out_dir = Path(output_base_dir) / rbase_dir.name
        collect_tasks(rbase_dir, current_out_dir)
        
        nested_rbase_dir = rbase_dir / "rbase"
        if nested_rbase_dir.exists() and nested_rbase_dir.is_dir():
            nested_out_dir = current_out_dir / "rbase"
            collect_tasks(nested_rbase_dir, nested_out_dir)

    total_tasks = len(tasks)
    if total_tasks == 0:
        print("No PNG images found to process.")
        return

    def worker_thread(img_path, out_stats_path):
        try:
            subprocess.run(
                [stats_bin, str(img_path), str(out_stats_path)],
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL, 
                check=True
            )
            return True, img_path.name
        except subprocess.CalledProcessError:
            return False, img_path.name

    print(f"\n[THREADS] Launching concurrent threads to process {total_tasks} images...")

    count = 0
    with ThreadPoolExecutor(max_workers=None) as executor:
        futures = {executor.submit(worker_thread, img, out): img for img, out in tasks}
        
        for future in as_completed(futures):
            count += 1
            success, img_name = future.result()
            
            print(f"\r    [{count:3d}/{total_tasks:3d}] Processed {img_name}", end="", flush=True)
            if not success:
                print(f"\n    [ERROR] C++ binary failed on image: {img_name}")

    print(f"\nDone. All statistics saved to base directory: '{output_base_dir}'")


# print(" -> Generating synthetic plots...")
# generate_synthetic_data_training(method=method)
# print(" -> Extracting Portilla-Simoncelli statistics via C++...")
# extract_ps_stats_from_images(
#         input_dir="assets/statistics_evolution_corr", 
#         stats_bin="./extract_stats", 
#         output_base_dir="summary_stats_corr")
    
# ==============================
# LOAD ELBOW DATA
# ==============================

scatter_csv_path = "lda_feature_importance_scatterplots.csv"
pcp_csv_path     = "lda_feature_importance_parallel_coordinates.csv"

df_scatter = pd.read_csv(scatter_csv_path)
df_pcp     = pd.read_csv(pcp_csv_path)

# Normalize
df_scatter['Normalized_Importance'] = (
    df_scatter['Mean_Absolute_Importance'] / df_scatter['Mean_Absolute_Importance'].sum()
) * 100

df_pcp['Normalized_Importance'] = (
    df_pcp['Mean_Absolute_Importance'] / df_pcp['Mean_Absolute_Importance'].sum()
) * 100

# Sort
df_scatter_sorted = df_scatter.sort_values(by='Normalized_Importance', ascending=False).reset_index(drop=True)
df_pcp_sorted     = df_pcp.sort_values(by='Normalized_Importance', ascending=False).reset_index(drop=True)

df_scatter_sorted['Plot_Rank'] = df_scatter_sorted.index + 1
df_pcp_sorted['Plot_Rank']     = df_pcp_sorted.index + 1

# ==============================
# LOAD STAT DATA (PCP + SCATTER)
# ==============================

def load_stats(data_dir):
    num_stats = 1254
    correlations = np.arange(0, 1.0025, 0.0025)

    all_data = []

    for i, corr in enumerate(correlations, start=1):
        filename = f"{i}_{corr:.4f}.csv"
        filepath = os.path.join(data_dir, filename)

        if os.path.exists(filepath):
            df_file = pd.read_csv(filepath, header=None)
            arr = pd.to_numeric(df_file.values.flatten(), errors='coerce')
            arr = arr[~np.isnan(arr)]

            if len(arr) >= num_stats:
                arr = arr[:num_stats]

            if len(arr) == num_stats:
                all_data.append(arr)
            else:
                all_data.append(np.full(num_stats, np.nan))
        else:
            all_data.append(np.full(num_stats, np.nan))

    df = pd.DataFrame(all_data, index=correlations)
    df.columns = [f"Stat_{i}" for i in range(1, num_stats+1)]

    df = (df - df.mean()) / df.std()

    return df

df_pcp_stats     = load_stats(os.path.join('summary_stats_corr', 'positive_pcp'))
df_scatter_stats = load_stats(os.path.join('summary_stats_corr', 'positive_scatterplots'))


def build_stat_names():
    stat_names = {}

    # -------------------------
    # 1–10: Low-pass skew/kurt
    # -------------------------
    scale = 1
    for i in range(1, 11, 2):
        stat_names[f"Stat_{i}"]   = f"{i}: Low-pass Scale {scale} Skewness"
        stat_names[f"Stat_{i+1}"] = f"{i+1}: Low-pass Scale {scale} Kurtosis"
        scale += 1

    # -------------------------
    # 11: High-band variance
    # -------------------------
    stat_names["Stat_11"] = "11: High-band Variance"

    # -------------------------
    # 12–17: Pixel stats
    # -------------------------
    pixel_stats = ["Min", "Max", "Mean", "Variance", "Skewness", "Kurtosis"]
    for i, name in enumerate(pixel_stats, start=12):
        stat_names[f"Stat_{i}"] = f"{i}: Pixel {name}"

    # -------------------------
    # 18–262: Low-pass autocorr
    # -------------------------
    idx = 18
    shifts = [(dx, dy) for dx in range(-3, 4) for dy in range(-3, 4)]

    for scale in range(1, 6):
        for dx, dy in shifts:
            stat_names[f"Stat_{idx}"] = (
                f"{idx}: Low-pass Scale {scale} Autocorr Shift ({dx},{dy})"
            )
            idx += 1

    # -------------------------
    # 263–1046: Oriented autocorr
    # -------------------------
    orientations = ["0°", "45°", "90°", "135°"]

    for scale in range(1, 5):
        for orient in orientations:
            for dx, dy in shifts:
                stat_names[f"Stat_{idx}"] = (
                    f"{idx}: Scale {scale} Orient {orient} Autocorr Shift ({dx},{dy})"
                )
                idx += 1

    # -------------------------
    # 1047–1110: Same-scale cross corr
    # -------------------------
    for scale in range(1, 5):
        for o1 in orientations:
            for o2 in orientations:
                stat_names[f"Stat_{idx}"] = (
                    f"{idx}: Scale {scale} Cross Corr ({o1} × {o2})"
                )
                idx += 1

    # -------------------------
    # 1111–1158: Cross-scale corr
    # -------------------------
    for scale in range(1, 4):
        for o1 in orientations:
            for o2 in orientations:
                stat_names[f"Stat_{idx}"] = (
                    f"{idx}: Cross Scale ({scale}->{scale+1}) ({o1} × {o2})"
                )
                idx += 1

    # -------------------------
    # 1159–1254: Phase-doubled
    # -------------------------
    parts = ["Real→Real", "Real→Imag"]
    for scale in range(1, 4):
        for o1 in orientations:
            for o2 in orientations:
                for part in parts:
                    stat_names[f"Stat_{idx}"] = (
                        f"{idx}: Phase-Doubled ({scale}->{scale+1}) ({o1} × {o2}) {part}"
                    )
                    idx += 1

    return stat_names

stat_name_map = build_stat_names()

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([

    dcc.Store(id='global-stats', data=[]),
    dcc.Store(id='pcp-local', data=[]),
    dcc.Store(id='scatter-local', data=[]),

    html.H1("PS Statistics Explorer", style={'textAlign': 'center'}),

    html.Div([

        # LEFT COLUMN (PCP)
        html.Div([

            html.H3("Parallel Coordinates"),

            dcc.Graph(id='pcp-elbow'),

            dcc.Dropdown(
                id='pcp-dropdown',
                options=[
                        {'label': stat_name_map[f"Stat_{i}"], 'value': f"Stat_{i}"}
                        for i in range(1, 1255)
                ],
                multi=True,
                placeholder="Select PCP stats...",
                style={'width': '90%'}
            ),

            dcc.Graph(id='pcp-trend')

        ], style={'width': '48%', 'display': 'inline-block'}),

        # RIGHT COLUMN (SCATTER)
        html.Div([

            html.H3("Scatterplots"),

            dcc.Graph(id='scatter-elbow'),

            dcc.Dropdown(
                id='scatter-dropdown',
                options=[
                    {'label': stat_name_map[f"Stat_{i}"], 'value': f"Stat_{i}"}
                    for i in range(1, 1255)
                ],
                multi=True,
                placeholder="Select Scatter stats...",
                style={'width': '90%'}
            ),

            dcc.Graph(id='scatter-trend')

        ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'})

    ]),

    html.Hr(),

    html.Div([

        html.H2("Correlation Analysis", style={'textAlign': 'center'}),

        html.Div([

            # PCP HEATMAP
            html.Div([
                html.H3("PCP Correlation Heatmap"),
                dcc.Graph(id='pcp-heatmap')
            ], style={'width': '48%', 'display': 'inline-block'}),

            # SCATTER HEATMAP
            html.Div([
                html.H3("Scatter Correlation Heatmap"),
                dcc.Graph(id='scatter-heatmap')
            ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'})

        ])

    ])
])

# ==============================
# ELBOW PLOTS
# ==============================

@app.callback(Output('pcp-elbow', 'figure'), Input('pcp-elbow', 'id'))
def pcp_elbow(_):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_pcp_sorted['Plot_Rank'],
        y=df_pcp_sorted['Normalized_Importance'],
        mode='lines+markers',
        customdata=list(zip(
            df_pcp_sorted['Feature Index'],
            df_pcp_sorted['Mean_Absolute_Importance'],
            df_pcp_sorted['True_Stable_Rank']
        )),
        name="PCP",
        hovertemplate=(
            "<b>Rank on Curve:</b> %{x}<br>"
            "<b>% of Total Weight:</b> %{y:.4f}%<br>"
            "<b>Original Feature Index:</b> %{customdata[0]}<br>"
            "<b>Absolute Mean Importance:</b> %{customdata[1]}<br>"
            "<b>True Stable Rank:</b> %{customdata[2]}<br>"
            "<extra></extra>"
        )
    ))

    fig.update_layout(title="PCP Elbow Plot")
    return fig


@app.callback(Output('scatter-elbow', 'figure'), Input('scatter-elbow', 'id'))
def scatter_elbow(_):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_scatter_sorted['Plot_Rank'],
        y=df_scatter_sorted['Normalized_Importance'],
        mode='lines+markers',
        customdata=list(zip(
            df_scatter_sorted['Feature Index'],
            df_scatter_sorted['Mean_Absolute_Importance'],
            df_scatter_sorted['True_Stable_Rank']
        )),
        name="Scatter",
        hovertemplate=(
            "<b>Rank on Curve:</b> %{x}<br>"
            "<b>% of Total Weight:</b> %{y:.4f}%<br>"
            "<b>Original Feature Index:</b> %{customdata[0]}<br>"
            "<b>Absolute Mean Importance:</b> %{customdata[1]}<br>"
            "<b>True Stable Rank:</b> %{customdata[2]}<br>"
            "<extra></extra>"
        )
    ))

    fig.update_layout(title="Scatter Elbow Plot")
    return fig

# ==============================
# GLOBAL (ELBOW CLICK)
# ==============================

@app.callback(
    Output('global-stats', 'data'),
    Input('pcp-elbow', 'clickData'),
    Input('scatter-elbow', 'clickData'),
    State('global-stats', 'data'),
    prevent_initial_call=True
)
def update_global(pcp_click, scatter_click, current):

    ctx = callback_context
    if not ctx.triggered:
        return current

    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    click = pcp_click if trigger == 'pcp-elbow' else scatter_click

    if not click or 'points' not in click:
        return current

    point = click['points'][0]

    # 🔒 SAFETY CHECK
    if 'customdata' not in point:
        print("⚠️ Missing customdata:", point)
        return current

    stat_idx = int(point['customdata'][0])
    stat_name = f"Stat_{stat_idx}"

    current = current or []

    if stat_name in current:
        return [s for s in current if s != stat_name]
    else:
        return current + [stat_name]

# ==============================
# LOCAL DROPDOWNS
# ==============================

@app.callback(
    Output('pcp-local', 'data'),
    Input('pcp-dropdown', 'value')
)
def update_pcp_local(vals):
    return vals if vals else []


@app.callback(
    Output('scatter-local', 'data'),
    Input('scatter-dropdown', 'value')
)
def update_scatter_local(vals):
    return vals if vals else []

# ==============================
# SYNC GLOBAL → DROPDOWNS
# ==============================

@app.callback(
    Output('pcp-dropdown', 'value'),
    Input('global-stats', 'data'),
    State('pcp-local', 'data'),
    prevent_initial_call=True
)
def sync_pcp_dropdown(global_stats, local):
    return list(set((local or []) + (global_stats or [])))


@app.callback(
    Output('scatter-dropdown', 'value'),
    Input('global-stats', 'data'),
    State('scatter-local', 'data'),
    prevent_initial_call=True
)
def sync_scatter_dropdown(global_stats, local):
    return list(set((local or []) + (global_stats or [])))

# ==============================
# TREND PLOTS
# ==============================

def combine(global_stats, local_stats):
    return list(set((global_stats or []) + (local_stats or [])))


def build_trend_figure(df, selected_stats, title):
    fig = go.Figure()

    for stat in selected_stats:
        if stat in df.columns:
            y = df[stat]
            x = df.index

            valid = ~np.isnan(y)

            fig.add_trace(go.Scatter(
                x=x[valid],
                y=y[valid],
                mode='markers',
                name=stat_name_map.get(stat, stat),
                customdata=[stat] * len(x[valid])
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Pearson Correlation",
        yaxis_title="Standardized Value",
        template="plotly_white"
    )

    return fig


@app.callback(
    Output('pcp-trend', 'figure'),
    Input('global-stats', 'data'),
    Input('pcp-local', 'data')
)
def update_pcp(global_stats, local_stats):
    selected = combine(global_stats, local_stats)
    return build_trend_figure(df_pcp_stats, selected, "PCP Trends")


@app.callback(
    Output('scatter-trend', 'figure'),
    Input('global-stats', 'data'),
    Input('scatter-local', 'data')
)
def update_scatter(global_stats, local_stats):
    selected = combine(global_stats, local_stats)
    return build_trend_figure(df_scatter_stats, selected, "Scatter Trends")

#==============================
# HEATMAPS
#=============================

@app.callback(
    Output('pcp-heatmap', 'figure'),
    Input('global-stats', 'data'),
    Input('pcp-local', 'data')
)
def update_pcp_heatmap(global_stats, local_stats):

    selected = list(set((global_stats or []) + (local_stats or [])))

    if len(selected) < 2:
        fig = go.Figure()
        fig.update_layout(title="Select at least 2 stats (PCP)")
        return fig

    valid_stats = [s for s in selected if s in df_pcp_stats.columns]

    if len(valid_stats) < 2:
        return go.Figure()

    df_selected = df_pcp_stats[valid_stats]

    corr = df_selected.corr()
    labels = [stat_name_map.get(s, s) for s in corr.columns]

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=labels,
        y=labels,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        hovertemplate=(
            "Stat X: %{x}<br>"
            "Stat Y: %{y}<br>"
            "Corr: %{z:.3f}<extra></extra>"
        )
    ))

    fig.update_layout(title="PCP Feature Correlation")

    return fig


@app.callback(
    Output('scatter-heatmap', 'figure'),
    Input('global-stats', 'data'),
    Input('scatter-local', 'data')
)
def update_scatter_heatmap(global_stats, local_stats):

    selected = list(set((global_stats or []) + (local_stats or [])))

    if len(selected) < 2:
        fig = go.Figure()
        fig.update_layout(title="Select at least 2 stats (Scatter)")
        return fig

    valid_stats = [s for s in selected if s in df_scatter_stats.columns]

    if len(valid_stats) < 2:
        return go.Figure()

    df_selected = df_scatter_stats[valid_stats]

    corr = df_selected.corr()
    labels = [stat_name_map.get(s, s) for s in corr.columns]

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=labels,
        y=labels,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        hovertemplate=(
            "Stat X: %{x}<br>"
            "Stat Y: %{y}<br>"
            "Corr: %{z:.3f}<extra></extra>"
        )
    ))

    fig.update_layout(title="Scatter Feature Correlation")

    return fig

# ==============================
# RUN
# ==============================

if __name__ == '__main__':
    app.run_server(debug=False)