import json
import os
import sys
from functools import lru_cache
from typing import Dict, List

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html, dash_table

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.llm.pipeline import ACTION_TO_ID, RTFSASPipeline


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

DEFAULT_CHECKPOINT = os.path.join(BASE_DIR, "checkpoints", "gnn_weighted.pt")
DEFAULT_Q_SCORER = os.path.join(BASE_DIR, "checkpoints", "q_scorer_best.pt")
DEFAULT_INDEX_DIR = os.path.join(BASE_DIR, "index")
DEFAULT_METRICS_JSON = os.path.join(BASE_DIR, "report", "ablation_metrics.json")


def load_metrics(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def build_pitch_figure(x: float, y: float) -> go.Figure:
    fig = go.Figure()

    # Pitch outline (StatsBomb 120x80 scale)
    fig.add_trace(
        go.Scatter(
            x=[0, 120, 120, 0, 0],
            y=[0, 0, 80, 80, 0],
            mode="lines",
            line={"color": "#D1FAE5", "width": 2},
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Midline
    fig.add_trace(
        go.Scatter(
            x=[60, 60],
            y=[0, 80],
            mode="lines",
            line={"color": "#A7F3D0", "width": 1.5, "dash": "dot"},
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Ball
    fig.add_trace(
        go.Scatter(
            x=[x],
            y=[y],
            mode="markers+text",
            marker={"size": 14, "color": "#F59E0B", "line": {"width": 1, "color": "#111827"}},
            text=["Ball"],
            textposition="top center",
            showlegend=False,
            hovertemplate="x=%{x:.1f}, y=%{y:.1f}<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1220",
        plot_bgcolor="#0B1220",
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        xaxis={"range": [0, 120], "title": "Pitch X", "showgrid": False, "zeroline": False},
        yaxis={"range": [0, 80], "title": "Pitch Y", "showgrid": False, "zeroline": False, "scaleanchor": "x"},
    )
    return fig


def empty_retrieval_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["rank", "match_id", "minute", "event_type", "next_event", "similarity"]
    )


@lru_cache(maxsize=16)
def get_pipeline(
    checkpoint_path: str,
    index_dir: str,
    hidden_dim: int,
    embed_dim: int,
    num_classes: int,
    q_scorer_path: str,
) -> RTFSASPipeline:
    q_ckpt = q_scorer_path.strip() if q_scorer_path else ""
    return RTFSASPipeline(
        gnn_checkpoint_path=checkpoint_path,
        index_dir=index_dir,
        retriever_k=5,
        gnn_hidden_dim=hidden_dim,
        gnn_embed_dim=embed_dim,
        gnn_num_classes=num_classes,
        q_scorer_checkpoint_path=q_ckpt if q_ckpt and os.path.isfile(q_ckpt) else None,
    )


def metrics_cards(metrics: Dict) -> List[dbc.Col]:
    retrieval = metrics.get("retrieval", {})
    advice = metrics.get("advice_quality", {})
    rows = [
        ("Top-1 Match Rate", retrieval.get("top1_next_event_match_rate", "NA")),
        ("Top-k Contains Rate", retrieval.get("topk_next_event_contains_rate", "NA")),
        ("Avg Top-1 Similarity", retrieval.get("avg_top1_similarity", "NA")),
        ("Full Specificity", advice.get("full_avg_specificity_score", "NA")),
    ]
    out = []
    for title, value in rows:
        val = f"{value:.3f}" if isinstance(value, float) else str(value)
        out.append(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div(title, className="metric-title"),
                            html.H4(val, className="metric-value"),
                        ]
                    ),
                    className="metric-card",
                ),
                md=3,
            )
        )
    return out


app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="RT-FSAS Dashboard",
)
server = app.server

app.layout = dbc.Container(
    fluid=True,
    className="app-shell",
    children=[
        html.Div(
            [
                html.H2("RT-FSAS Tactical Simulation Dashboard", className="app-title"),
                html.P(
                    "Simulate live events, retrieve historical analogs, and generate coaching advice.",
                    className="app-subtitle",
                ),
            ],
            className="header-wrap",
        ),
        dbc.Row(id="metrics-row", children=metrics_cards(load_metrics(DEFAULT_METRICS_JSON)), className="g-3"),
        dbc.Row(
            className="g-3 mt-2",
            children=[
                dbc.Col(
                    md=3,
                    children=dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Simulation Controls"),
                                dbc.Label("Team Name"),
                                dbc.Input(id="team-name", value="Barcelona"),
                                dbc.Label("Scoreline", className="mt-2"),
                                dbc.Input(id="scoreline", value="1-1"),
                                dbc.Label("Minute", className="mt-2"),
                                dbc.Input(id="minute", type="number", min=1, max=120, value=67),
                                dbc.Label("Event Type", className="mt-2"),
                                dcc.Dropdown(
                                    id="event-type",
                                    options=[{"label": k, "value": k} for k in ACTION_TO_ID.keys()],
                                    value="Pass",
                                    clearable=False,
                                ),
                                dbc.Label("X", className="mt-2"),
                                dcc.Slider(id="event-x", min=0, max=120, step=1, value=82),
                                dbc.Label("Y", className="mt-2"),
                                dcc.Slider(id="event-y", min=0, max=80, step=1, value=25),
                                html.Hr(),
                                dbc.Label("Checkpoint Path"),
                                dbc.Input(id="checkpoint-path", value=DEFAULT_CHECKPOINT),
                                dbc.Label("Index Dir", className="mt-2"),
                                dbc.Input(id="index-dir", value=DEFAULT_INDEX_DIR),
                                dbc.Label("Hidden Dim", className="mt-2"),
                                dbc.Input(id="hidden-dim", type="number", value=96),
                                dbc.Label("Q-Scorer checkpoint (optional)", className="mt-2"),
                                dbc.Input(
                                    id="q-scorer-path",
                                    value=DEFAULT_Q_SCORER,
                                    placeholder="checkpoints/q_scorer_best.pt",
                                ),
                                dbc.Button("Run Simulation", id="run-btn", className="mt-3 w-100", color="success"),
                                html.Div(id="run-status", className="status-text mt-2"),
                            ]
                        ),
                        className="panel-card",
                    ),
                ),
                dbc.Col(
                    md=5,
                    children=dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Live Pitch Snapshot"),
                                dcc.Graph(id="pitch-graph", figure=build_pitch_figure(82, 25), config={"displayModeBar": False}),
                                html.Hr(),
                                html.H5("Retrieved Similar Situations"),
                                dash_table.DataTable(
                                    id="retrieval-table",
                                    columns=[{"name": c, "id": c} for c in empty_retrieval_table().columns],
                                    data=[],
                                    style_table={"overflowX": "auto"},
                                    style_cell={"backgroundColor": "#0F172A", "color": "#E5E7EB", "border": "1px solid #1F2937", "fontSize": 13},
                                    style_header={"backgroundColor": "#111827", "fontWeight": "bold"},
                                    page_size=5,
                                ),
                            ]
                        ),
                        className="panel-card",
                    ),
                ),
                dbc.Col(
                    md=4,
                    children=dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Simulation Output"),
                                html.Div(id="qdelta-box", className="kpi-box", children="q_delta: --"),
                                html.H6("Coaching Advice", className="mt-3"),
                                html.Pre(id="advice-text", className="advice-box", children="Run simulation to generate advice."),
                            ]
                        ),
                        className="panel-card",
                    ),
                ),
            ],
        ),
    ],
)


@app.callback(
    Output("pitch-graph", "figure"),
    Input("event-x", "value"),
    Input("event-y", "value"),
)
def update_pitch(x: float, y: float) -> go.Figure:
    return build_pitch_figure(float(x or 60), float(y or 40))


@app.callback(
    Output("qdelta-box", "children"),
    Output("advice-text", "children"),
    Output("retrieval-table", "data"),
    Output("run-status", "children"),
    Input("run-btn", "n_clicks"),
    State("team-name", "value"),
    State("scoreline", "value"),
    State("minute", "value"),
    State("event-type", "value"),
    State("event-x", "value"),
    State("event-y", "value"),
    State("checkpoint-path", "value"),
    State("index-dir", "value"),
    State("hidden-dim", "value"),
    State("q-scorer-path", "value"),
    prevent_initial_call=True,
)
def run_simulation(
    _n_clicks: int,
    team_name: str,
    scoreline: str,
    minute: int,
    event_type: str,
    x: float,
    y: float,
    checkpoint_path: str,
    index_dir: str,
    hidden_dim: int,
    q_scorer_path: str,
):
    try:
        pipe = get_pipeline(
            checkpoint_path=checkpoint_path,
            index_dir=index_dir,
            hidden_dim=int(hidden_dim or 96),
            embed_dim=128,
            num_classes=11,
            q_scorer_path=q_scorer_path or "",
        )
        event = {
            "location": [float(x or 60), float(y or 40)],
            "minute": int(minute or 60),
            "type": {"name": event_type or "Pass"},
            "team": {"name": team_name or "Team A"},
            "possession_team": {"id": 1},
            "score": scoreline or "unknown",
            "match_id": 0,
        }
        result = pipe.process([event], current_minute=int(minute or 60))
        retrieved = result.get("retrieved", [])[:5]
        table_rows = [
            {
                "rank": r.get("rank"),
                "match_id": r.get("match_id"),
                "minute": r.get("minute"),
                "event_type": r.get("event_type"),
                "next_event": r.get("next_event"),
                "similarity": f"{float(r.get('similarity', 0.0)):.4f}",
            }
            for r in retrieved
        ]
        return (
            f"q_delta: {float(result.get('q_delta', 0.0)):.4f}",
            result.get("advice", "No advice generated."),
            table_rows,
            "Simulation complete.",
        )
    except Exception as exc:
        return "q_delta: --", f"Error: {exc}", [], "Simulation failed."


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)

