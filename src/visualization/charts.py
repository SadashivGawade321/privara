"""Visualization utilities for Privara.

Generates Plotly interactive charts styled with TECHNOVA 2026 dark-navy theme:
cyan (#06b6d4), blue (#3b82f6), teal (#10b981), amber (#f59e0b), red (#ef4444).
"""

from typing import List, Dict, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Theme palette matching TECHNOVA slides
THEME = {
    "bg_dark": "#0b1329",
    "card_bg": "#111c44",
    "border": "#1e293b",
    "text": "#e2e8f0",
    "muted": "#94a3b8",
    "cyan": "#06b6d4",
    "blue": "#3b82f6",
    "teal": "#10b981",
    "amber": "#f59e0b",
    "red": "#ef4444",
    "purple": "#8b5cf6"
}


def _apply_dark_theme(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    """Apply unified dark-navy styling to Plotly figure."""
    fig.update_layout(
        title={
            "text": title,
            "y": 0.95,
            "x": 0.05,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 14, "color": THEME["text"], "family": "Inter, sans-serif"}
        },
        paper_bgcolor=THEME["bg_dark"],
        plot_bgcolor=THEME["bg_dark"],
        font={"color": THEME["text"], "family": "Inter, sans-serif"},
        height=height,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font={"size": 11, "color": THEME["muted"]},
            bgcolor="rgba(0,0,0,0)"
        ),
        xaxis=dict(
            gridcolor=THEME["border"],
            zerolinecolor=THEME["border"],
            tickfont={"color": THEME["muted"]}
        ),
        yaxis=dict(
            gridcolor=THEME["border"],
            zerolinecolor=THEME["border"],
            tickfont={"color": THEME["muted"]}
        )
    )
    return fig


def plot_accuracy_curves(
    rounds: List[int],
    test_accs: List[float],
    train_accs: Optional[List[float]] = None
) -> go.Figure:
    """Plot global test and training accuracy progression across rounds."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rounds,
        y=test_accs,
        mode="lines+markers",
        name="Global Test Accuracy",
        line=dict(color=THEME["cyan"], width=3),
        marker=dict(size=6, color=THEME["cyan"])
    ))
    if train_accs:
        fig.add_trace(go.Scatter(
            x=rounds,
            y=train_accs,
            mode="lines",
            name="Client Avg Train Acc",
            line=dict(color=THEME["blue"], width=2, dash="dash")
        ))
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
    return _apply_dark_theme(fig, "Training Convergence & Test Accuracy")


def plot_privacy_utility_tradeoff(data: List[Dict[str, Any]]) -> go.Figure:
    """Plot Accuracy vs Achieved Epsilon curve (Experiment A)."""
    # Filter out infinity for numeric plotting
    filtered = [d for d in data if d.get("achieved_epsilon") not in ["inf", float("inf"), 999.0]]
    filtered = sorted(filtered, key=lambda x: x["achieved_epsilon"])
    
    epsilons = [d["achieved_epsilon"] for d in filtered]
    accuracies = [d["test_accuracy"] for d in filtered]
    names = [d.get("config_name", f"ε={e}") for d, e in zip(filtered, epsilons)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epsilons,
        y=accuracies,
        mode="lines+markers+text",
        text=names,
        textposition="top center",
        name="Test Accuracy",
        line=dict(color=THEME["blue"], width=3),
        marker=dict(size=8, color=THEME["cyan"])
    ))
    fig.update_xaxes(title_text="Privacy Budget (ε) — Lower is Stronger Privacy")
    fig.update_yaxes(title_text="Test Accuracy", range=[0.4, 1.0], tickformat=".0%")
    return _apply_dark_theme(fig, "Privacy – Utility Trade-off (Test Accuracy vs. ε)")


def plot_communication_bytes_per_round(
    rounds: List[int],
    uncompressed_mb: List[float],
    compressed_mb: List[float]
) -> go.Figure:
    """Grouped bar chart comparing Uncompressed vs Compressed MB per round."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=rounds,
        y=uncompressed_mb,
        name="Uncompressed",
        marker_color=THEME["blue"]
    ))
    fig.add_trace(go.Bar(
        x=rounds,
        y=compressed_mb,
        name="Compressed (Top-K)",
        marker_color=THEME["teal"]
    ))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Bytes / MB per Round")
    fig.update_xaxes(title_text="Federated Round")
    return _apply_dark_theme(fig, "Communication Cost Analysis (MB per Round)")


def plot_communication_breakdown(
    rounds: List[int],
    model_update_mb: List[float],
    metadata_mb: List[float],
    overhead_mb: List[float]
) -> go.Figure:
    """Stacked bar chart showing payload breakdown: model update, metadata, overhead."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=rounds,
        y=model_update_mb,
        name="Model Updates",
        marker_color=THEME["blue"]
    ))
    fig.add_trace(go.Bar(
        x=rounds,
        y=metadata_mb,
        name="Metadata",
        marker_color=THEME["purple"]
    ))
    fig.add_trace(go.Bar(
        x=rounds,
        y=overhead_mb,
        name="Overhead",
        marker_color=THEME["amber"]
    ))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title_text="Data (MB)")
    fig.update_xaxes(title_text="Round")
    return _apply_dark_theme(fig, "Communication Cost Breakdown")


def plot_client_data_distribution(client_stats: Dict[str, Dict[str, Any]]) -> go.Figure:
    """Stacked bar chart showing class label counts across clients C1..C10."""
    records = []
    for cid, stats in client_stats.items():
        for cls_label, count in stats["class_distribution"].items():
            records.append({
                "Client": cid,
                "Class": f"Class {cls_label}",
                "Samples": count
            })
    df = pd.DataFrame(records)
    if df.empty:
        return go.Figure()
    
    fig = px.bar(
        df,
        x="Client",
        y="Samples",
        color="Class",
        title="Client Sample & Label Heterogeneity (Non-IID Skew)",
        color_discrete_sequence=px.colors.qualitative.Dark24
    )
    return _apply_dark_theme(fig, "Client Data Distribution across Classes", height=320)
