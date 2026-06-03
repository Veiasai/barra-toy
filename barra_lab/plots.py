from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def heatmap(df: pd.DataFrame, title: str, color_scale: str = "RdBu") -> go.Figure:
    fig = px.imshow(
        df,
        color_continuous_scale=color_scale,
        aspect="auto",
        title=title,
    )
    fig.update_layout(height=520, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def bar(series: pd.Series, title: str, y_title: str = "value") -> go.Figure:
    fig = px.bar(
        series.reset_index(),
        x=series.index.name or "index",
        y=series.name or y_title,
        title=title,
    )
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=40), xaxis_tickangle=-35)
    return fig


def line(df: pd.DataFrame, title: str) -> go.Figure:
    fig = px.line(df, title=title)
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=50, b=40))
    return fig

