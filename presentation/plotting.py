from __future__ import annotations
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PALETTE = {
    "teal": "#0b7285",
    "green": "#2f9e44",
    "cyan": "#4dabf7",
    "slate": "#6b7f86",
    "dark": "#18333a",
    "bg": "#f4f8f8"
}

def line_profile(df: pd.DataFrame, columns: list[str], title: str, y_title: str):
    fig = go.Figure()
    colors = [PALETTE["teal"], PALETTE["green"], PALETTE["cyan"], PALETTE["slate"], "#8ca6ad"]
    for idx, col in enumerate(columns):
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df["timestamp"], y=df[col], mode="lines", name=col, line=dict(color=colors[idx % len(colors)], width=2)))
    fig.update_layout(title=title, paper_bgcolor=PALETTE["bg"], plot_bgcolor="white", yaxis_title=y_title, legend_orientation="h", margin=dict(l=20, r=20, t=50, b=20))
    return fig

def heatmap(results: pd.DataFrame, z_col: str = "lcomeoh_usd_per_t_meoh"):
    pivot = results.pivot_table(index="storage_kg_h2", columns="electrolyzer_power_mw", values=z_col, aggfunc="mean")
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale=["#d9f0ef", "#8dd3d2", "#0b7285"], labels=dict(color=z_col))
    fig.update_layout(title=f"Design heatmap — {z_col}", paper_bgcolor=PALETTE["bg"], plot_bgcolor="white")
    fig.update_xaxes(title="Electrolyzer nominal power [MW]")
    fig.update_yaxes(title="Usable H2 storage [kg H2]")
    return fig

def tornado(sens: pd.DataFrame):
    data = sens.groupby("parameter", as_index=False)["lcomeoh_usd_per_t_meoh"].agg(lambda s: s.max() - s.min())
    data = data.rename(columns={"lcomeoh_usd_per_t_meoh": "impact_usd_per_t_meoh"}).sort_values("impact_usd_per_t_meoh")
    fig = px.bar(data, x="impact_usd_per_t_meoh", y="parameter", orientation="h", color_discrete_sequence=[PALETTE["green"]])
    fig.update_layout(title="Sensitivity tornado — LCOMeOH impact", paper_bgcolor=PALETTE["bg"], plot_bgcolor="white", xaxis_title="Impact on LCOMeOH [USD/t MeOH]", yaxis_title="Parameter")
    return fig

