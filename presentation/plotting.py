from __future__ import annotations
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def lineprofile(df: pd.DataFrame, columns: list[str], title: str, ytitle: str) -> go.Figure:
    fig = go.Figure()
    palette = ['#0b7285', '#2f9e44', '#4dabf7', '#a12c7b', '#da7101']
    for i, col in enumerate(columns):
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df[col], mode='lines', name=col, line=dict(color=palette[i % len(palette)], width=2)))
    fig.update_layout(title=title, yaxis_title=ytitle, paper_bgcolor='#f4f8f8', plot_bgcolor='white', margin=dict(l=20, r=20, t=50, b=20), legend=dict(orientation='h'))
    return fig

def line_profile(df: pd.DataFrame, columns: list[str], title: str, ytitle: str) -> go.Figure:
    return lineprofile(df, columns, title, ytitle)

def heatmap(df: pd.DataFrame, z_col: str = 'lcomeoh_usd_per_t_meoh') -> go.Figure:
    needed = {'electrolyzer_power_mw', 'storage_kg_h2', z_col}
    if not needed.issubset(set(df.columns)):
        return go.Figure()
    pivot = df.pivot_table(index='storage_kg_h2', columns='electrolyzer_power_mw', values=z_col, aggfunc='mean')
    fig = px.imshow(pivot, aspect='auto', color_continuous_scale='Viridis', labels=dict(x='Electrolyzer power [MW]', y='Storage [kg H2]', color=z_col))
    fig.update_layout(title=f'Heatmap of {z_col}', paper_bgcolor='#f4f8f8', plot_bgcolor='white', margin=dict(l=20, r=20, t=50, b=20))
    return fig

def tornado(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()
    metric = 'npv_usd' if 'npv_usd' in df.columns else df.columns[-1]
    fig = px.bar(df.sort_values(metric), x=metric, y='parameter', orientation='h', color_discrete_sequence=['#0b7285'])
    fig.update_layout(title='Sensitivity tornado view', paper_bgcolor='#f4f8f8', plot_bgcolor='white', margin=dict(l=20, r=20, t=50, b=20))
    return fig
