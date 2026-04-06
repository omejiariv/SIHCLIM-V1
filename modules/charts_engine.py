# modules/charts_engine.py

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ==============================================================================
# 1. GRÁFICOS DE SERIES DE TIEMPO Y TENDENCIAS
# ==============================================================================

def plot_serie_anual(df, col_anio, col_valor, col_estacion):
    """Genera el gráfico de líneas de precipitación anual."""
    fig = px.line(
        df, x=col_anio, y=col_valor, color=col_estacion, markers=True,
        labels={col_valor: "Lluvia (mm)", col_anio: "Año"}
    )
    fig.update_layout(hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20))
    return fig

def plot_ranking_multianual(df, col_estacion, col_valor, sort_opt):
    """Genera el gráfico de barras del ranking de precipitación media."""
    df_plot = df.copy()
    if sort_opt == "Mayor a Menor": 
        df_plot = df_plot.sort_values(col_valor, ascending=False)
    elif sort_opt == "Menor a Mayor": 
        df_plot = df_plot.sort_values(col_valor, ascending=True)
    else: 
        df_plot = df_plot.sort_values(col_estacion)

    fig = px.bar(
        df_plot, x=col_estacion, y=col_valor, color=col_valor, 
        color_continuous_scale='Blues', text_auto=".0f"
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig

def plot_serie_mensual(df, show_markers, show_regional):
    """Genera la serie histórica mensual con opción de promedio regional."""
    fig = px.line(
        df, x='fecha', y='valor', color='id_estacion',
        markers=show_markers, title="Precipitación Mensual"
    )
    if show_regional:
        reg_mean = df.groupby('fecha')['valor'].mean().reset_index()
        fig.add_trace(go.Scatter(
            x=reg_mean['fecha'], y=reg_mean['valor'], mode="lines",
            name="PROMEDIO REGIONAL", line=dict(color="black", width=3, dash="dash")
        ))
    fig.update_layout(hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20))
    return fig

# ==============================================================================
# 2. GRÁFICOS DE CICLO ANUAL Y ESTACIONALIDAD
# ==============================================================================

def plot_ciclo_anual(df, year_comp=None):
    """Genera el régimen de lluvias (ciclo promedio) con comparación opcional."""
    ciclo = df.groupby(['Mes', 'Nombre_Mes', 'id_estacion'])['valor'].mean().reset_index().sort_values('Mes')
    
    fig = px.line(
        ciclo, x='Nombre_Mes', y='valor', color='id_estacion', markers=True,
        title="Ciclo Anual Promedio"
    )
    
    if year_comp:
        df_year = df[df['Año'] == year_comp].sort_values('Mes')
        for est in df_year['id_estacion'].unique():
            df_y_st = df_year[df_year['id_estacion'] == est]
            fig.add_trace(go.Scatter(
                x=df_y_st['Nombre_Mes'], y=df_y_st['valor'],
                mode='lines+markers', name=f"{est} ({year_comp})",
                line=dict(dash='dot', width=2), marker=dict(symbol='x')
            ))
            
    fig.update_layout(hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20))
    return fig

def plot_spaghetti_estacional(df_st, col_anio, col_valor, hl_year=None):
    """Genera el Spaghetti Plot del ciclo anual comparativo."""
    years = sorted(df_st[col_anio].unique(), reverse=True)
    fig = go.Figure()
    
    # Trazas por año
    for yr in years:
        df_y = df_st[df_st[col_anio] == yr].sort_values("MES_NUM")
        color, width, opacity, show_leg = "rgba(200, 200, 200, 0.4)", 1, 0.5, False
        if hl_year and yr == hl_year:
            color, width, opacity, show_leg = "red", 4, 1.0, True
        
        fig.add_trace(go.Scatter(
            x=df_y["Nombre_Mes"], y=df_y[col_valor], mode="lines",
            name=str(yr), line=dict(color=color, width=width), 
            opacity=opacity, showlegend=show_leg
        ))
    
    # Promedio
    clim = df_st.groupby("MES_NUM")[col_valor].mean().sort_index()
    meses_mapa = {1:'Ene', 2:'Feb', 3:'Mar', 4:'Abr', 5:'May', 6:'Jun', 7:'Jul', 8:'Ago', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dic'}
    names_clim = [meses_mapa.get(m, str(m)) for m in clim.index]
    
    fig.add_trace(go.Scatter(
        x=names_clim, y=clim.values, mode="lines+markers",
        name="Promedio Histórico", line=dict(color="black", width=3, dash="dot")
    ))
    
    fig.update_xaxes(categoryorder='array', categoryarray=list(meses_mapa.values()), title="Mes")
    fig.update_layout(hovermode="x unified", margin=dict(l=20, r=20, t=20, b=20))
    return fig

def plot_cajas_estacional(df_st, col_valor):
    """Genera el Boxplot de variabilidad estacional."""
    fig = px.box(
        df_st, x="Nombre_Mes", y=col_valor, color="Nombre_Mes", points="all",
        category_orders={"Nombre_Mes": ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]}
    )
    fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
    return fig

# ==============================================================================
# 3. GRÁFICOS ESTADÍSTICOS
# ==============================================================================

def plot_distribucion_estadistica(df, col_x, col_y, chart_typ, sort_ord):
    """Genera Violín, Histograma o ECDF según lo seleccionado."""
    cat_orders = {}
    if sort_ord != "Alfabético":
        medians = df.groupby(col_x)[col_y].median()
        order_list = medians.sort_values(ascending=False).index.tolist()
        cat_orders = {col_x: order_list}

    if "Violín" in chart_typ:
        fig = px.violin(df, x=col_x, y=col_y, color=col_x, box=True, points="all", category_orders=cat_orders)
    elif "Histograma" in chart_typ:
        fig = px.histogram(df, x=col_y, color=col_x, marginal="box", barmode="overlay", category_orders=cat_orders)
    else:
        fig = px.ecdf(df, x=col_y, color=col_x)

    fig.update_layout(height=600, showlegend=(chart_typ != "Violín"), margin=dict(l=20, r=20, t=40, b=20))
    return fig
