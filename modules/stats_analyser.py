# modules/stats_analyser.py
import pandas as pd
import numpy as np
import pymannkendall as mk
from scipy import stats
from modules.config import Config

def get_safe_cols(df):
    """
    Cerebro de Compatibilidad: Detecta automáticamente columnas de fecha y valor.
    Basado en la lógica blindada de analysis.py.
    """
    if df is None or df.empty: return None, None
    
    # Candidatos para fecha
    c_date = next((c for c in ['fecha', Config.DATE_COL, 'ds', 'fecha_mes_año', 'Date', 'time'] if c in df.columns), None)
    
    # Candidatos para valor (Lluvia/Precipitación)
    c_val = next((c for c in ['valor', Config.PRECIPITATION_COL, 'precipitation', 'Pptn', 'lluvia', 'rain'] if c in df.columns), None)
    
    return c_date, c_val

def calcular_tendencia_mk_estacion(serie_tiempo):
    """
    Calcula Mann-Kendall con interpretación visual y significancia estadística.
    """
    try:
        serie_limpia = serie_tiempo.dropna()
        if len(serie_limpia) < 10:
            return "Insuficiente", 1.0, 0.0, "➖ Estable", "Datos Insuficientes"
        
        res = mk.original_test(serie_limpia)
        
        # 1. Icono Visual
        trend_icon = "➖ Estable"
        if res.trend == "increasing":
            trend_icon = "📈 (Aumento)"
        elif res.trend == "decreasing":
            trend_icon = "📉 (Disminución)"
            
        # 2. Interpretación de Significancia (Confianza al 95%)
        is_significant = res.p < 0.05
        sig_text = "✅ Significativo (Confianza > 95%)" if is_significant else "⚠️ No Significativo"
            
        return res.trend, res.p, res.slope, trend_icon, sig_text
    except Exception as e:
        return f"Error: {e}", 1.0, 0.0, "❌ Error", "N/A"
        
def calcular_anomalias_climatologicas(df_mensual, df_historico, inicio_base=1991, fin_base=2020):
    """
    Calcula anomalías respecto a un periodo de referencia (Baseline).
    Basado en calculate_climatological_anomalies de analysis.py.
    """
    df_work = df_mensual.copy()
    df_ref = df_historico.copy()

    # Detección de columnas
    c_date_ref, c_val_ref = get_safe_cols(df_ref)
    c_date_work, c_val_work = get_safe_cols(df_work)
    
    c_name = next((c for c in ['nombre', Config.STATION_NAME_COL, 'nom_est'] if c in df_ref.columns), None)
    c_month = next((c for c in ['mes', Config.MONTH_COL, 'month'] if c in df_ref.columns), None)
    c_year = next((c for c in ['año', Config.YEAR_COL, 'year'] if c in df_ref.columns), None)

    if not all([c_val_ref, c_name, c_month, c_year]):
        return df_work

    # Filtrar periodo base (OMM: 1991-2020)
    baseline = df_ref[(df_ref[c_year] >= inicio_base) & (df_ref[c_year] <= fin_base)]

    # Climatología: Promedio multianual por mes y estación
    climatology = (
        baseline.groupby([c_name, c_month])[c_val_ref]
        .mean()
        .reset_index()
        .rename(columns={c_val_ref: "media_historica"})
    )

    # Unión y cálculo de anomalía
    df_res = pd.merge(df_work, climatology, on=[c_name, c_month], how="left")
    df_res["anomalia_abs"] = df_res[c_val_work] - df_res["media_historica"]
    df_res["anomalia_pct"] = (df_res["anomalia_abs"] / df_res["media_historica"]) * 100
    
    return df_res

def obtener_resumen_extremos(df_serie, p_bajo=10, p_alto=90):
    """
    Identifica umbrales de percentiles para eventos extremos.
    """
    _, c_val = get_safe_cols(df_serie)
    if not c_val: return None, None, None
    
    datos = df_serie[c_val].dropna()
    if datos.empty: return None, None, None
    
    u_bajo = np.percentile(datos, p_bajo)
    u_alto = np.percentile(datos, p_alto)
    promedio = datos.mean()
    
    return u_bajo, u_alto, promedio
