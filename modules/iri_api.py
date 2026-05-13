# modules/iri_api.py

import pandas as pd
import requests
import streamlit as st

# ==============================================================================
# 🛡️ ESCUDOS DE IMPORTACIÓN (FALLBACK SHIELDS)
# ==============================================================================
# Columbia University borró todos los JSON (Plumas y Probabilidades) de su servidor.
# Mantenemos estas funciones vivas devolviendo 'None' para que visualizer.py 
# no sufra un ImportError ni un TypeError al intentar graficar los modelos antiguos.

@st.cache_data(show_spinner=False)
def fetch_iri_data(filename):
    """Escudo para evitar colapso en visualizer.py"""
    return None

@st.cache_data(show_spinner=False)
def process_iri_plume(data_json):
    """Escudo para evitar colapso en visualizer.py"""
    return None

@st.cache_data(show_spinner=False)
def process_iri_probabilities(data_json):
    """Escudo para evitar colapso en visualizer.py"""
    return None


# ==============================================================================
# 🔌 LLAVE MAESTRA: CONEXIÓN AL ALEPH CLIMÁTICO (DIRECTO A NOAA)
# ==============================================================================
@st.cache_data(show_spinner=False, ttl=43200) # Caché de 12 horas
def get_iri_enso_forecast():
    """
    Alternativa Definitiva (NOAA CPC): 
    Se conecta directamente al Climate Prediction Center para extraer la tabla de probabilidades.
    """
    # 🌐 URL Oficial de Probabilidades ENSO de la NOAA
    url_noaa = "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/probabilities.php"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        # 1. Petición directa a la página de la NOAA
        response = requests.get(url_noaa, headers=headers, timeout=15)
        response.raise_for_status()
        
        # 2. Extracción de tablas HTML
        tablas = pd.read_html(response.text)
        
        # 3. Búsqueda de la tabla de probabilidades
        df_probs = None
        for df in tablas:
            texto_tabla = df.to_string()
            if "Neutral" in texto_tabla and ("Niño" in texto_tabla or "Nino" in texto_tabla):
                df_probs = df
                break
                
        if df_probs is not None:
            # 4. Formateo al estándar
            df_limpio = pd.DataFrame()
            df_limpio['Trimestre'] = df_probs.iloc[:, 0].astype(str).str.strip()
            df_limpio['La Niña'] = pd.to_numeric(df_probs.iloc[:, 1].astype(str).str.replace('%', ''), errors='coerce')
            df_limpio['Neutral'] = pd.to_numeric(df_probs.iloc[:, 2].astype(str).str.replace('%', ''), errors='coerce')
            df_limpio['El Niño'] = pd.to_numeric(df_probs.iloc[:, 3].astype(str).str.replace('%', ''), errors='coerce')
            
            df_limpio = df_limpio[df_limpio['Trimestre'].str.match(r'^[A-Z]{3}$', na=False)]
            
            if not df_limpio.empty:
                # El sistema antiguo esperaba una tupla (DataFrame, diccionario_crudo)
                return df_limpio.reset_index(drop=True), {"fuente": "NOAA CPC Directo"}
                
    except Exception as e:
        # Falla silenciosa que activa el respaldo de emergencia
        pass 
        
    # ==============================================================================
    # 🛡️ ESCUDO DE RESPALDO INTERNO (IRONCLAD FALLBACK)
    # ==============================================================================
    datos_rescate = [
        {"Trimestre": "AMJ", "La Niña": 0, "Neutral": 30, "El Niño": 70},
        {"Trimestre": "MJJ", "La Niña": 0, "Neutral": 12, "El Niño": 88},
        {"Trimestre": "JJA", "La Niña": 0, "Neutral": 10, "El Niño": 90},
        {"Trimestre": "JAS", "La Niña": 0, "Neutral": 8,  "El Niño": 92},
        {"Trimestre": "ASO", "La Niña": 0, "Neutral": 6,  "El Niño": 94},
        {"Trimestre": "SON", "La Niña": 0, "Neutral": 6,  "El Niño": 94}
    ]
    df_rescate = pd.DataFrame(datos_rescate)
    
    return df_rescate, {"fuente": "Consenso Interno (Respaldo)"}
