# modules/climate_api.py

import pandas as pd
import requests
import streamlit as st

# ==============================================================================
# 🛡️ ESCUDOS DE IMPORTACIÓN (FALLBACK SHIELDS)
# ==============================================================================
@st.cache_data(show_spinner=False)
def fetch_iri_data(filename): return None

@st.cache_data(show_spinner=False)
def process_iri_plume(data_json): return None

@st.cache_data(show_spinner=False)
def process_iri_probabilities(data_json): return None

# ==============================================================================
# 🔌 LLAVE 1: CONEXIÓN DE PROBABILIDADES ENSO (DIRECTO A NOAA)
# ==============================================================================
@st.cache_data(show_spinner=False, ttl=43200) # Caché de 12 horas
def get_iri_enso_forecast():
    """Extrae la tabla de probabilidades trimestrales de la NOAA."""
    url_noaa = "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/probabilities.php"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(url_noaa, headers=headers, timeout=15)
        response.raise_for_status()
        tablas = pd.read_html(response.text)
        
        df_probs = None
        for df in tablas:
            if "Neutral" in df.to_string() and ("Niño" in df.to_string() or "Nino" in df.to_string()):
                df_probs = df
                break
                
        if df_probs is not None:
            df_limpio = pd.DataFrame()
            df_limpio['Trimestre'] = df_probs.iloc[:, 0].astype(str).str.strip()
            df_limpio['La Niña'] = pd.to_numeric(df_probs.iloc[:, 1].astype(str).str.replace('%', ''), errors='coerce')
            df_limpio['Neutral'] = pd.to_numeric(df_probs.iloc[:, 2].astype(str).str.replace('%', ''), errors='coerce')
            df_limpio['El Niño'] = pd.to_numeric(df_probs.iloc[:, 3].astype(str).str.replace('%', ''), errors='coerce')
            df_limpio = df_limpio[df_limpio['Trimestre'].str.match(r'^[A-Z]{3}$', na=False)]
            
            if not df_limpio.empty:
                return df_limpio.reset_index(drop=True), {"fuente": "NOAA CPC Directo"}
    except Exception:
        pass 
        
    # Respaldo Interno
    df_rescate = pd.DataFrame([
        {"Trimestre": "AMJ", "La Niña": 0, "Neutral": 30, "El Niño": 70},
        {"Trimestre": "MJJ", "La Niña": 0, "Neutral": 12, "El Niño": 88},
        {"Trimestre": "JJA", "La Niña": 0, "Neutral": 10, "El Niño": 90},
        {"Trimestre": "JAS", "La Niña": 0, "Neutral": 8,  "El Niño": 92}
    ])
    return df_rescate, {"fuente": "Consenso Interno (Respaldo)"}

# ==============================================================================
# 🔌 LLAVE 2: CONEXIÓN DE ÍNDICE HISTÓRICO ONI (EN VIVO)
# ==============================================================================
@st.cache_data(show_spinner=False, ttl=43200) # Caché de 12 horas
def get_live_oni_data():
    """Descarga el registro histórico oficial del ONI en formato de texto plano y lo estructura."""
    url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
    try:
        # La NOAA publica esto como texto delimitado por espacios
        df_oni = pd.read_csv(url, delim_whitespace=True)
        if not {'SEAS', 'YR', 'ANOM'}.issubset(df_oni.columns): return None
        
        # Mapeamos los trimestres a un mes numérico para poder graficar la serie de tiempo
        mes_map = {
            'DJF': '01', 'JFM': '02', 'FMA': '03', 'MAM': '04',
            'AMJ': '05', 'MJJ': '06', 'JJA': '07', 'JAS': '08',
            'ASO': '09', 'SON': '10', 'OND': '11', 'NDJ': '12'
        }
        df_oni['mes'] = df_oni['SEAS'].map(mes_map)
        df_oni = df_oni.dropna(subset=['mes'])
        
        # Creamos una columna de fecha perfecta (Año-Mes-01)
        df_oni['fecha'] = pd.to_datetime(df_oni['YR'].astype(str) + '-' + df_oni['mes'] + '-01')
        
        df_live = pd.DataFrame({
            'fecha': df_oni['fecha'],
            'anomalia_oni': pd.to_numeric(df_oni['ANOM'], errors='coerce')
        })
        return df_live.dropna()
    except Exception as e:
        print(f"Error extrayendo ONI: {e}")
        return None
