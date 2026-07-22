# modules/climate_api.py

import pandas as pd
import requests
import streamlit as st
from io import StringIO

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
    """Extrae la tabla de probabilidades trimestrales de la NOAA dinámicamente."""
    url_noaa = "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/probabilities.php"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(url_noaa, headers=headers, timeout=10) # Reducimos el timeout para evitar cuelgues
        response.raise_for_status()
        tablas = pd.read_html(StringIO(response.text))
        
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
            
            # 🚀 FIX: Eliminamos el filtro duro. 
            # La NOAA actualiza su tabla y elimina automáticamente los pasados. 
            # Tomaremos el primero disponible en selectors.py.
            
            if not df_limpio.empty:
                return df_limpio.reset_index(drop=True), {"fuente": "NOAA CPC Directo"}
    except Exception:
        pass 
        
    # Respaldo Interno Dinámico (Actualizado a JJA - Verano 2026)
    df_rescate = pd.DataFrame([
        {"Trimestre": "JJA", "La Niña": 0, "Neutral": 2, "El Niño": 98},
        {"Trimestre": "JAS", "La Niña": 0, "Neutral": 2,  "El Niño": 98},
        {"Trimestre": "ASO", "La Niña": 0, "Neutral": 4,  "El Niño": 96},
        {"Trimestre": "SON", "La Niña": 0, "Neutral": 5,  "El Niño": 95}
    ])
    return df_rescate, {"fuente": "Consenso Interno (Respaldo Actualizado)"}

# ==============================================================================
# 🔌 LLAVE 2: CONEXIÓN EN VIVO ÍNDICE ONI (NOAA PSL) - VERSIÓN ÚNICA
# ==============================================================================
@st.cache_data(show_spinner=False, ttl=43200)
def get_live_oni_data():
    """Descarga el registro oficial en vivo del Oceanic Niño Index (ONI) desde NOAA."""
    url = "https://psl.noaa.gov/data/correlation/oni.data"
    try:
        df = pd.read_csv(url, skiprows=1, delim_whitespace=True, header=None, 
                         names=['YEAR','01','02','03','04','05','06','07','08','09','10','11','12'])
        
        df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce')
        df = df.dropna(subset=['YEAR'])
        
        df_melt = df.melt(id_vars=['YEAR'], var_name='mes', value_name='oni')
        df_melt['oni'] = pd.to_numeric(df_melt['oni'], errors='coerce')
        df_melt = df_melt[df_melt['oni'] > -90.0] 
        
        df_melt['fecha'] = pd.to_datetime(df_melt['YEAR'].astype(int).astype(str) + '-' + df_melt['mes'] + '-01')
        df_melt = df_melt.sort_values('fecha').reset_index(drop=True)
        
        def clasificar_fase(val):
            if val >= 0.5: return "Niño"
            if val <= -0.5: return "Niña"
            return "Neutro"
            
        df_melt['fase_enso'] = df_melt['oni'].apply(clasificar_fase)
        df_melt.rename(columns={'oni': 'anomalia_oni'}, inplace=True)
        
        return df_melt[['fecha', 'anomalia_oni', 'fase_enso']]
    except Exception as e:
        return None

# ==============================================================================
# 🔌 LLAVE 3: CONEXIÓN DE ÍNDICE SOI (EN VIVO NOAA CPC)
# ==============================================================================
@st.cache_data(show_spinner=False, ttl=43200)
def get_live_soi_data():
    """Descarga el registro histórico oficial del SOI en formato de texto plano."""
    url = "https://www.cpc.ncep.noaa.gov/data/indices/soi"
    try:
        df = pd.read_csv(url, skiprows=3, delim_whitespace=True, header=None, 
                         names=['YEAR','01','02','03','04','05','06','07','08','09','10','11','12'])
        df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce')
        df = df.dropna(subset=['YEAR'])
        
        df_melt = df.melt(id_vars=['YEAR'], var_name='mes', value_name='soi')
        df_melt['soi'] = pd.to_numeric(df_melt['soi'], errors='coerce')
        df_melt = df_melt.dropna()
        
        df_melt['fecha'] = pd.to_datetime(df_melt['YEAR'].astype(int).astype(str) + '-' + df_melt['mes'] + '-01')
        return pd.DataFrame({'fecha': df_melt['fecha'], 'soi': df_melt['soi']}).sort_values('fecha').reset_index(drop=True)
    except Exception:
        return None

# ==============================================================================
# 🔌 LLAVE 4: CONEXIÓN DE ÍNDICE IOD (EN VIVO NOAA PSL)
# ==============================================================================
@st.cache_data(show_spinner=False, ttl=43200)
def get_live_iod_data():
    """Descarga el registro histórico oficial del Dipolo del Océano Índico (DMI)."""
    url = "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data"
    try:
        df = pd.read_csv(url, skiprows=1, delim_whitespace=True, header=None, 
                         names=['YEAR','01','02','03','04','05','06','07','08','09','10','11','12'])
        df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce')
        df = df.dropna(subset=['YEAR'])
        df = df[df['YEAR'] > 1800] 
        
        df_melt = df.melt(id_vars=['YEAR'], var_name='mes', value_name='iod')
        df_melt['iod'] = pd.to_numeric(df_melt['iod'], errors='coerce')
        df_melt = df_melt[df_melt['iod'] > -99] 
        
        df_melt['fecha'] = pd.to_datetime(df_melt['YEAR'].astype(int).astype(str) + '-' + df_melt['mes'] + '-01')
        return pd.DataFrame({'fecha': df_melt['fecha'], 'iod': df_melt['iod']}).sort_values('fecha').reset_index(drop=True)
    except Exception:
        return None
