# modules/iri_api.py

import json
import os
import pandas as pd
import requests
import streamlit as st

# --- 🌐 NUEVA RUTA AUTENTICADA (Columbia University ENSO Data) ---
# Cambiamos a la URL de alta disponibilidad proporcionada en tu correo
IRI_BASE_URL = "https://ftp.iri.columbia.edu/ensodata/"
LOCAL_DATA_PATH = "iri"

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_iri_data(filename):
    """
    Descarga datos del IRI y expone los errores HTTP reales a Streamlit.
    """
    try:
        user = st.secrets["iri"]["username"]
        pwd = st.secrets["iri"]["password"]
    except Exception:
        st.error("❌ Aleph Climático: No se encontraron las credenciales 'iri' en secrets.toml")
        return None

    url = f"{IRI_BASE_URL}{filename}"
    
    try:
        # Disfraz de navegador
        headers_disfraz = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'
        }
        
        response = requests.get(url, auth=(user, pwd), headers=headers_disfraz, timeout=15)
        
        # Evaluamos la respuesta exacta del servidor
        if response.status_code == 200:
            try:
                return response.json()
            except json.JSONDecodeError:
                # El servidor respondió, pero entregó un HTML (ej. página de bloqueo) en lugar de un JSON
                st.error(f"❌ Aleph Climático: Archivo corrupto o bloqueado. Respuesta del servidor: {response.text[:150]}")
                return None
        elif response.status_code in [401, 403]:
            st.error(f"❌ Aleph Climático (Error {response.status_code}): Columbia University denegó el acceso. Verifica que el usuario y clave en secrets.toml sean exactos.")
            return None
        elif response.status_code == 404:
            st.error(f"❌ Aleph Climático (Error 404): El archivo '{filename}' ya no existe en esa ruta de Columbia University.")
            return None
        else:
            st.error(f"❌ Aleph Climático (Error {response.status_code}): Fallo desconocido en el servidor IRI.")
            return None
            
    except requests.exceptions.Timeout:
        st.error("❌ Aleph Climático: El servidor de Columbia tardó demasiado en responder (Timeout).")
        return None
    except Exception as e:
        st.error(f"❌ Aleph Climático: Falla de red crítica: {str(e)}")
        return None

def _fallback_local(filename):
    """Lógica de rescate para leer archivos locales."""
    file_path = os.path.join(LOCAL_DATA_PATH, filename)
    if not os.path.exists(file_path):
        # Fallback extra en raíz
        file_path = filename if os.path.exists(filename) else None
    
    if file_path:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except: pass
    return None

# --- FUNCIONES DE PROCESAMIENTO (RESTAURADAS Y SEGURAS) ---

@st.cache_data(show_spinner=False)
def process_iri_plume(data_json):
    """Procesa el JSON de plumas (modelos spaghetti)."""
    if not data_json or "years" not in data_json: return None
    try:
        last_year_entry = data_json.get("years", [])[-1]
        if not last_year_entry.get("months"): return None
        last_month_entry = last_year_entry.get("months", [])[-1]
        
        month_idx = last_month_entry.get("month")
        year = last_year_entry.get("year")

        models_data = []
        for m in last_month_entry.get("models", []):
            clean_values = [x if x is not None and x > -100 else None for x in m.get("data", [])]
            models_data.append({"name": m.get("model", "Unknown"), "type": m.get("type", "Unknown"), "values": clean_values})

        seasons_base = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
        start_idx = (month_idx + 1) % 12
        forecast_seasons = [seasons_base[(start_idx + i) % 12] for i in range(9)]

        return {"year": year, "month_idx": month_idx, "seasons": forecast_seasons, "models": models_data}
    except: return None

@st.cache_data(show_spinner=False)
def process_iri_probabilities(data_json):
    """Procesa el JSON de probabilidades (barras)."""
    if not data_json or "years" not in data_json: return None
    try:
        last_year_entry = data_json.get("years", [])[-1]
        last_month_entry = last_year_entry.get("months", [])[-1]
        probs = []
        for p in last_month_entry.get("probabilities", []):
            probs.append({
                "Trimestre": p.get("season", ""),
                "La Niña": p.get("lanina", 0),
                "Neutral": p.get("neutral", 0),
                "El Niño": p.get("elnino", 0),
            })
        return pd.DataFrame(probs) if probs else None
    except: return None

# ==============================================================================
# 🔌 LLAVE MAESTRA: CONEXIÓN AL ALEPH CLIMÁTICO
# ==============================================================================
@st.cache_data(show_spinner=False)
def get_iri_enso_forecast():
    """
    Wrapper centralizado para obtener las probabilidades ENSO.
    """
    filename = "enso_iri_prob.json"
    data_json = fetch_iri_data(filename)
    
    if data_json is None:
        raise ValueError("La descarga fue bloqueada o falló. Revisa los mensajes de error en pantalla.")
        
    df_probs = process_iri_probabilities(data_json)
    
    # Si la descarga funcionó pero el procesador falló, es porque el IRI cambió el formato del JSON
    if df_probs is None or df_probs.empty:
        llaves_encontradas = list(data_json.keys())[:5]
        raise ValueError(f"JSON descargado con éxito, pero la estructura es irreconocible. Llaves raíz: {llaves_encontradas}")
    
    return df_probs, data_json
