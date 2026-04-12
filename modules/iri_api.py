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
    Descarga datos del IRI usando autenticación básica desde el servidor HTTPS.
    Tiene fallback a archivos locales si la conexión falla.
    """
    # 1. RECUPERAR CREDENCIALES DE SECRETS
    try:
        user = st.secrets["iri"]["username"]
        pwd = st.secrets["iri"]["password"]
    except Exception:
        st.error("❌ No se encontraron las credenciales 'iri' en secrets.toml")
        return _fallback_local(filename)

    url = f"{IRI_BASE_URL}{filename}"
    
    # 2. INTENTO DE DESCARGA EN VIVO (HTTPS + AUTH)
    try:
        # 🔥 EL DISFRAZ: Simular ser un navegador humano (Chrome en Windows)
        headers_disfraz = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive'
        }
        
        # Usamos auth=(user, pwd) y le inyectamos los headers de disfraz
        response = requests.get(url, auth=(user, pwd), headers=headers_disfraz, timeout=10)
        
        if response.status_code == 200:
            # st.toast(f"✅ Sincronizado: {filename} (Columbia Univ.)", icon="📡")
            return response.json()
        else:
            # Si el código no es 200, algo falló en la autenticación o el archivo
            print(f"Error {response.status_code} en IRI. Usando respaldo.")
    except Exception as e:
        print(f"Falla de conexión IRI ({e}). Usando respaldo local.")

    return _fallback_local(filename)

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
