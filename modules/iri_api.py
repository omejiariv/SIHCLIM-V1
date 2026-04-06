# modules/iri_api.py

import json
import os
import pandas as pd
import streamlit as st

# --- 📂 NUEVA RUTA CORREGIDA (Acorde a la limpieza de GitHub) ---
LOCAL_DATA_PATH = "iri"

@st.cache_data(show_spinner=False, ttl=3600) # Caché de 1 hora para evitar lecturas constantes
def fetch_iri_data(filename):
    """
    Carga los datos del IRI desde archivos locales (JSON) de la carpeta /iri.
    """
    file_path = os.path.join(LOCAL_DATA_PATH, filename)

    try:
        if not os.path.exists(file_path):
            # Intento de fallback: buscar en la raíz del proyecto por si acaso
            if os.path.exists(filename):
                file_path = filename
            else:
                st.error(f"⚠️ Archivo no encontrado: `{file_path}`. Verifica que esté en la carpeta `iri/`.")
                return None

        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    except json.JSONDecodeError:
        st.error(f"❌ El archivo `{filename}` no es un JSON válido o está corrupto.")
        return None
    except Exception as e:
        st.error(f"❌ Error leyendo archivo `{filename}`: {e}")
        return None


# --- FUNCIONES DE PROCESAMIENTO (BLINDADAS CON .get()) ---

@st.cache_data(show_spinner=False)
def process_iri_plume(data_json):
    """Procesa el JSON de plumas (modelos spaghetti) de manera segura."""
    if not data_json or "years" not in data_json:
        return None

    try:
        # Busca el último año disponible
        last_year_entry = data_json.get("years", [])[-1]
        year = last_year_entry.get("year")

        if not last_year_entry.get("months"):
            return None
            
        # Busca el último mes disponible
        last_month_entry = last_year_entry.get("months", [])[-1]
        month_idx = last_month_entry.get("month")

        models_data = []
        # .get() evita que el código colapse si la llave "models" no existe
        for m in last_month_entry.get("models", []):
            # Limpieza de valores centinela (-999, etc)
            clean_values = [
                x if x is not None and x > -100 else None for x in m.get("data", [])
            ]
            models_data.append(
                {"name": m.get("model", "Unknown"), "type": m.get("type", "Unknown"), "values": clean_values}
            )

        seasons_base = [
            "DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ",
            "JJA", "JAS", "ASO", "SON", "OND", "NDJ",
        ]
        start_idx = (month_idx + 1) % 12
        forecast_seasons = [seasons_base[(start_idx + i) % 12] for i in range(9)]

        return {
            "year": year,
            "month_idx": month_idx,
            "seasons": forecast_seasons,
            "models": models_data,
        }
    except Exception as e:
        st.error(f"Error procesando estructura Plume: {e}")
        return None


@st.cache_data(show_spinner=False)
def process_iri_probabilities(data_json):
    """Procesa el JSON de probabilidades (barras) de manera segura."""
    if not data_json or "years" not in data_json:
        return None

    try:
        last_year_entry = data_json.get("years", [])[-1]
        if not last_year_entry.get("months"):
            return None

        last_month_entry = last_year_entry.get("months", [])[-1]

        probs = []
        for p in last_month_entry.get("probabilities", []):
            probs.append(
                {
                    "Trimestre": p.get("season", ""),
                    "La Niña": p.get("lanina", 0),
                    "Neutral": p.get("neutral", 0),
                    "El Niño": p.get("elnino", 0),
                }
            )

        return pd.DataFrame(probs) if probs else None
    except Exception as e:
        st.error(f"Error procesando estructura Probabilities: {e}")
        return None
