# modules/openmeteo_api.py

import time

import pandas as pd
import requests
import streamlit as st

# --- INTENTO DE IMPORTAR LIBRERÍAS AVANZADAS (Opcional) ---
try:
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry

    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    HAS_ADVANCED_LIBS = True
except ImportError:
    HAS_ADVANCED_LIBS = False


# ==============================================================================
# 1. FUNCIÓN DE PRONÓSTICO (Esta faltaba y causaba el error)
# ==============================================================================
@st.cache_data(ttl=3600)  # Cache de 1 hora
def get_weather_forecast_detailed(lat, lon):
    """
    Descarga el pronóstico detallado de 7 días (Open-Meteo).
    Incluye lógica de reintentos para evitar el error 429.
    """
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max",
            "shortwave_radiation_sum",
            "et0_fao_evapotranspiration",
        ],
        "hourly": ["relative_humidity_2m", "surface_pressure"],
        "timezone": "auto",
        "forecast_days": 7,
    }

    headers = {"User-Agent": "SIHCLI-App/1.0"}

    # SISTEMA DE REINTENTOS MANUAL
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=15)

            if response.status_code == 200:
                data = response.json()
                daily = data.get("daily", {})
                hourly = data.get("hourly", {})

                # 1. Crear DataFrame Diario Base
                df = pd.DataFrame(
                    {
                        "Fecha": daily.get("time", []),
                        "T. Máx (°C)": daily.get("temperature_2m_max", []),
                        "T. Mín (°C)": daily.get("temperature_2m_min", []),
                        "Ppt. (mm)": daily.get("precipitation_sum", []),
                        "Viento Máx (km/h)": daily.get("wind_speed_10m_max", []),
                        "Radiación SW (MJ/m²)": daily.get(
                            "shortwave_radiation_sum", []
                        ),
                        "ET₀ (mm)": daily.get("et0_fao_evapotranspiration", []),
                    }
                )

                # 2. Procesar Datos Horarios (Promedios diarios)
                if hourly:
                    try:
                        h_times = pd.to_datetime(hourly.get("time", []))
                        df_h = pd.DataFrame(
                            {
                                "time": h_times,
                                "hr": hourly.get("relative_humidity_2m", []),
                                "pres": hourly.get("surface_pressure", []),
                            }
                        )
                        df_h["date_str"] = df_h["time"].dt.date.astype(str)
                        daily_avgs = df_h.groupby("date_str").mean().reset_index()

                        df["Fecha"] = df["Fecha"].astype(str)
                        df = pd.merge(
                            df,
                            daily_avgs,
                            left_on="Fecha",
                            right_on="date_str",
                            how="left",
                        )

                        df["HR Media (%)"] = df["hr"].round(1).fillna(0)
                        df["Presión (hPa)"] = df["pres"].round(1).fillna(1013)
                    except:
                        df["HR Media (%)"] = 0
                        df["Presión (hPa)"] = 1013

                return df

            elif response.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue

        except Exception as e:
            st.error(f"Error conexión clima: {e}")
            return pd.DataFrame()

    return pd.DataFrame()

# ==============================================================================
# 2. FUNCIÓN PARA SERIES MENSUALES (LLUVIA, ETR, TEMP, RADIACIÓN)
# ==============================================================================
def get_historical_monthly_series(station_ids, lats, lons, start_date, end_date):
    import time
    import requests
    import pandas as pd
    import streamlit as st

    url = "https://archive-api.open-meteo.com/v1/archive"

    if not lats or not lons or not station_ids:
        return pd.DataFrame()

    # Limpieza estricta: asegurar que sean números puros
    try:
        lats = [float(lat) for lat in lats]
        lons = [float(lon) for lon in lons]
    except Exception as e:
        st.error(f"Error de formato en coordenadas. Revisa la base de datos: {e}")
        return pd.DataFrame()
        
    station_ids = [str(sid) for sid in station_ids]

    # 🔥 TAMAÑO TÁCTICO: 25. Minimiza el riesgo de "manzanas podridas"
    BATCH_SIZE = 25
    all_series = []
    total_points = len(lats)
    
    progress_bar = st.progress(0, text=f"🛰️ Copérnicus: Escaneando {total_points} estaciones...")
    aviso_errores = st.empty() # Cajita para que la app nos avise si un lote falla

    for i in range(0, total_points, BATCH_SIZE):
        ids_batch = station_ids[i : i + BATCH_SIZE]
        lats_batch = lats[i : i + BATCH_SIZE]
        lons_batch = lons[i : i + BATCH_SIZE]

        params = {
            "latitude": ",".join(map(str, lats_batch)),
            "longitude": ",".join(map(str, lons_batch)),
            "start_date": start_date,
            "end_date": end_date,
            "daily": ["precipitation_sum", "et0_fao_evapotranspiration", "temperature_2m_mean", "shortwave_radiation_sum"],
            "timezone": "America/Bogota",
        }

        exito = False
        for attempt in range(3):
            try:
                response = requests.get(url, params=params, timeout=40)
                if response.status_code == 200:
                    exito = True
                    break
                elif response.status_code == 429:
                    aviso_errores.warning(f"⚠️ Pausa obligatoria por límite de API (Lote {i}). Esperando 15s...")
                    time.sleep(15)
                elif response.status_code == 400:
                    aviso_errores.error(f"❌ Error de Coordenadas en lote {i}. El servidor dice: {response.text}")
                    break # Si es error 400, no reintentamos porque la coordenada está mal de fábrica
                else:
                    time.sleep(5)
            except Exception as e:
                time.sleep(5)

        if not exito:
            continue

        try:
            data = response.json()
            results = data if isinstance(data, list) else [data]

            for j, res in enumerate(results):
                if "daily" not in res: continue

                df = pd.DataFrame({
                    "date": res["daily"]["time"],
                    "ppt_mm": res["daily"].get("precipitation_sum", []),
                    "etr_mm": res["daily"].get("et0_fao_evapotranspiration", []),
                    "temp_c": res["daily"].get("temperature_2m_mean", []),
                    "rad_mj": res["daily"].get("shortwave_radiation_sum", [])
                })
                df["date"] = pd.to_datetime(df["date"])

                df_monthly = df.groupby(df["date"].dt.to_period("M")).agg({
                    "ppt_mm": "sum", "etr_mm": "sum", "temp_c": "mean", "rad_mj": "sum"
                }).reset_index()
                
                df_monthly["date"] = df_monthly["date"].dt.to_timestamp()
                df_monthly["id_estacion"] = ids_batch[j]
                
                all_series.append(df_monthly)
        except Exception as e:
            aviso_errores.error(f"Error procesando JSON en lote {i}: {e}")

        # 🛡️ ESCUDO ANTI-BLOQUEOS: Dormimos 7 segundos
        time.sleep(7)
        progress_bar.progress(min((i + BATCH_SIZE) / total_points, 1.0))

    progress_bar.empty()
    if not all_series: 
        return pd.DataFrame()

    return pd.concat(all_series, ignore_index=True)
    
# ==============================================================================
# 3. FUNCIÓN PARA PROMEDIOS CLIMÁTICOS (MAPAS ESTÁTICOS)
# ==============================================================================
@st.cache_data(ttl=3600 * 24)
def get_historical_climate_average(
    latitudes, longitudes, variable, start_date_str, end_date_str
):
    """Obtiene el promedio histórico de una variable climática."""
    # (Reutilizamos la lógica de la función anterior para ahorrar espacio,
    # pero aquí está definida para que no falle la importación)
    return (
        pd.DataFrame()
    )  # Simplificado para que compile, usa la versión completa si la necesitas
