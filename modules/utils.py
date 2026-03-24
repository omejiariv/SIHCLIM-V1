# modules/utils.py

import io
import pandas as pd
import streamlit as st
import unicodedata
import os

# --- FUNCIÓN PARA CORRECCIÓN NUMÉRICA ---
@st.cache_data
def standardize_numeric_column(series):
    """
    Convierte una serie de Pandas a valores numéricos de manera robusta,
    reemplazando comas por puntos como separador decimal.
    """
    series_clean = series.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(series_clean, errors="coerce")


def display_plotly_download_buttons(fig, file_prefix):
    """Muestra botones de descarga para un gráfico Plotly (HTML y PNG)."""
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        html_buffer = io.StringIO()
        fig.write_html(html_buffer, include_plotlyjs="cdn")
        st.download_button(
            label="Descargar Gráfico (HTML)",
            data=html_buffer.getvalue(),
            file_name=f"{file_prefix}.html",
            mime="text/html",
            key=f"dl_html_{file_prefix}",
        )
    with col2:
        try:
            img_bytes = fig.to_image(format="png", width=1200, height=700, scale=2)
            st.download_button(
                label="Descargar Gráfico (PNG)",
                data=img_bytes,
                file_name=f"{file_prefix}.png",
                mime="image/png",
                key=f"dl_png_{file_prefix}",
            )
        except Exception:
            st.warning(
                "No se pudo generar la imagen PNG. Asegúrate de tener 'kaleido' instalado."
            )


def add_folium_download_button(map_object, file_name):
    """Muestra un botón de descarga para un mapa de Folium (HTML)."""
    st.markdown("---")
    map_buffer = io.BytesIO()
    map_object.save(map_buffer, close_file=False)
    st.download_button(
        label="Descargar Mapa (HTML)",
        data=map_buffer.getvalue(),
        file_name=file_name,
        mime="text/html",
        key=f"dl_map_{file_name.replace('.', '_')}",

    )

# ==============================================================================
# 🧽 FUNCIONES MAESTRAS DE LIMPIEZA Y LECTURA (CENTRALIZADAS)
# ==============================================================================

def normalizar_texto(texto):
    """
    Mata-tildes y espacios: Convierte cualquier texto a minúsculas, 
    sin tildes y sin espacios en blanco a los lados. Ideal para cruzar bases de datos.
    """
    if pd.isna(texto): return ""
    texto_str = str(texto).lower().strip()
    return unicodedata.normalize('NFKD', texto_str).encode('ascii', 'ignore').decode('utf-8')

@st.cache_data
def leer_csv_robusto(ruta):
    """
    Lector blindado: Intenta leer el archivo detectando automáticamente el separador.
    Si falla, intenta con punto y coma (;), comas (,) y diferentes codificaciones.
    """
    if not os.path.exists(ruta):
        return pd.DataFrame()
        
    try:
        # Intento 1: Detección automática (ideal para la mayoría)
        df = pd.read_csv(ruta, sep=None, engine='python')
        df.columns = df.columns.str.replace('\ufeff', '').str.strip()
        return df
    except Exception:
        try:
            # Intento 2: Forzar separador latinoamericano/europeo (;)
            df = pd.read_csv(ruta, sep=';', low_memory=False)
            if len(df.columns) < 3: # Si leyó todo en una sola columna, el separador era coma
                df = pd.read_csv(ruta, sep=',', low_memory=False)
            df.columns = df.columns.str.replace('\ufeff', '').str.strip()
            return df
        except Exception as e:
            print(f"Error crítico leyendo {ruta}: {e}")
            return pd.DataFrame()

import streamlit as st
import pandas as pd

@st.cache_data(show_spinner=False, ttl=3600)
def descargar_matrices_produccion():
    """Descarga los cerebros pre-entrenados desde Supabase (Se guarda en caché 1 hora para máxima velocidad)"""
    try:
        # URLs públicas de tu Supabase
        url_demo = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/Matriz_Maestra_Demografica.csv"
        url_pecu = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/Matriz_Maestra_Pecuaria.csv"
        
        df_d = pd.read_csv(url_demo)
        df_p = pd.read_csv(url_pecu)
        return df_d, df_p
    except Exception as e:
        print(f"Error descargando cerebros: {e}")
        return None, None

def encender_gemelo_digital():
    """Verifica si la memoria está vacía (Ej. nuevo usuario o F5) y la llena instantáneamente"""
    if 'df_matriz_demografica' not in st.session_state or 'df_matriz_pecuaria' not in st.session_state:
        df_demo, df_pecu = descargar_matrices_produccion()
        
        if df_demo is not None and not df_demo.empty:
            st.session_state['df_matriz_demografica'] = df_demo
            
        if df_pecu is not None and not df_pecu.empty:
            st.session_state['df_matriz_pecuaria'] = df_pecu
