# modules/utils.py

import io
import os
import unicodedata
import pandas as pd
import numpy as np
import streamlit as st

# ==============================================================================
# 💉 SISTEMA INMUNOLÓGICO Y VARIABLES GLOBALES
# ==============================================================================
def inicializar_torrente_sanguineo():
    """
    Vacuna de memoria: Asegura que todas las arterias del Gemelo Digital 
    tengan un valor base (Fallback) para evitar colapsos por saltos de página.
    """
    diccionario_maestro = {
        # 1. Biofísica y Geomorfología
        'aleph_q_max_m3s': 0.0,
        'geomorfo_q_pico_racional': 0.0,
        'aleph_twi_umbral': 0.0,
        'ultima_zona_procesada': "",
        'gdf_rios': None, 'grid_obj': None, 'acc_obj': None, 'fdir_obj': None,
        
        # 2. Ecohidrología y Tormentas
        'eco_lodo_total_m3': 0.0,
        'eco_lodo_colas_m3': 0.0,
        'eco_lodo_fondo_m3': 0.0,
        'eco_lodo_abrasivo_m3': 0.0,
        'eco_fosforo_kg': 0.0,
        'eco_sobrecosto_usd': 0.0,
        'activar_tormenta_sankey': False,
        
        # 3. Metabolismo y Calidad
        'carga_dbo_total_ton': 0.0,
        'carga_dbo_mitigada_ton': 0.0,
        'ica_bovinos_calc_met': 1500,  # Valor de supervivencia
        'ica_porcinos_calc_met': 500,   # Valor de supervivencia
        'pob_hum_calc_met': 5000,       # Valor de supervivencia
        
        # 4. Estado de Aplicación
        'ejecutar_aleph': False,
        'beta_unlocked': False
    }

    # Inyección silenciosa: Solo crea la variable si no existe
    for llave, valor_seguro in diccionario_maestro.items():
        if llave not in st.session_state:
            st.session_state[llave] = valor_seguro


# ==============================================================================
# 🧽 FUNCIONES MAESTRAS DE LIMPIEZA Y LECTURA (CENTRALIZADAS)
# ==============================================================================
@st.cache_data
def standardize_numeric_column(series):
    """
    Convierte una serie de Pandas a valores numéricos de manera robusta,
    reemplazando comas por puntos como separador decimal.
    """
    series_clean = series.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(series_clean, errors="coerce")

def normalizar_texto(texto):
    """
    Limpiador extremo: Convierte texto a minúsculas, sin tildes y sin espacios extras. 
    Ideal para evitar fallos en búsquedas SQL o cruces de bases de datos.
    """
    if pd.isna(texto): return ""
    texto_str = str(texto).lower().strip()
    return unicodedata.normalize('NFKD', texto_str).encode('ascii', 'ignore').decode('utf-8')

@st.cache_data
def leer_csv_robusto(ruta):
    """
    Lector blindado: Intenta leer el archivo detectando automáticamente el separador.
    """
    if not os.path.exists(ruta):
        return pd.DataFrame()
        
    try:
        df = pd.read_csv(ruta, sep=None, engine='python')
        df.columns = df.columns.str.replace('\ufeff', '').str.strip()
        return df
    except Exception:
        try:
            df = pd.read_csv(ruta, sep=';', low_memory=False)
            if len(df.columns) < 3: 
                df = pd.read_csv(ruta, sep=',', low_memory=False)
            df.columns = df.columns.str.replace('\ufeff', '').str.strip()
            return df
        except Exception as e:
            # Silenciamos el error en interfaz pero lo dejamos disponible para debug
            return pd.DataFrame()

# ==============================================================================
# 📥 DESCARGAS Y EXPORTACIÓN DE GRÁFICOS
# ==============================================================================
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
            st.warning("No se pudo generar la imagen PNG. Asegúrate de tener 'kaleido' instalado.")

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
# 🧠 CEREBRO CENTRAL: MATRICES MAESTRAS Y METABOLISMO
# ==============================================================================
@st.cache_data(show_spinner=False, ttl=3600)
def descargar_matrices_produccion():
    """Descarga los cerebros pre-entrenados desde Supabase (Se guarda en caché 1 hora)"""
    try:
        url_demo = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/Matriz_Maestra_Demografica.csv"
        url_pecu = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/Matriz_Maestra_Pecuaria.csv"
        url_prop = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/Matriz_Proporciones_Veredales_Final.csv" 
        
        df_d = pd.read_csv(url_demo)
        df_p = pd.read_csv(url_pecu)
        df_prop = pd.read_csv(url_prop)
        return df_d, df_p, df_prop
    except Exception as e:
        return None, None, None

def encender_gemelo_digital():
    """
    Secuencia de encendido maestro: Inyecta el Torrente Sanguíneo (Fallback variables) 
    y descarga las matrices si la memoria está vacía.
    """
    # 1. Aplicamos la vacuna de variables primero
    inicializar_torrente_sanguineo()
    
    # 2. Descargamos los datos estáticos si no existen
    if 'df_matriz_proporciones' not in st.session_state:
        df_demo, df_pecu, df_prop = descargar_matrices_produccion()
        
        if df_demo is not None and not df_demo.empty:
            st.session_state['df_matriz_demografica'] = df_demo
        if df_pecu is not None and not df_pecu.empty:
            st.session_state['df_matriz_pecuaria'] = df_pecu
        if df_prop is not None and not df_prop.empty:
            st.session_state['df_matriz_proporciones'] = df_prop


def obtener_metabolismo_exacto(nombre_seleccion, anio_destino=None):
    """
    🧠 Buscador Multi-Nivel: 
    1. Busca nombre exacto. 
    2. Busca traducido (R. -> Río). 
    3. Busca palabra clave (LIKE).
    """
    try:
        from modules.db_manager import get_engine
        engine = get_engine()
    except Exception:
        engine = None

    res = {
        'pob_urbana': 0.0, 'pob_rural': 0.0, 'pob_total': 0.0,
        'bovinos': 0.0, 'porcinos': 0.0, 'aves': 0.0,
        'origen_humano': "Sin Datos (0)",
        'origen_pecuario': "Sin Datos (0)"
    }
    
    if engine is None: return res

    # --- PREPARACIÓN DE NOMBRES ---
    nombre_exacto = str(nombre_seleccion).strip().lower()
    
    nombre_traducido = str(nombre_seleccion).strip()
    if nombre_traducido.startswith("R. "): nombre_traducido = nombre_traducido.replace("R. ", "Río ")
    elif nombre_traducido.startswith("Q. "): nombre_traducido = nombre_traducido.replace("Q. ", "Quebrada ")
    nombre_traducido = nombre_traducido.lower()
    
    palabra_clave = nombre_exacto.replace("r. ", "").replace("río ", "").replace("rio ", "").replace("q. ", "").replace("quebrada ", "").strip()

    def proyectar(fila, col_base, col_anio_base):
        """Aplica fórmulas matemáticas de proyección poblacional"""
        if anio_destino is None: return float(fila[col_base])
        x_norm = anio_destino - fila[col_anio_base]
        try:
            mod = fila.get('Modelo_Recomendado', 'Polinomial_3')
            if mod == 'Logístico': return max(0.0, fila['Log_K'] / (1 + fila['Log_a'] * np.exp(-fila['Log_r'] * x_norm)))
            elif mod == 'Exponencial': return max(0.0, fila['Exp_a'] * np.exp(fila['Exp_b'] * x_norm))
            else: return max(0.0, fila['Poly_A']*(x_norm**3) + fila['Poly_B']*(x_norm**2) + fila['Poly_C']*x_norm + fila['Poly_D'])
        except: return float(fila[col_base])

    # --- 1. MOTOR DEMOGRÁFICO ---
    try:
        q_demo = "SELECT * FROM matriz_maestra_demografica WHERE LOWER(TRIM(\"Territorio\")) = %(z)s"
        df_demo = pd.read_sql(q_demo, engine, params={"z": nombre_exacto})
        
        if df_demo.empty and nombre_exacto != nombre_traducido:
            df_demo = pd.read_sql(q_demo, engine, params={"z": nombre_traducido})
            
        if df_demo.empty and len(palabra_clave) > 3:
            q_demo_like = "SELECT * FROM matriz_maestra_demografica WHERE LOWER(\"Territorio\") LIKE %(z)s"
            df_demo = pd.read_sql(q_demo_like, engine, params={"z": f"%{palabra_clave}%"})

        if not df_demo.empty:
            fu = df_demo[df_demo['Area'] == 'Urbana']
            fr = df_demo[df_demo['Area'] == 'Rural']
            pob_u = proyectar(fu.iloc[0], 'Pob_Base', 'Año_Base') if not fu.empty else 0.0
            pob_r = proyectar(fr.iloc[0], 'Pob_Base', 'Año_Base') if not fr.empty else 0.0
            res['pob_urbana'], res['pob_rural'], res['pob_total'] = pob_u, pob_r, pob_u + pob_r
            res['origen_humano'] = "Matriz Maestra SQL"
    except Exception: pass # Silencioso para producción

    # --- 2. MOTOR PECUARIO ---
    try:
        q_pecu = "SELECT * FROM matriz_maestra_pecuaria WHERE LOWER(TRIM(\"Territorio\")) = %(z)s"
        df_pecu = pd.read_sql(q_pecu, engine, params={"z": nombre_exacto})
        
        if df_pecu.empty and nombre_exacto != nombre_traducido:
            df_pecu = pd.read_sql(q_pecu, engine, params={"z": nombre_traducido})
            
        if df_pecu.empty and len(palabra_clave) > 3:
            q_pecu_like = "SELECT * FROM matriz_maestra_pecuaria WHERE LOWER(\"Territorio\") LIKE %(z)s"
            df_pecu = pd.read_sql(q_pecu_like, engine, params={"z": f"%{palabra_clave}%"})

        if not df_pecu.empty:
            f_bov = df_pecu[df_pecu['Especie'] == 'Bovinos']
            f_por = df_pecu[df_pecu['Especie'] == 'Porcinos']
            f_ave = df_pecu[df_pecu['Especie'] == 'Aves']
            if not f_bov.empty: res['bovinos'] = proyectar(f_bov.iloc[0], 'Poblacion_Base', 'Año_Base')
            if not f_por.empty: res['porcinos'] = proyectar(f_por.iloc[0], 'Poblacion_Base', 'Año_Base')
            if not f_ave.empty: res['aves'] = proyectar(f_ave.iloc[0], 'Poblacion_Base', 'Año_Base')
            res['origen_pecuario'] = "Matriz Pecuaria SQL"
    except Exception: pass # Silencioso para producción

    return res
