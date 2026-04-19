# modules/utils.py

import io
import os
import re
import unicodedata
import warnings
import pandas as pd
import numpy as np
import streamlit as st
from sqlalchemy import text

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
        
        # 3. Metabolismo y Calidad (Valores de supervivencia sincronizados)
        'carga_dbo_total_ton': 0.0,
        'carga_dbo_mitigada_ton': 0.0,
        'ica_bovinos_calc_met': 0.0,
        'ica_porcinos_calc_met': 0.0,
        'pob_hum_calc_met': 0.0,
        
        # 4. Contexto Territorial (Sincronización con Aleph)
        'aleph_lugar': "Antioquia",
        'aleph_escala': "Departamental",
        'aleph_anio': 2024,
        'aleph_pob_total': 0.0,
        
        # 5. Estado de Aplicación
        'ejecutar_aleph': False,
        'beta_unlocked': False
    }

    for llave, valor_seguro in diccionario_maestro.items():
        if llave not in st.session_state:
            st.session_state[llave] = valor_seguro


# ==============================================================================
# 🧽 FUNCIONES MAESTRAS DE LIMPIEZA (CENTRALIZADAS Y BLINDADAS V2)
# ==============================================================================

@st.cache_data(ttl=3600)
def cargar_diccionario_veredas():
    # ☁️ LECTURA DIRECTA DESDE TU BUCKET PÚBLICO EN SUPABASE
    url = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/homologacion_veredas.csv"
    try:
        return pd.read_csv(url)
    except:
        return pd.DataFrame()

def normalizar_texto_maestro(t, municipio_padre=""):
    """
    La aplanadora de texto definitiva para el SIHCLI-POTER.
    Maneja el 99% de las inconsistencias geográficas y ortográficas.
    """
    if not t or pd.isna(t): return ""
    
    # 1. Base minúscula y limpieza de bordes
    t = str(t).lower().strip()

    # -------------------------------------------------------------
    # 💉 VACUNA VEREDAL: LECTURA DEL DICCIONARIO EXTERNO (NUBE)
    # -------------------------------------------------------------
    if municipio_padre:
        id_busqueda = t.upper() + "_" + str(municipio_padre).upper().strip()
        id_busqueda = re.sub(r'[^A-Z0-9_]', '', id_busqueda)
        
        df_homologacion = cargar_diccionario_veredas()
        if not df_homologacion.empty and 'ID_TABLA' in df_homologacion.columns:
            match = df_homologacion[df_homologacion['ID_TABLA'] == id_busqueda]
            if not match.empty:
                id_curado = str(match.iloc[0]['ID_MAPA'])
                return id_curado.split("_")[0].lower()
    # ------------------------------------------------------------- 

    # 2. Quitar sufijos técnicos de la UI si vienen pegados (NSS, SZH, etc.)
    t = re.sub(r'\s*-\s*nss.*|\s*-\s*szh.*|\s*-\s*zh.*|\s*-\s*ah.*', '', t)

    # 3. Quitar tildes y caracteres especiales
    t = ''.join(c for c in unicodedata.normalize('NFD', t) if unicodedata.category(c) != 'Mn')

    # 4. DICCIONARIO GENÉRICO HIDROLÓGICO Y ESPACIAL
    reemplazos_hidro = {
        r'\brio\b': 'r', r'\br\.\s*': 'r ',
        r'\bquebrada\b': 'q', r'\bqda\.?\s*': 'q ', r'\bq\.\s*': 'q ',
        r'\bcano\b': 'cn', r'\bc\.\s*': 'cn ',
        r'\barroyo\b': 'a', r'\ba\.\s*': 'a ',
        r'\bcienaga\b': 'cga', r'\bcga\.\s*': 'cga '
    }
    for patron, reemplazo in reemplazos_hidro.items():
        t = re.sub(patron, reemplazo, t)

    # 5. Stop words (Veredas, sectores)
    stop_words = [r'\bvereda\b', r'\bvda\.?\b', r'\bsector\b', r'\bcaserio\b', r'\bcentro poblado\b', r'\bcp\b', r'\bcorregimiento\b', r'\bcorreg\b', r'\bcge\b']
    for word in stop_words: 
        t = re.sub(word, '', t)

    # 6. Rebeldes Municipales (Antioquia)
    rebeldes_mpio = {
        r'\bel carmen de viboral\b': 'carmen de viboral',
        r'\bsan vicente ferrer\b': 'san vicente',
        r'\bsan jose de la montana\b': 'san jose de la montana',
        r'\bdonmatias\b': 'don matias',
        r'\bsantafe de antioquia\b': 'santa fe de antioquia',
        r'\bel santuario\b': 'santuario',
        r'\bel penol\b': 'penol'
    }
    for regex, reemplazo in rebeldes_mpio.items(): 
        t = re.sub(regex, reemplazo, t)

    # 7. Destruir puntuación restante y colapsar espacios
    t = re.sub(r'[^a-z0-9\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()

    # 8. Diccionario final estricto
    diccionario_final = {
        "bogotadc": "bogota", "sanjosedecucuta": "cucuta", 
        "laguajira": "guajira", "valle": "valledelcauca",
        "r aures": "aures", "r buey": "buey"
    }
    
    t_sin_espacios = t.replace(" ", "")
    if t_sin_espacios in diccionario_final: 
        return diccionario_final[t_sin_espacios]

    return t

# 🔥 ALIAS DE ORO: Evita que se rompan las otras páginas que importan 'normalizar_texto'
normalizar_texto = normalizar_texto_maestro

@st.cache_data
def standardize_numeric_column(series):
    """Convierte series a números manejando separadores de miles y decimales latinos."""
    if series.dtype == object:
        series = series.str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    return pd.to_numeric(series, errors="coerce")

# ==============================================================================
# 🧠 CEREBRO CENTRAL: MATRICES Y GEMELO DIGITAL
# ==============================================================================

def encender_gemelo_digital():
    """
    Secuencia de ignición: Inyecta memoria base y descarga matrices de producción.
    """
    inicializar_torrente_sanguineo()
    
    if 'df_matriz_demografica' not in st.session_state:
        try:
            url_base = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/"
            st.session_state['df_matriz_demografica'] = pd.read_csv(f"{url_base}Matriz_Maestra_Demografica.csv")
            st.session_state['df_matriz_pecuaria'] = pd.read_csv(f"{url_base}Matriz_Maestra_Pecuaria.csv")
        except Exception as e:
            st.error(f"⚠️ Error al conectar con el cerebro digital (Supabase): {e}")

def obtener_metabolismo_exacto(nombre_seleccion, anio_destino=None):
    """
    Buscador de alta precisión: Cruza el territorio con las matrices SQL 
    y proyecta la población/ganado al año deseado.
    """
    from modules.db_manager import get_engine
    engine = get_engine()
    
    res = {
        'pob_urbana': 0.0, 'pob_rural': 0.0, 'pob_total': 0.0,
        'bovinos': 0.0, 'porcinos': 0.0, 'aves': 0.0,
        'status': "Sin Datos"
    }

    if not engine: return res
    
    # 🔥 NORMALIZACIÓN AGRESIVA
    nombre_q = normalizar_texto(nombre_seleccion)

    def proyectar_valor(fila, anio):
        if not anio: return float(fila['Pob_Base'] if 'Pob_Base' in fila else fila['Poblacion_Base'])
        t = anio - fila['Año_Base']
        mod = fila.get('Modelo_Recomendado', 'Polinomial_3')
        try:
            if mod == 'Logístico': return fila['Log_K'] / (1 + fila['Log_a'] * np.exp(-fila['Log_r'] * t))
            if mod == 'Exponencial': return fila['Exp_a'] * np.exp(fila['Exp_b'] * t)
            return fila['Poly_A']*(t**3) + fila['Poly_B']*(t**2) + fila['Poly_C']*t + fila['Poly_D']
        except: return 0.0

    try:
        # 1. Consulta Demográfica (Búsqueda por MATCH_ID normalizado agresivamente)
        with engine.connect() as conn:
            q_demo = text('SELECT * FROM matriz_maestra_demografica')
            df_all = pd.read_sql(q_demo, conn)
            # Aplicamos la aplanadora a toda la base de datos en memoria para garantizar el cruce
            df_all['MATCH'] = df_all['Territorio'].apply(normalizar_texto)
            
            # Buscamos coincidencias exactas primero
            df_res = df_all[df_all['MATCH'] == nombre_q]
            
            # Si no hay coincidencia exacta, aplicamos Fuzzy Matching como salvavidas
            if df_res.empty:
                import difflib
                territorios_db = df_all['MATCH'].unique().tolist()
                matches = difflib.get_close_matches(nombre_q, territorios_db, n=1, cutoff=0.7)
                if matches:
                    df_res = df_all[df_all['MATCH'] == matches[0]]
            
            if not df_res.empty:
                for area in ['Urbana', 'Rural', 'Total']:
                    # Hacemos case-insensitive el match del Área
                    row = df_res[df_res['Area'].str.capitalize() == area]
                    if not row.empty:
                        val = proyectar_valor(row.iloc[0], anio_destino)
                        if area == 'Urbana': res['pob_urbana'] = val
                        if area == 'Rural': res['pob_rural'] = val
                        if area == 'Total': res['pob_total'] = val
                res['status'] = "Sincronizado (DANE)"
    except Exception as e:
        res['status'] = f"Error: {e}"

    return res

# ==============================================================================
# 📥 EXPORTACIÓN Y UI
# ==============================================================================

def display_plotly_download_buttons(fig, file_prefix):
    """Muestra botones de descarga para un gráfico Plotly."""
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        html_bytes = fig.to_html(include_plotlyjs="cdn").encode('utf-8')
        st.download_button("Descargar HTML", html_bytes, f"{file_prefix}.html", "text/html")
    with col2:
        try:
            img_bytes = fig.to_image(format="png")
            st.download_button("Descargar PNG", img_bytes, f"{file_prefix}.png", "image/png")
        except:
            st.info("💡 Para descarga PNG instala: `pip install kaleido`")
