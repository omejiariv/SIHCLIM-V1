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
# 🧽 FUNCIONES MAESTRAS DE LIMPIEZA (CENTRALIZADAS)
# ==============================================================================

def normalizar_texto(t):
    if not t or pd.isna(t): return ""
    import unicodedata
    import re
    
    t = str(t).upper().strip()
    
    # 🚀 FIX ESTRUCTURAL CUENCAS: "Rio Aburra - Q. La Iguaná" -> "Q. La Iguaná"
    if "-" in t:
        t = t.split("-")[-1].strip()
        
    t = ''.join(c for c in unicodedata.normalize('NFD', t) if unicodedata.category(c) != 'Mn')
    
    t = re.sub(r'\bQ\.\s*', 'QUEBRADA ', t)
    t = re.sub(r'\bR\.\s*', 'RIO ', t)
    t = re.sub(r'\bCGA\.\s*', 'CIENAGA ', t)
    
    stop_words = [r'\bVEREDA\b', r'\bVDA\.?\b', r'\bSECTOR\b', r'\bCASERIO\b', r'\bCENTRO POBLADO\b', r'\bCP\b', r'\bCORREGIMIENTO\b', r'\bCORREG\b', r'\bCGE\b']
    for word in stop_words: t = re.sub(word, '', t)
        
    rebeldes_mpio = {
        r'\bEL CARMEN DE VIBORAL\b': 'CARMEN DE VIBORAL',
        r'\bSAN VICENTE FERRER\b': 'SAN VICENTE',
        r'\bSAN JOSE DE LA MONTANA\b': 'SAN JOSE DE LA MONTANA',
        r'\bDONMATIAS\b': 'DON MATIAS',
        r'\bSANTAFE DE ANTIOQUIA\b': 'SANTA FE DE ANTIOQUIA',
        r'\bEL SANTUARIO\b': 'SANTUARIO',
        r'\bEL PENOL\b': 'PENOL'
    }
    for regex, reemplazo in rebeldes_mpio.items(): t = re.sub(regex, reemplazo, t)
        
    t = re.sub(r'[^A-Z0-9]', '', t) 
    
    diccionario_final = {
        "BOGOTADC": "BOGOTA", 
        "SANJOSEDECUCUTA": "CUCUTA", 
        "LAGUAJIRA": "GUAJIRA", 
        "VALLE": "VALLEDELCAUCA",
        "RIOAURES": "AURES",   # <--- FIX: Río Aures (Abejorral)
        "RIOBUEY": "BUEY"      # <--- FIX: Río Buey (Abejorral)
    }
    if t in diccionario_final: return diccionario_final[t]
    
    return t
    
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
    
    # Normalización para búsqueda en SQL
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
        # 1. Consulta Demográfica (Búsqueda por MATCH_ID normalizado)
        with engine.connect() as conn:
            # Usamos REGEXP o REPLACE en SQL para asegurar el cruce
            q_demo = text('SELECT * FROM matriz_maestra_demografica')
            df_all = pd.read_sql(q_demo, conn)
            df_all['MATCH'] = df_all['Territorio'].apply(normalizar_texto)
            df_res = df_all[df_all['MATCH'] == nombre_q]
            
            if not df_res.empty:
                for area in ['Urbana', 'Rural', 'Total']:
                    row = df_res[df_res['Area'] == area]
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
