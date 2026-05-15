# pages/07_💧_Calidad_y_Vertimientos.py

import os
import sys
import io
import unicodedata
import warnings
import requests

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd

import streamlit as st

# --- 1. CONFIGURACIÓN DE PÁGINA (SIEMPRE PRIMERO) ---
st.set_page_config(page_title="Calidad y Vertimientos", page_icon="💧", layout="wide")
warnings.filterwarnings("ignore")

# --- 📂 IMPORTACIÓN ROBUSTA DE MÓDULOS ---
try:
    from modules import selectors
    from modules.config import Config
    from modules.utils import encender_gemelo_digital
except ImportError:
    # Fallback de rutas por si hay problemas de lectura entre carpetas
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules import selectors
    from modules.config import Config
    from modules.utils import encender_gemelo_digital

try:
    import modules.water_quality as wq
except ImportError:
    # Fallback si el nombre del archivo es diferente en tu estructura
    import water_quality as wq

# ==========================================
# 📂 NUEVO: MENÚ DE NAVEGACIÓN PERSONALIZADO
# ==========================================
# Llama al menú expandible y resalta la página actual
selectors.renderizar_menu_navegacion("Calidad y Vertimientos")

# Encendido automático del Gemelo Digital (Lectura de matrices maestras)
encender_gemelo_digital()

st.title("💧 Demanda, Calidad del Agua y Metabolismo Hídrico")
st.markdown("""
Modelo integral del ciclo hidrosocial: Desde la captación y uso del recurso, hasta el análisis de vertimientos y la autodepuración de los cuerpos de agua receptores.
""")

# ====================================================================
# 🛡️ ESTRUCTURA RAÍZ: PREVENCIÓN DE VARIABLES HUÉRFANAS
# ====================================================================
# Garantizamos que estas variables existan desde el segundo 0
carga_total_anual_ton = 0.0
max_dbo = 0.0
od_minimo = 0.0
max_od_sat = 0.0
oxigeno_salud_pct = 0.0
demanda_m3s = st.session_state.get('demanda_total_m3s', 0.0)

# 🛡️ ESCUDO DE INTERFAZ: Valores por defecto para la pestaña de Escenarios
cobertura_ptar = 15.0
eficiencia_ptar = 80.0
vol_suero = 2000.0
trat_porc = 20.0
trat_ave = 75.0

# ==============================================================================
# 🧽 FUNCIÓN NORMALIZADORA (MATA-TILDES Y ESPACIOS)
# ==============================================================================
def normalizar_texto(texto):
    if pd.isna(texto): return ""
    texto_str = str(texto).lower().strip()
    return unicodedata.normalize('NFKD', texto_str).encode('ascii', 'ignore').decode('utf-8')

# ==============================================================================
# 🔌 CONECTORES A BASES DE DATOS MAESTRAS (SUPABASE DIRECTO)
# ==============================================================================
@st.cache_data(show_spinner=False, ttl=3600)
def cargar_maestros_nube(tipo="vertimientos"):
    import geopandas as gpd
    from supabase import create_client
    
    # Búsqueda exhaustiva de credenciales (La verdadera versión a prueba de balas)
    url_sb = None
    key_sb = None
    if "SUPABASE_URL" in st.secrets:
        url_sb = st.secrets["SUPABASE_URL"]
        key_sb = st.secrets["SUPABASE_KEY"]
    elif "supabase" in st.secrets:
        url_sb = st.secrets["supabase"].get("url") or st.secrets["supabase"].get("SUPABASE_URL")
        key_sb = st.secrets["supabase"].get("key") or st.secrets["supabase"].get("SUPABASE_KEY")
    elif "iri" in st.secrets and "SUPABASE_URL" in st.secrets["iri"]:
        url_sb = st.secrets["iri"]["SUPABASE_URL"]
        key_sb = st.secrets["iri"]["SUPABASE_KEY"]
    elif "connections" in st.secrets and "supabase" in st.secrets["connections"]:
        url_sb = st.secrets["connections"]["supabase"]["SUPABASE_URL"]
        key_sb = st.secrets["connections"]["supabase"]["SUPABASE_KEY"]
        
    if not url_sb or not key_sb:
        st.error("❌ Faltan credenciales de Supabase en secrets.")
        return gpd.GeoDataFrame()
        
    try:
        cliente = create_client(url_sb, key_sb)
        bucket = "sihcli_maestros"
        
        # Seleccionar el archivo correcto según lo que pida la página
        if tipo == "vertimientos": archivo = "Vertimientos_Antioquia_Maestro.geojson"
        else: archivo = "Metabolismo_Hidrico_Antioquia_Maestro.geojson"
            
        rutas_posibles = [f"Puntos_de_interes/{archivo}", archivo]
        
        gdf_maestro = gpd.GeoDataFrame()
        for ruta in rutas_posibles:
            try:
                url_publica = cliente.storage.from_(bucket).get_public_url(ruta)
                gdf_maestro = gpd.read_file(url_publica)
                if not gdf_maestro.empty: break
            except Exception:
                continue
        
        if not gdf_maestro.empty and gdf_maestro.crs != "EPSG:3116":
            gdf_maestro = gdf_maestro.to_crs(epsg=3116)
            
        return gdf_maestro
        
    except Exception as e:
        st.error(f"❌ Error crítico procesando la base de {tipo}: {e}")
        return gpd.GeoDataFrame()
        
from modules.config import Config

@st.cache_data(show_spinner=False, ttl=3600)
def cargar_municipios():
    try:
        # ☁️ NUBE: Leemos el maestro demográfico oficial desde Supabase
        df = pd.read_parquet(Config.POBLACION_MAESTRA_URL)
        if 'departamento' in df.columns: df.rename(columns={'departamento': 'depto_nom'}, inplace=True)
        if not df.empty and 'municipio' in df.columns:
            df.dropna(subset=['municipio'], inplace=True)
            df['municipio'] = df['municipio'].astype(str).str.strip().str.title()
            df['municipio_norm'] = df['municipio'].apply(normalizar_texto)
            return df
    except Exception as e:
        pass
    return pd.DataFrame()

# 🌉 PUENTES DE COMPATIBILIDAD (RECUPERADOS)
@st.cache_data(show_spinner=False)
def cargar_vertimientos():
    import pandas as pd
    gdf = cargar_maestros_nube("vertimientos")
    if gdf.empty: return pd.DataFrame()
    
    df = pd.DataFrame(gdf)
    df['caudal_vert_lps'] = df['Caudal_Lps']
    df['municipio_norm'] = df['Municipio'].apply(normalizar_texto)
    df['tipo_vertimiento'] = df['Tipo_Vertimiento']
    df['car_norm'] = df['Autoridad'].apply(normalizar_texto)
    
    # Extracción de coordenadas
    centroides = gdf.geometry.centroid
    df['coordenada_x'] = centroides.x.fillna(0)
    df['coordenada_y'] = centroides.y.fillna(0)
    
    # 🛡️ EXTIRPACIÓN DE LA COLUMNA ASESINA DE MEMORIA
    if 'geometry' in df.columns:
        df = df.drop(columns=['geometry'])
        
    return df

@st.cache_data(show_spinner=False)
def cargar_concesiones():
    import pandas as pd
    gdf = cargar_maestros_nube("concesiones")
    if gdf.empty: return pd.DataFrame()
    
    df = pd.DataFrame(gdf)
    df['caudal_lps'] = df['Caudal_Lps']
    df['municipio_norm'] = df['Municipio'].apply(normalizar_texto)
    df['tipo_agua'] = df['Tipo_Fuente']
    df['uso_detalle'] = df['Uso_Agua']
    df['estado'] = df['Estado']
    df['car_norm'] = df['Autoridad'].apply(normalizar_texto)
    
    # Extracción de coordenadas
    centroides = gdf.geometry.centroid
    df['coordenada_x'] = centroides.x.fillna(0)
    df['coordenada_y'] = centroides.y.fillna(0)
    
    def clasificar_uso(u):
        u = normalizar_texto(u)
        if any(x in u for x in ['domestico', 'consumo humano', 'abastecimiento']): return 'Doméstico'
        elif any(x in u for x in ['agricola', 'pecuario', 'riego']): return 'Agrícola/Pecuario'
        elif any(x in u for x in ['industrial', 'mineria']): return 'Industrial'
        else: return 'Otros'
        
    df['Sector_Sihcli'] = df['uso_detalle'].apply(clasificar_uso)
    
    # 🛡️ EXTIRPACIÓN DE LA COLUMNA ASESINA DE MEMORIA
    if 'geometry' in df.columns:
        df = df.drop(columns=['geometry'])
        
    return df

@st.cache_data(show_spinner=False, ttl=3600)
def cargar_maestros_pecuarios():
    """Descarga los archivos maestros pecuarios desde Supabase a la memoria RAM"""
    import unicodedata
    urls = {
        "bovinos": f"{Config.BUCKET_MAESTROS}/Censo_Maestro_Bovinos.csv",
        "porcinos": f"{Config.BUCKET_MAESTROS}/Censo_Maestro_Porcinos.csv",
        "aves": f"{Config.BUCKET_MAESTROS}/Censo_Maestro_Aves.csv"
    }
    dfs = {}
    for key, url in urls.items():
        try:
            df_temp = pd.read_csv(url, encoding='utf-8-sig', sep=None, engine='python')
            df_temp.columns = df_temp.columns.str.upper().str.replace(' ', '_').str.strip()
            if 'MUNICIPIO' in df_temp.columns:
                df_temp['MUNICIPIO_NORM'] = df_temp['MUNICIPIO'].astype(str).str.upper().apply(
                    lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8')
                )
            dfs[key] = df_temp
        except Exception:
            dfs[key] = pd.DataFrame()
    return dfs

@st.cache_data(show_spinner=False, ttl=86400)
def cargar_territorio_maestro():
    from supabase import create_client
    import io
    import unicodedata
    
    url_sb = st.secrets.get("SUPABASE_URL") or st.secrets.get("supabase", {}).get("url") or st.secrets.get("connections", {}).get("supabase", {}).get("SUPABASE_URL")
    key_sb = st.secrets.get("SUPABASE_KEY") or st.secrets.get("supabase", {}).get("key") or st.secrets.get("connections", {}).get("supabase", {}).get("SUPABASE_KEY")
        
    if not url_sb or not key_sb: return pd.DataFrame()
        
    try:
        cliente = create_client(url_sb, key_sb)
        res = cliente.storage.from_("sihcli_maestros").download("territorio_maestro.xlsx")
        df_territorio = pd.read_excel(io.BytesIO(res))
        
        def normalizar(texto):
            if pd.isna(texto): return ""
            return unicodedata.normalize('NFKD', str(texto).lower().strip()).encode('ascii', 'ignore').decode('utf-8')
            
        df_territorio.columns = df_territorio.columns.str.lower().str.strip()
        if 'municipio' in df_territorio.columns: df_territorio['municipio_norm'] = df_territorio['municipio'].apply(normalizar)
        if 'region' in df_territorio.columns: df_territorio['region'] = df_territorio['region'].astype(str).str.title()
        if 'car' in df_territorio.columns: df_territorio['car'] = df_territorio['car'].astype(str).str.upper()
        return df_territorio
    except Exception as e:
        return pd.DataFrame()

# ---------------------------------------------------------------------
# 🚀 INICIALIZACIÓN DE DATOS MAESTROS (100% CLOUD)
# ---------------------------------------------------------------------
df_mpios = cargar_municipios()                                         
df_concesiones = cargar_concesiones()
df_vertimientos = cargar_vertimientos()
df_territorio = cargar_territorio_maestro()

dict_pecuarios = cargar_maestros_pecuarios()
df_bovinos = dict_pecuarios.get("bovinos", pd.DataFrame())
df_porcinos = dict_pecuarios.get("porcinos", pd.DataFrame())
df_aves = dict_pecuarios.get("aves", pd.DataFrame())

# ==============================================================================
# 🌍 1. SELECTOR ESPACIAL Y RECEPTOR DE CONTEXTO (SÍNTESIS UNIFICADA)
# ==============================================================================
from modules import selectors

# El selector nos entrega los IDs, el nombre, la altitud de referencia y el mapa (GeoDataFrame)
ids_seleccionados, nombre_seleccion, altitud_ref, gdf_zona = selectors.render_selector_espacial()

if gdf_zona is None or gdf_zona.empty:
    st.info("👈 Por favor, utiliza el menú lateral para seleccionar un territorio.")
    st.stop() 

# ==============================================================================
# 🧠 2. NÚCLEO DE METABOLISMO UNIFICADO (Sincronía con Llave Universal)
# ==============================================================================
import numpy as np

anio_analisis = 2025 
pob_urb_base, pob_rur_base, pob_total = 0, 0, 0
total_bovinos, total_porcinos, total_aves = 0, 0, 0
origen_dato = "No Identificado"
metodo_animal = "Sin Datos"

# 1. ENRUTADOR MAESTRO (Sincronizado con Módulo 09)
nivel_req = st.session_state.get('nivel_activo_global', 'NINGUNO')

if nivel_req in ["AH", "ZH", "SZH", "NSS1", "NSS2", "NSS3"]:
    nivel_demo = "Cuenca"
elif "CORPOAMB" in nivel_req.upper() or "CAR" in nivel_req.upper():
    nivel_demo = "CAR"
    nivel_req = "CAR"
else:
    nivel_demo = nivel_req
    
nivel_sel_interno = nivel_demo # Alias compatibilidad visor
nivel_sel_visual = nivel_demo

@st.cache_data(ttl=3600)
def consultar_matriz_sql_calidad(tabla, territorio, nivel, col_nivel="Nivel"):
    try:
        from sqlalchemy import text
        from modules.db_manager import get_engine
        import pandas as pd
        
        engine_sql = get_engine()
        
        # 🚀 BÚSQUEDA BLINDADA: Busca directamente en las columnas
        q = text(f'''
            SELECT * FROM {tabla} 
            WHERE TRIM(UPPER("Territorio")) = UPPER(:t) 
            AND TRIM(UPPER("{col_nivel}")) = UPPER(:n)
        ''')
        df_res = pd.read_sql(q, engine_sql, params={'t': str(territorio).strip(), 'n': str(nivel).strip()})

        if df_res.empty:
            q_fallback = text(f'''
                SELECT * FROM {tabla} 
                WHERE UPPER("Territorio") LIKE UPPER(:t) 
                AND UPPER("{col_nivel}") LIKE UPPER(:n)
            ''')
            df_res = pd.read_sql(q_fallback, engine_sql, params={'t': f"%{str(territorio).strip()}%", 'n': f"%{str(nivel)[:3]}%"})
            
        return df_res
    except Exception as e: 
        return pd.DataFrame()

def proyectar_modelo_calidad(f, anio_obj):
    x_norm = anio_obj - f.get('Año_Base', 2018)
    mod = str(f.get('Modelo_Recomendado', 'Logístico'))
    try:
        if 'Logistico' in mod or 'Logístico' in mod: return f.get('Log_K',0) / (1 + f.get('Log_a',0) * np.exp(-f.get('Log_r',0) * x_norm))
        elif 'Exponencial' in mod: return f.get('Exp_a',0) * np.exp(f.get('Exp_b',0) * x_norm)
        elif 'Lineal' in mod: return f.get('Lin_m',0) * x_norm + f.get('Lin_b',0)
        else: return f.get('Poly_A',0)*(x_norm**3) + f.get('Poly_B',0)*(x_norm**2) + f.get('Poly_C',0)*x_norm + f.get('Poly_D',0)
    except: return 0.0

# --- 2. CONEXIÓN AL MOTOR HUMANO ---
df_demo = consultar_matriz_sql_calidad("matriz_maestra_demografica", nombre_seleccion, nivel_demo, "Nivel")

if not df_demo.empty:
    f_t = df_demo[df_demo['Area'].str.lower().str.contains('tot')]
    f_u = df_demo[df_demo['Area'].str.lower().str.contains('urb|cab')]
    f_r = df_demo[df_demo['Area'].str.lower().str.contains('rur|resto')]
    
    if not f_t.empty: pob_total = max(0.0, proyectar_modelo_calidad(f_t.iloc[0], anio_analisis))
    pob_urb_base = max(0.0, proyectar_modelo_calidad(f_u.iloc[0], anio_analisis)) if not f_u.empty else pob_total * 0.85
    pob_rur_base = max(0.0, proyectar_modelo_calidad(f_r.iloc[0], anio_analisis)) if not f_r.empty else pob_total * 0.15
    origen_dato = f"Matriz SQL Exacta ({nivel_demo})"
    if pob_total == 0: pob_total = pob_urb_base + pob_rur_base
else:
    origen_dato = "Error SQL (Sin Datos)"

# --- 3. CONEXIÓN AL MOTOR PECUARIO (Con Bypass AMVA) ---
# 🚀 FIX: Usamos nivel_req (AH, ZH, SZH) para coincidir con la nueva Matriz Pecuaria
df_pec = consultar_matriz_sql_calidad("matriz_maestra_pecuaria", nombre_seleccion, nivel_req, "Nivel")

# 🚑 RESCATE: Si no encuentra como AH/ZH, intenta con el nivel genérico "Cuenca"
if df_pec.empty and nivel_demo == "Cuenca":
    df_pec = consultar_matriz_sql_calidad("matriz_maestra_pecuaria", nombre_seleccion, "Cuenca", "Nivel")

if not df_pec.empty:
    for _, f in df_pec.iterrows():
        if f['Especie'] == 'Bovinos': total_bovinos = max(0.0, proyectar_modelo_calidad(f, anio_analisis))
        if f['Especie'] == 'Porcinos': total_porcinos = max(0.0, proyectar_modelo_calidad(f, anio_analisis))
        if f['Especie'] == 'Aves': total_aves = max(0.0, proyectar_modelo_calidad(f, anio_analisis))
    metodo_animal = f"Matriz SQL Exacta ({nivel_req})"
else:
    # 🛡️ BYPASS JURISDICCIONAL AMVA
    if nombre_seleccion == "AMVA":
        total_bovinos, total_porcinos, total_aves = 0, 0, 0
        metodo_animal = "Bypass Jurisdicción (Corantioquia asume Rural)"
    else:
        total_bovinos, total_porcinos, total_aves = 0, 0, 0
        metodo_animal = "Error SQL (Sin Datos)"

# 🧠 Sincronización con la Memoria Global para gráficas y simulación
st.session_state['ica_bovinos_calc_met'] = total_bovinos
st.session_state['ica_porcinos_calc_met'] = total_porcinos
st.session_state['ica_aves_calc_met'] = total_aves

# ==============================================================================
# 🎨 3. INTERFAZ DE SÍNTESIS Y CALIBRACIÓN DE CAMPO
# ==============================================================================

with st.expander(f"📍 Contexto Demográfico Base ({anio_analisis}): {nombre_seleccion}", expanded=False):
    
    if "SQL" in origen_dato:
        st.success(f"🧠 **Humanos:** Datos extraídos de la Matriz Maestra. Escala: **{origen_dato.split('(')[1].replace(')','')}**.")
    elif "Aleph" in origen_dato:
        st.info(f"🌐 **Humanos:** Datos inyectados desde la sesión actual.")
    else:
        st.warning(f"⚠️ **Atención:** No se halló modelo matemático humano.")
        
    st.info(f"🐄 **Animales:** {metodo_animal}")
    
    st.markdown("⚙️ **Ajuste Manual de Población (Calibración)**")
    col_p1, col_p2 = st.columns(2)
    with col_p1: 
        pob_urbana = st.number_input("Pob. Urbana (Conectada a Alcantarillado):", min_value=0.0, value=float(pob_urb_base), step=1.0)
        st.caption(f"Bovinos base: {total_bovinos:,.0f}")
    with col_p2: 
        pob_rural = st.number_input("Pob. Rural (Sistemas Individuales/Sépticos):", min_value=0.0, value=float(pob_rur_base), step=1.0)
        st.caption(f"Porcinos base: {total_porcinos:,.0f} | Aves base: {total_aves:,.0f}")
    
    pob_total = pob_urbana + pob_rural

st.success(f"📌 **SÍNTESIS ACTIVA |** 📍 Territorio: **{nombre_seleccion}** | 👥 Humanos: **{pob_total:,.0f}** | 🐄 Bov: **{total_bovinos:,.0f}** | 🐖 Porc: **{total_porcinos:,.0f}** | 🐔 Aves: **{total_aves:,.0f}**")

# --- 🌉 ALIAS DE COMPATIBILIDAD ---
lugar_sel = nombre_seleccion
nivel_sel_interno = "Municipal" if "DANE" in origen_dato else "Cuenca Hidrográfica"
nivel_sel_visual = nivel_sel_interno

# ==============================================================================
# 🚀 FILTRO GEOGRÁFICO AVANZADO (Modo Ultra-Ligero + Bypass Departamental)
# ==============================================================================
with st.spinner(f"Cruzando datos espacialmente con {nombre_seleccion}..."):
    from shapely.geometry import Point
    from shapely.prepared import prep
    import gc

    # 🛑 EL BYPASS: Si seleccionó todo el departamento, no hacemos matemática espacial
    es_todo_antioquia = ("antioquia" in nombre_seleccion.lower() or nivel_sel_interno == "Departamental")

    if not es_todo_antioquia:
        # 🔥 FIX GEOSException: Reparar geometrías inválidas antes de unirlas
        gdf_limpio = gdf_zona.to_crs(epsg=3116).copy()
        gdf_limpio['geometry'] = gdf_limpio.geometry.buffer(0) 
        
        try:
            poligono_zona = gdf_limpio.unary_union.simplify(50)
        except Exception:
            import shapely
            gdf_limpio['geometry'] = gdf_limpio.geometry.apply(
                lambda geom: shapely.validation.make_valid(geom) if not geom.is_valid else geom
            )
            poligono_zona = gdf_limpio.unary_union.simplify(50)
            
        poligono_preparado = prep(poligono_zona)
        minx, miny, maxx, maxy = poligono_zona.bounds
    else:
        minx, miny, maxx, maxy = 0, 0, 2000000, 2000000 
        
    # ---------------------------------------------------------
    # 1. VERTIMIENTOS
    # ---------------------------------------------------------
    if not df_vertimientos.empty:
        cols_v = [c for c in df_vertimientos.columns if c in ['caudal_vert_lps', 'tipo_vertimiento', 'coordenada_x', 'coordenada_y']]
        df_v_light = df_vertimientos[cols_v].copy()

        if es_todo_antioquia:
            df_v = df_v_light.copy()
        else:
            mask_box = (df_v_light['coordenada_x'] >= minx) & (df_v_light['coordenada_x'] <= maxx) & \
                       (df_v_light['coordenada_y'] >= miny) & (df_v_light['coordenada_y'] <= maxy)
            df_v_box = df_v_light[mask_box].copy()

            if not df_v_box.empty:
                mask_exacta = df_v_box.apply(lambda row: poligono_preparado.contains(Point(row['coordenada_x'], row['coordenada_y'])), axis=1)
                df_v = df_v_box[mask_exacta].copy()
            else:
                df_v = pd.DataFrame()
            del df_v_box, mask_box

        del df_v_light
    else:
        df_v = pd.DataFrame()

    # ---------------------------------------------------------
    # 2. CONCESIONES
    # ---------------------------------------------------------
    if not df_concesiones.empty:
        cols_c = [c for c in df_concesiones.columns if c in ['caudal_lps', 'tipo_agua', 'Sector_Sihcli', 'coordenada_x', 'coordenada_y', 'uso_detalle', 'estado']]
        df_c_light = df_concesiones[cols_c].copy()

        if es_todo_antioquia:
            df_c = df_c_light.copy()
        else:
            mask_box = (df_c_light['coordenada_x'] >= minx) & (df_c_light['coordenada_x'] <= maxx) & \
                       (df_c_light['coordenada_y'] >= miny) & (df_c_light['coordenada_y'] <= maxy)
            df_c_box = df_c_light[mask_box].copy()

            if not df_c_box.empty:
                mask_exacta = df_c_box.apply(lambda row: poligono_preparado.contains(Point(row['coordenada_x'], row['coordenada_y'])), axis=1)
                df_c = df_c_box[mask_exacta].copy()
            else:
                df_c = pd.DataFrame()
            del df_c_box, mask_box

        if not df_c.empty:
            def normalizar_fuente(x):
                x_str = str(x).lower()
                if pd.isna(x) or 'no especi' in x_str or 'nan' in x_str: return 'Superficial'
                return 'Subterránea' if 'subterr' in x_str else 'Superficial'

            def normalizar_sector(x):
                x_str = str(x).lower()
                if pd.isna(x) or 'otros' in x_str or 'no registr' in x_str or 'nan' in x_str: return 'Doméstico'
                if 'agr' in x_str or 'pec' in x_str: return 'Agrícola/Pecuario'
                if 'ind' in x_str: return 'Industrial'
                return 'Doméstico'

            df_c['tipo_agua'] = df_c['tipo_agua'].apply(normalizar_fuente)
            df_c['Sector_Sihcli'] = df_c['Sector_Sihcli'].apply(normalizar_sector)

            df_c['caudal_lps'] = pd.to_numeric(df_c['caudal_lps'], errors='coerce').fillna(0.0)
            df_c['caudal_lps'] = df_c['caudal_lps'].apply(lambda x: 0.5 if x <= 0.0 else x)
        
        del df_c_light
    else:
        df_c = pd.DataFrame()

    if not es_todo_antioquia:
        del poligono_zona, poligono_preparado
    gc.collect()
        
# ==============================================================================
# 🐄 MOTOR MATEMÁTICO PECUARIO (Conectado a la Nube - Censo ICA Maestro)
# ==============================================================================
@st.cache_data(show_spinner=False, ttl=3600)
def cargar_maestros_pecuarios():
    """Descarga los 3 archivos maestros desde Supabase a la memoria RAM"""
    import pandas as pd
    import unicodedata
    urls = {
        "bovinos": "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/Censo_Maestro_Bovinos.csv",
        "porcinos": "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/Censo_Maestro_Porcinos.csv",
        "aves": "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/Censo_Maestro_Aves.csv"
    }
    dfs = {}
    for key, url in urls.items():
        try:
            dfs[key] = pd.read_csv(url, encoding='utf-8-sig', sep=None, engine='python')
            if 'MUNICIPIO' in dfs[key].columns:
                dfs[key]['MUNICIPIO_NORM'] = dfs[key]['MUNICIPIO'].astype(str).str.upper().apply(
                    lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8')
                )
        except Exception as e:
            dfs[key] = pd.DataFrame()
    return dfs

@st.cache_data(show_spinner=False, ttl=60)
def obtener_censo_pecuario(nombre_seleccion, nivel_sel_interno, anio_analisis=None):
    """
    Motor Pecuario Inmortal de Respaldo: Conecta a Supabase forzando la actualización en vivo,
    y utiliza un radar global si la escala de búsqueda falla.
    """
    try:
        import unicodedata
        import pandas as pd
        import time
        
        # 🛡️ ESCUDO 1: URL actualizada al nuevo archivo HISTÓRICO
        url_supabase = f"https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/Censo_Pecuario_Historico_Cuencas.csv?t={int(time.time())}"
        df_censo = pd.read_csv(url_supabase)
        
        # 🔥 FILTRO TEMPORAL CLAVE: Relajamos el filtro para evitar DataFrames vacíos si el año no coincide exacto
        if anio_analisis is not None and 'Anio' in df_censo.columns:
            anio_max = df_censo['Anio'].max()
            anio_filtro = min(anio_analisis, anio_max)
            # Buscamos el año más cercano disponible
            anios_disponibles = df_censo['Anio'].unique()
            if anio_filtro not in anios_disponibles:
                anio_filtro = max([a for a in anios_disponibles if a <= anio_filtro] or [min(anios_disponibles)])
            df_censo = df_censo[df_censo['Anio'] == anio_filtro]
        
        # 🛡️ ESCUDO 2: Traductor Universal de Escalas
        mapa_escala = {
            "zona hidrográfica": "Zona_Hidrografica",
            "zona hidrografica": "Zona_Hidrografica",
            "cuenca": "Subcuenca",       
            "subcuenca": "Subcuenca",
            "sistema": "Sistema",
            "municipio": "Municipio_Norm",
            "mpio": "Municipio_Norm"
        }
        
        col_limpia = str(nivel_sel_interno).lower().strip()
        columna_escala = mapa_escala.get(col_limpia, "Subcuenca") 
        
        def normalizar(texto):
            if pd.isna(texto): return ""
            t = str(texto).lower().strip()
            import re
            t = unicodedata.normalize('NFKD', t).encode('ascii', 'ignore').decode('utf-8')
            t = re.sub(r'[^a-z0-9]', ' ', t)
            return " ".join(t.split())
        
        busqueda = normalizar(nombre_seleccion)
        df_filtrado = pd.DataFrame()
        
        # Intento 1: Búsqueda Inteligente en la columna seleccionada
        if columna_escala in df_censo.columns:
            nombres_csv = df_censo[columna_escala].apply(normalizar)
            df_filtrado = df_censo[nombres_csv == busqueda]
            
            if df_filtrado.empty:
                palabra_clave = busqueda.replace("rio", "").replace("quebrada", "").replace("q", "").replace("r", "").strip()
                if palabra_clave:
                    mask = nombres_csv.str.contains(palabra_clave, na=False)
                    df_filtrado = df_censo[mask]
                    
        # 🛡️ ESCUDO 3: RADAR GLOBAL. Si falló todo, buscamos en TODO el mapa.
        if df_filtrado.empty:
            palabra_clave = busqueda.replace("rio", "").replace("quebrada", "").replace("q", "").replace("r", "").strip()
            for col in ['Subcuenca', 'Sistema', 'Zona_Hidrografica', 'Municipio_Norm']:
                if col in df_censo.columns:
                    nombres_csv = df_censo[col].apply(normalizar)
                    mask = nombres_csv.str.contains(palabra_clave, na=False)
                    if mask.any():
                        df_filtrado = df_censo[mask]
                        break 
                        
        if df_filtrado.empty:
            return 0, 0, 0
            
        bov = df_filtrado['Bovinos'].sum()
        por = df_filtrado['Porcinos'].sum()
        ave = df_filtrado['Aves'].sum()
        
        return int(bov), int(por), int(ave)
        
    except Exception as e:
        return 0, 0, 0
        
# Arrays para gráficas de tendencias a futuro (DBO)
anios_evo = np.arange(anio_analisis, anio_analisis + 31)
factor_evo = (1 + 0.015) ** (anios_evo - anio_analisis) # Crecimiento proxy 1.5%
pob_evo = pob_total * factor_evo

tab_demanda, tab_fuentes, tab_dilucion, tab_mitigacion, tab_mapa, tab_sirena, tab_extern, tab_lactosuero = st.tabs([
    "🚰 2. Demanda y Eficiencia",
    "🏭 3. Inventario de Cargas", 
    "🌊 4. Asimilación y Dilución", 
    "🛡️ 5. Escenarios de Mitigación",
    "🗺️ 6. Mapa de Calor (Visor)",
    "📊 7. Explorador Ambiental",
    "⚠️ 8. Externalidades",
    "🥛 9. Economía Circular" 
])

# =====================================================================
# 🧠 EXTRACCIÓN UNIFICADA DEL CEREBRO PECUARIO (Cero Duplicados)
# =====================================================================
# Consultamos la memoria viva una sola vez para toda la página
bov_proj = st.session_state.get('ica_bovinos_calc_met')
por_proj = st.session_state.get('ica_porcinos_calc_met')
ave_proj = st.session_state.get('ica_aves_calc_met')

if bov_proj is not None and por_proj is not None and ave_proj is not None:
    bov_base, por_base, ave_base = bov_proj, por_proj, ave_proj
    mensaje_pecuario = "🧬 **Sincronización Activa:** Usando proyecciones del Modelo Pecuario (Pág 06)."
else:
    # Fallback: Si no hay proyecciones, consultamos la BD una sola vez
    bov_base, por_base, ave_base = obtener_censo_pecuario(nombre_seleccion, nivel_sel_interno, anio_analisis)
    mensaje_pecuario = "📊 **Fallback de Datos:** Usando histórico de la base de datos oficial."

# 🛡️ ESCUDO ANTI-NONE DEFINITIVO (Aplica para todas las pestañas)
bov_base = 0 if bov_base is None else bov_base
por_base = 0 if por_base is None else por_base
ave_base = 0 if ave_base is None else ave_base


# ------------------------------------------------------------------------------
# TAB 1: DEMANDA HÍDRICA Y EFICIENCIA
# ------------------------------------------------------------------------------
with tab_demanda:
    # =====================================================================
    # 🎨 REDISEÑO PREMIUM DE DEMANDA Y EFICIENCIA (ESTILO GERENCIAL)
    # =====================================================================
    st.markdown("### 🚰 Demanda Hídrica, Eficiencia y Formalización")
    st.markdown(f"Modelo de requerimientos hídricos sectoriales y análisis de brechas legales para **{nombre_seleccion}**.")
    
    # --- CONEXIÓN CON DEMOGRAFÍA (MEMORIA GLOBAL) ---
    if 'poblacion_total' in st.session_state:
        pob_total = st.session_state['poblacion_total']
        st.info(f"👥 Población base inyectada desde el Modelo Demográfico: **{pob_total:,.0f} habitantes**.")
    
    st.info(mensaje_pecuario)
    st.divider()

    # 1. RECUPERACIÓN DE VALORES POR DEFECTO PARA LOS INPUTS
    dotacion_def = st.session_state.get('td_dotacion', 120.0)
    perd_dom_def = st.session_state.get('td_perd_dom', 25.0)
    
    # Consumo calculado con las variables unificadas y blindadas
    consumo_animales_lpd = (bov_base * 40) + (por_base * 15) + (ave_base * 0.3)
    q_animales_ls = consumo_animales_lpd / 86400
    
    q_agr_def = st.session_state.get('td_q_agr', float(q_animales_ls + 45.0))
    perd_agr_def = st.session_state.get('td_perd_agr', 30.0)
    
    q_ind_def = st.session_state.get('td_q_ind', 20.0)
    perd_ind_def = st.session_state.get('td_perd_ind', 10.0)
    
    # ⚙️ 2. PANEL DE CALIBRACIÓN (EXPANDER PRINCIPAL)
    with st.expander("⚙️ Calibración de Demandas Sectoriales y Eficiencia", expanded=True):
        col_c1, col_c2, col_c3 = st.columns(3)
        
        with col_c1:
            st.markdown("##### 🏘️ Uso Doméstico")
            dotacion = st.number_input("Dotación Neta (L/hab/d):", value=dotacion_def, step=5.0, key="td_dotacion")
            perd_dom = st.slider("Pérdidas Acueducto (%):", 0.0, 100.0, perd_dom_def, step=1.0, key="td_perd_dom")
            
        with col_c2:
            st.markdown("##### 🌾 Uso Agrícola/Pecuario")
            q_necesario_agr = st.number_input("Demanda Neta Agrícola (L/s):", value=q_agr_def, step=5.0, key="td_q_agr")
            perd_agr = st.slider("Pérdidas Sist. Riego (%):", 0.0, 100.0, perd_agr_def, step=1.0, key="td_perd_agr")
            
        with col_c3:
            st.markdown("##### 🏭 Uso Industrial")
            q_necesario_ind = st.number_input("Demanda Neta Ind. (L/s):", value=q_ind_def, step=2.0, key="td_q_ind")
            perd_ind = st.slider("Pérdidas Industria (%):", 0.0, 100.0, perd_ind_def, step=1.0, key="td_perd_ind")

    # 3. EJECUCIÓN MATEMÁTICA (SILENCIOSA)
    # Doméstico
    q_necesario_dom = (pob_total * dotacion) / 86400
    q_efectivo_dom = q_necesario_dom / (1 - (perd_dom/100)) if perd_dom < 100 else q_necesario_dom
    # Agrícola e Industrial
    q_efectivo_agr = q_necesario_agr / (1 - (perd_agr/100)) if perd_agr < 100 else q_necesario_agr
    q_efectivo_ind = q_necesario_ind / (1 - (perd_ind/100)) if perd_ind < 100 else q_necesario_ind
    
    # Análisis Legal
    q_sup, q_sub, q_legal_agr, q_legal_ind = 0.0, 0.0, 0.0, 0.0
    df_usos_detalle = pd.DataFrame()
    
    if 'df_c' in locals() and not df_c.empty:
        df_filtro_c = df_c.copy()
        df_dom = df_filtro_c[df_filtro_c['Sector_Sihcli'] == 'Doméstico']
        q_sup = df_dom[df_dom['tipo_agua'] == 'Superficial']['caudal_lps'].sum()
        q_sub = df_dom[df_dom['tipo_agua'] == 'Subterránea']['caudal_lps'].sum()
        q_legal_agr = df_filtro_c[df_filtro_c['Sector_Sihcli'] == 'Agrícola/Pecuario']['caudal_lps'].sum()
        q_legal_ind = df_filtro_c[df_filtro_c['Sector_Sihcli'] == 'Industrial']['caudal_lps'].sum()
        
        if 'uso_detalle' in df_filtro_c.columns:
            df_usos_detalle = df_filtro_c.groupby(['uso_detalle', 'tipo_agua'])['caudal_lps'].sum().reset_index()
            df_usos_detalle.rename(columns={'uso_detalle':'Uso Específico', 'tipo_agua':'Fuente', 'caudal_lps':'Caudal (L/s)'}, inplace=True)
            df_usos_detalle = df_usos_detalle.sort_values(by='Caudal (L/s)', ascending=False)
            
    q_concesionado_dom = q_sup + q_sub

    # 🚀 4. KPIs GLOBALES DE DEMANDA (OUTPUTS PROMINENTES)
    st.markdown("### 📊 Tablero de Demandas y Presión Hídrica (Extracción Bruta)")
    k1, k2, k3, k4 = st.columns(4)
    
    k1.metric("Demanda Doméstica", f"{q_efectivo_dom:,.1f} L/s", f"Pérdidas: {(q_efectivo_dom - q_necesario_dom):.1f} L/s", delta_color="inverse")
    k2.metric("Demanda Agropecuaria", f"{q_efectivo_agr:,.1f} L/s")
    k3.metric("Demanda Industrial", f"{q_efectivo_ind:,.1f} L/s")
    
    margen = 0.05 
    estado_dom = "Equilibrio"
    color_estado = "off"
    if q_concesionado_dom > q_efectivo_dom * (1 + margen): 
        estado_dom = "Sobreconcesión"
        color_estado = "inverse"
    elif q_concesionado_dom < q_efectivo_dom * (1 - margen): 
        estado_dom = "Riesgo Subregistro"
        color_estado = "inverse"
        
    k4.metric("Concesión Legal Doméstica", f"{q_concesionado_dom:,.1f} L/s", estado_dom, delta_color=color_estado)

    # ⚖️ 5. ANÁLISIS DE FORMALIZACIÓN (EXPANDER)
    with st.expander("⚖️ Análisis de Formalización y Legalidad", expanded=False):
        c_form1, c_form2 = st.columns([1.5, 1])
        
        with c_form1:
            df_chart = pd.DataFrame([
                {"Categoría": "Demanda Efectiva (Bruta)", "Componente": "Consumo Neto", "Caudal (L/s)": q_necesario_dom},
                {"Categoría": "Demanda Efectiva (Bruta)", "Componente": "Pérdidas de Acueducto", "Caudal (L/s)": (q_efectivo_dom - q_necesario_dom)},
                {"Categoría": "Registro Oficial (Legal)", "Componente": "Concesión Superficial", "Caudal (L/s)": q_sup},
                {"Categoría": "Registro Oficial (Legal)", "Componente": "Concesión Subterránea", "Caudal (L/s)": q_sub}
            ])
            fig_sub = px.bar(df_chart, x="Categoría", y="Caudal (L/s)", color="Componente", color_discrete_map={"Consumo Neto": "#2980b9", "Pérdidas de Acueducto": "#e67e22", "Concesión Superficial": "#3498db", "Concesión Subterránea": "#2ecc71"}, title="Demanda Bruta vs Permisos Otorgados (Sector Doméstico)")
            fig_sub.update_layout(height=350, margin=dict(t=40, b=0, l=0, r=0))
            fig_sub.add_hline(y=q_efectivo_dom, line_dash="dash", line_color="red", annotation_text="Límite Extracción Bruta")
            st.plotly_chart(fig_sub, use_container_width=True)
            
        with c_form2:
            st.markdown("<br>", unsafe_allow_html=True)
            if q_concesionado_dom > q_efectivo_dom * (1 + margen): 
                st.error(f"🔴 **Sobreconcesión:** Otorgado {q_concesionado_dom - q_efectivo_dom:,.1f} L/s por encima de la extracción bruta requerida.")
            elif q_concesionado_dom < q_efectivo_dom * (1 - margen): 
                st.warning(f"⚠️ **Riesgo de Subregistro:** Se requiere evaluar el estado de legalidad de por lo menos {q_efectivo_dom - q_concesionado_dom:,.1f} L/s adicionales que no aparecen formalizados en la corporación.")
            else: 
                st.success(f"✅ **Equilibrio Hídrico:** La concesión ({q_concesionado_dom:,.1f} L/s) cubre perfectamente la demanda y las pérdidas del sistema.")
            
            st.markdown("##### Desglose Oficial")
            st.write(f"- **Superficial Doméstico:** {q_sup:,.2f} L/s")
            st.write(f"- **Subterráneo Doméstico:** {q_sub:,.2f} L/s")
            st.write(f"- **Total Legal Doméstico:** {q_concesionado_dom:,.2f} L/s")

        st.divider()
        st.markdown(f"#### 📋 Consolidado de Todos los Usos Registrados en {nombre_seleccion}")
        if not df_usos_detalle.empty:
            c1, c2 = st.columns([2,1])
            with c1: 
                if len(df_usos_detalle) > 1000:
                    st.caption(f"⚠️ Mostrando muestra de 1,000 registros de un total de {len(df_usos_detalle):,}.")
                    df_view_usos = df_usos_detalle.head(1000).astype(str)
                else:
                    df_view_usos = df_usos_detalle.astype(str)
                st.dataframe(df_view_usos, use_container_width=True)
            with c2:
                csv = df_usos_detalle.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Descargar Desglose Completo (CSV)", data=csv, file_name=f'Usos_{nombre_seleccion}.csv', mime='text/csv')
        else: 
            st.warning(f"⚠️ El territorio **{nombre_seleccion}** no registra datos formales.")

    # 👣 6. HUELLA HÍDRICA TERRITORIAL (EXPANDER WRI)
    with st.expander(f"👣 Huella Hídrica Territorial (Metabolismo de Extracción) en {nombre_seleccion}", expanded=False):
        st.markdown("Consolidación de las demandas brutas efectivas (Doméstica + Agrícola + Industrial) para evaluar el nivel de estrés del territorio.")
        
        caudal_total_efectivo_L_s = q_efectivo_dom + q_efectivo_agr + q_efectivo_ind
        caudal_total_m3_s = caudal_total_efectivo_L_s / 1000
        
        c_h1, c_h2, c_h3, c_h4 = st.columns(4)
        c_h1.metric("💧 Doméstica", f"{q_efectivo_dom:,.1f} L/s")
        c_h2.metric("🌾 Agrícola/Pecuaria", f"{q_efectivo_agr:,.1f} L/s")
        c_h3.metric("🏭 Industrial", f"{q_efectivo_ind:,.1f} L/s")
        c_h4.metric("🌊 Extracción Total Continua", f"{caudal_total_m3_s:,.3f} m³/s", delta_color="inverse")
        
        col_btn1, col_btn2 = st.columns([1, 2])
        with col_btn1:
            if st.button("🚀 Exportar Huella Total al Modelo WRI", use_container_width=True):
                st.session_state['demanda_total_m3s'] = caudal_total_m3_s
                st.success("✅ ¡Dato inyectado en la Memoria Global! Dirígete a 'Sistemas Hídricos' o 'Toma de Decisiones'.")
        with col_btn2:
            st.caption("Al hacer clic, el valor de extracción se convierte en la variable de 'Demanda' en los cálculos de Estrés Hídrico y Resiliencia del sistema.")

# ------------------------------------------------------------------------------
# TAB 2: INVENTARIO DE CARGAS (INTEGRACIÓN CIENTÍFICA CON MÓDULO WQ)
# ------------------------------------------------------------------------------
with tab_fuentes:
    st.header(f"Inventario de Cargas Contaminantes ({anio_analisis})")
    
    # 1. SINCRONIZACIÓN GEOGRÁFICA (ALEPH)
    aleph_activo = False
    if 'aleph_ha_pastos' in st.session_state and 'aleph_territorio_origen' in st.session_state:
        # Usamos normalización para evitar fallos por tildes o mayúsculas
        origen = normalizar_texto(st.session_state['aleph_territorio_origen'])
        destino = normalizar_texto(nombre_seleccion)
        if origen == destino:
            aleph_activo = True

    if aleph_activo:
        st.success(f"🌐 **Conexión Aleph Activa:** Coberturas vegetales para **{nombre_seleccion}** sincronizadas con el satélite.")
    
    # 2. CÁLCULO DINÁMICO DE CARGA DOMÉSTICA (LÍNEA BASE)
    # Extraemos población del Aleph de forma segura
    pob_urbana = st.session_state.get('aleph_pob_urbana', 0)
    pob_rural = st.session_state.get('aleph_pob_rural', 0)
    
    # Usamos las constantes de tu módulo (Convertimos gramos a kg dividiendo por 1000)
    factor_urbano_kg = wq.DBO_HAB_URBANO / 1000.0
    factor_rural_kg = wq.DBO_HAB_RURAL / 1000.0
    
    # --- INPUTS DE USUARIO ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🏘️ Saneamiento")
        cobertura_ptar = st.slider("Cobertura de Tratamiento PTAR Urbana %:", 0, 100, 15)
        eficiencia_ptar = st.slider("Remoción DBO en PTAR %:", 0, 100, 80)
        
    with col2:
        st.markdown("### 🏭 Agroindustria")
        vol_suero = st.number_input("Sueros Lácteos Vertidos (L/día):", min_value=0, value=2000, step=500)
        factor_suero_kg = wq.DBO_SUERO_LACTEO / 1000000.0 # g/m3 a kg/L
        
    with col3:
        st.markdown("### 🍓 Agricultura")
        val_def_papa = float(st.session_state.get('aleph_ha_agricola', 50.0)) if aleph_activo else 50.0
        val_def_pastos = float(st.session_state.get('aleph_ha_pastos', 200.0)) if aleph_activo else 200.0
        ha_papa = st.number_input("Cultivos / Mosaico [Ha]:", min_value=0.0, value=val_def_papa, step=5.0, disabled=aleph_activo)
        ha_pastos = st.number_input("Pastos [Ha]:", min_value=0.0, value=val_def_pastos, step=10.0, disabled=aleph_activo)

    st.markdown("---")
    st.markdown(f"### 🐄🚜 Censo Pecuario Calibrado: **{nombre_seleccion}**")
    
    # Extracción segura de bases pecuarias
    bov_base = st.session_state.get('ica_bovinos_calc_met', 0)
    por_base = st.session_state.get('ica_porcinos_calc_met', 0)
    ave_base = st.session_state.get('ica_aves_calc_met', 0)
    
    # 3. CENSO PECUARIO (Conectado a la Base de Datos)
    col_pec1, col_pec2, col_pec3 = st.columns(3)
    with col_pec1:
        st.subheader("Bovinos")
        cab_bov = st.number_input("Cabezas:", min_value=0, value=int(bov_base), step=100, key="inp_bov")
        sistema_bov = st.radio("Sistema Bovino:", ["Extensivo", "Estabulado"])
        f_vert_bov = 0.8 if "Estabulado" in sistema_bov else 0.15 
        factor_dbo_bov = (wq.DBO_VACA_ORDENO / 1000.0) * f_vert_bov
        
    with col_pec2:
        st.subheader("Porcinos")
        cab_por = st.number_input("Cabezas:", min_value=0, value=int(por_base), step=100, key="inp_por")
        trat_porc = st.slider("Eficiencia Biodigestor %:", 0, 100, 20)
        factor_dbo_porc = (wq.DBO_CERDO_CONFINADO / 1000.0) * (1 - (trat_porc/100))
        
    with col_pec3:
        st.subheader("Aves")
        cab_ave = st.number_input("Aves (Cabezas):", min_value=0, value=int(ave_base), step=1000, key="inp_ave")
        trat_ave = st.slider("Manejo Gallinaza %:", 0, 100, 75)
        factor_dbo_aves = 0.015 * (1 - (trat_ave/100))

    # 4. MOTOR DE CÁLCULO VECTORIAL (Cargas diarias en kg DBO5/día)
    dbo_urbana = pob_urbana * factor_urbano_kg * (1 - (cobertura_ptar/100 * eficiencia_ptar/100))
    dbo_rural = pob_rural * factor_rural_kg
    dbo_suero = vol_suero * factor_suero_kg
    dbo_agricola = (ha_papa + ha_pastos) * 0.8 
    dbo_bovinos = cab_bov * factor_dbo_bov
    dbo_porcinos = cab_por * factor_dbo_porc
    dbo_aves = cab_ave * factor_dbo_aves
    
    carga_total_dbo_dia = dbo_urbana + dbo_rural + dbo_suero + dbo_bovinos + dbo_porcinos + dbo_aves + dbo_agricola
    
    # 5. SINCRONIZACIÓN CON EL ALEPH (Para Tablero Gerencial)
    carga_total_anual_ton = (carga_total_dbo_dia * 365) / 1000.0
    st.session_state['carga_dbo_total_ton'] = float(carga_total_anual_ton)
    
    # Cálculo de concentración de efluente usando variables seguras
    coef_retorno = 0.85
    q_necesario_dom = st.session_state.get('demanda_domestica_lps', 0)
    q_necesario_ind = st.session_state.get('demanda_industrial_lps', 0)
    
    q_efluente_lps = (q_necesario_dom * coef_retorno) + (q_necesario_ind * 0.8) + (vol_suero / 86400) + ((cab_por * 40)/86400)
    conc_efluente_mg_l = (carga_total_dbo_dia * 1_000_000) / (q_efluente_lps * 86400) if q_efluente_lps > 0 else 0
    
    # =====================================================================
    # 🔮 SIMULADOR DINÁMICO DE METABOLISMO (Conectado a Memoria Global)
    # =====================================================================
    st.markdown("---")
    st.subheader("📈 Evolución de la Presión Ambiental (Gemelo Digital)")
    
    if 'df_matriz_demografica' in st.session_state and 'df_matriz_pecuaria' in st.session_state:
        df_demo = st.session_state['df_matriz_demografica']
        df_pecu = st.session_state['df_matriz_pecuaria']
        
        def calcular_curva_v2(fila, anios):
            x_norm = anios - fila['Año_Base']
            modelo = str(fila.get('Modelo_Recomendado', 'Logístico'))
            if 'Logistico' in modelo or 'Logístico' in modelo: return fila['Log_K'] / (1 + fila['Log_a'] * np.exp(-fila['Log_r'] * x_norm))
            elif 'Exponencial' in modelo: return fila['Exp_a'] * np.exp(fila['Exp_b'] * x_norm)
            elif 'Lineal' in modelo: return fila.get('Lin_m',0) * x_norm + fila.get('Lin_b',0)
            else: return fila['Poly_A']*(x_norm**3) + fila['Poly_B']*(x_norm**2) + fila['Poly_C']*x_norm + fila.get('Poly_D',0)

        def obtener_vector_v2(df, territorio, nivel_exacto, categoria, col_cat, anios, val_fallback):
            mask = (df['Territorio'].str.strip().str.upper() == str(territorio).strip().upper()) & \
                   (df[col_cat].str.strip().str.lower() == str(categoria).lower())
            
            mask_nivel = df['Nivel'].str.strip().str.upper() == str(nivel_exacto).strip().upper()
            if not mask_nivel.any() and nivel_exacto in ["AH", "ZH", "SZH"]:
                mask_nivel = df['Nivel'].str.strip().str.upper() == "CUENCA"
                
            filtro = df[mask & mask_nivel]
            if filtro.empty: return np.full(len(anios), val_fallback)
            return np.maximum(0, calcular_curva_v2(filtro.iloc[0], anios))
            
        anio_limite = st.slider("⏳ Horizonte de Simulación:", min_value=2025, max_value=2050, value=2035, step=1)
        anios_vector = np.arange(anio_analisis, anio_limite + 1)
        
        # Necesitamos el nivel exacto, lo extraemos con fallback a "MUNICIPIO"
        nivel_demo = st.session_state.get('mem_nivel_agregacion', 'MUNICIPIO').upper()
        nivel_req = nivel_demo
        
        # Extraemos los vectores evolutivos poblacionales
        v_pob_urbana = obtener_vector_v2(df_demo, nombre_seleccion, nivel_demo, 'urbana', 'Area', anios_vector, pob_urbana)
        v_pob_rural = obtener_vector_v2(df_demo, nombre_seleccion, nivel_demo, 'rural', 'Area', anios_vector, pob_rural)
        v_bovinos = obtener_vector_v2(df_pecu, nombre_seleccion, nivel_req, 'bovinos', 'Especie', anios_vector, cab_bov)
        v_porcinos = obtener_vector_v2(df_pecu, nombre_seleccion, nivel_req, 'porcinos', 'Especie', anios_vector, cab_por)
        v_aves = obtener_vector_v2(df_pecu, nombre_seleccion, nivel_req, 'aves', 'Especie', anios_vector, cab_ave)

        # 5. Motor Bioquímico Vectorial (Usando las constantes oficiales del módulo WQ)
        v_dbo_urbana = v_pob_urbana * factor_urbano_kg * (1 - (cobertura_ptar/100 * eficiencia_ptar/100))
        v_dbo_rural = v_pob_rural * factor_rural_kg
        v_dbo_bovinos = v_bovinos * factor_dbo_bov
        v_dbo_porcinos = v_porcinos * factor_dbo_porc
        v_dbo_aves = v_aves * factor_dbo_aves
        v_dbo_agricola = np.full(len(anios_vector), dbo_agricola)
        v_dbo_suero = np.full(len(anios_vector), dbo_suero)
        
        # 6. Renderizado de Gráficas Comparativas
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.caption(f"**Instantánea Actual ({anio_analisis})**")
            df_cargas_hoy = pd.DataFrame({
                "Fuente": ["Urbana", "Rural", "Agroindustria", "Agricultura", "Bovinos", "Porcinos", "Avicultura"], 
                "DBO_kg_dia": [v_dbo_urbana[0], v_dbo_rural[0], v_dbo_suero[0], v_dbo_agricola[0], v_dbo_bovinos[0], v_dbo_porcinos[0], v_dbo_aves[0]]
            })
            carga_hoy_total = df_cargas_hoy['DBO_kg_dia'].sum()
            fig_bar = px.bar(df_cargas_hoy, x="DBO_kg_dia", y="Fuente", orientation='h', title=f"Aportes de DBO5 ({carga_hoy_total:,.1f} kg/día)", color="Fuente", color_discrete_sequence=px.colors.qualitative.Bold)
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col_g2:
            st.caption(f"**Metabolismo Futuro Dinámico (hasta {anio_limite})**")
            fig_stack = go.Figure()
            fig_stack.add_trace(go.Scatter(x=anios_vector, y=v_dbo_urbana, mode='lines', stackgroup='one', name='Urbana (Afectada por PTAR)', fillcolor='#3498db', line=dict(color='#2980b9')))
            fig_stack.add_trace(go.Scatter(x=anios_vector, y=v_dbo_rural, mode='lines', stackgroup='one', name='Rural', fillcolor='#95a5a6', line=dict(color='#7f8c8d')))
            fig_stack.add_trace(go.Scatter(x=anios_vector, y=v_dbo_suero, mode='lines', stackgroup='one', name='Agroindustria (Suero)', fillcolor='#f1c40f', line=dict(color='#f39c12')))
            fig_stack.add_trace(go.Scatter(x=anios_vector, y=v_dbo_porcinos, mode='lines', stackgroup='one', name='Porcicultura', fillcolor='#e74c3c', line=dict(color='#c0392b')))
            fig_stack.add_trace(go.Scatter(x=anios_vector, y=v_dbo_aves, mode='lines', stackgroup='one', name='Avicultura', fillcolor='#f39c12', line=dict(color='#d35400')))
            fig_stack.add_trace(go.Scatter(x=anios_vector, y=v_dbo_bovinos, mode='lines', stackgroup='one', name='Bovinos', fillcolor='#2ecc71', line=dict(color='#27ae60')))
            fig_stack.add_trace(go.Scatter(x=anios_vector, y=v_dbo_agricola, mode='lines', stackgroup='one', name='Agricultura', fillcolor='#16a085', line=dict(color='#1abc9c')))

            fig_stack.update_layout(
                title=f"Crecimiento Multimodelo de Cargas Contaminantes ({anio_analisis} - {anio_limite})",
                xaxis_title="Año", yaxis_title="DBO5 (kg/día)", hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_stack, use_container_width=True)
            
    else:
        st.warning("⚠️ **Memoria Global Vacía:** No has entrenado los Motores Demográficos y Pecuarios en las páginas 06 y 06a.")
        st.info("Mostrando proyecciones estáticas (Crecimiento lineal básico). Para ver la simulación avanzada, entrena los modelos matemáticos primero.")
        
        # 🔥 EL DATAFRAME RESCATADO: Necesario para el modo estático
        df_cargas = pd.DataFrame({
            "Fuente": ["Urbana", "Rural", "Agroindustria", "Agricultura", "Bovinos", "Porcinos", "Avicultura"], 
            "DBO_kg_dia": [dbo_urbana, dbo_rural, dbo_suero, dbo_agricola, dbo_bovinos, dbo_porcinos, dbo_aves]
        })
        
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig_cargas = px.bar(df_cargas, x="DBO_kg_dia", y="Fuente", orientation='h', title=f"Aportes de DBO5 ({carga_total_dbo_dia:,.1f} kg/día)", color="Fuente", color_discrete_sequence=px.colors.qualitative.Bold)
            st.plotly_chart(fig_cargas, use_container_width=True)
        with col_g2:
            st.subheader("📈 Evolución de Carga Orgánica (Proyectada)")
            # Fallback seguro por si anios_evo y factor_evo no vienen de arriba
            factor_evo_seguro = 1.015 # 1.5% crecimiento lineal manual como fallback
            anios_evo_seguros = np.arange(anio_analisis, 2036)
            
            # Vectorizando el fallback lineal
            crecimiento_pob = pob_urbana * (factor_evo_seguro ** (anios_evo_seguros - anio_analisis))
            dbo_evo = (crecimiento_pob * factor_urbano_kg * (1 - (cobertura_ptar/100 * eficiencia_ptar/100))) + dbo_rural + dbo_suero + dbo_bovinos + dbo_porcinos + dbo_aves + dbo_agricola
            
            fig_dbo_evo = go.Figure()
            fig_dbo_evo.add_trace(go.Scatter(x=anios_evo_seguros, y=dbo_evo, mode='lines', fill='tozeroy', name='Carga DBO (kg/d)', line=dict(color='#e74c3c', width=3)))
            st.plotly_chart(fig_dbo_evo, use_container_width=True)
        
# =====================================================================
# 🌊 MÓDULO AVANZADO: ASIMILACIÓN Y CURVA DE OXÍGENO (STREETER-PHELPS)
# =====================================================================
st.markdown("---")
with st.expander(f"🌊 4. Capacidad de Asimilación del Río Receptor: {nombre_seleccion}", expanded=True):
    st.info("Modelo de Streeter-Phelps: Simula la caída y recuperación del Oxígeno Disuelto (OD) aguas abajo del vertimiento principal de la zona seleccionada.")

import modules.water_quality as wq
from modules.water_quality import calcular_streeter_phelps_multipunto

# -------------------------------------------------------------------------
# 🏔️ MOTOR HIPSOMÉTRICO DINÁMICO (Función Inversa A(H) y Límites Auto)
# -------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def obtener_datos_hipso_dinamicos(nombre_cuenca):
    import geopandas as gpd
    import os
    from modules.analysis import calculate_hypsometric_curve
    from modules.config import Config
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    ruta_cuencas = os.path.join(project_root, 'data', 'SubcuencasAinfluencia.geojson')
    
    try:
        gdf_cuencas = gpd.read_file(ruta_cuencas)
        gdf_zona = gdf_cuencas[gdf_cuencas['SUBC_LBL'] == nombre_cuenca]
        
        if gdf_zona.empty: return None
            
        dem_path = Config.DEM_FILE_PATH
        return calculate_hypsometric_curve(gdf_zona, dem_path=dem_path)
    except Exception:
        return None

def calcular_area_inversa(altitud_vertimiento, res_hipso):
    import numpy as np
    
    if res_hipso and res_hipso.get("source") == "DEM Real":
        elevs = res_hipso["elevations"]
        areas = res_hipso["area_percent"]
        
        # 1. Ajuste Polinómico Inverso: Área en función de la Altitud A(H)
        # Usamos grado 3 para capturar la forma de la cuenca
        coeffs = np.polyfit(elevs, areas, 3)
        poly_ah = np.poly1d(coeffs)
        
        # 2. Cálculo del porcentaje
        pct_area = poly_ah(altitud_vertimiento)
        # Restringimos entre 0% y 100% por seguridad en los extremos del polinomio
        pct_area = np.clip(pct_area, 0.0, 100.0) 
        
        # 3. Construcción de la Ecuación LaTeX visible A(H)
        c3, c2, c1, c0 = coeffs
        ecuacion_latex = rf"A(H) = {c3:.2e}H^3 {c2:+.2e}H^2 {c1:+.2e}H {c0:+.2f}"
        
        fuente = "🏔️ Función Hipsométrica Inversa (Basada en DEM)"
        return pct_area / 100.0, fuente, ecuacion_latex

    # Fallback Analítico si falla el mapa
    alt_max, alt_min = 2800.0, 1000.0
    alt_segura = np.clip(altitud_vertimiento, alt_min, alt_max)
    fraccion_area = (alt_max - alt_segura) / (alt_max - alt_min)
    ecuacion = r"A(H) = \frac{H_{max} - H}{H_{max} - H_{min}}"
    return fraccion_area, "📐 Fallback Analítico", ecuacion

# =========================================================================
# 1. Parámetros Físicos del Río (Interactivos y Visibles)
# =========================================================================
# 1.1 Extraer topografía ANTES de renderizar la interfaz para ajustar límites
nombre_c = nombre_seleccion if nivel_sel_interno == "Cuenca Hidrográfica" else "Generica"
h_min_cuenca, h_max_cuenca, h_med_cuenca = 1000.0, 2800.0, 1500.0

res_hipso_actual = obtener_datos_hipso_dinamicos(nombre_c)

if res_hipso_actual and res_hipso_actual.get("source") == "DEM Real":
    import numpy as np
    elevs_reales = res_hipso_actual["elevations"]
    if len(elevs_reales) > 0:
        h_min_cuenca = float(np.min(elevs_reales))
        h_max_cuenca = float(np.max(elevs_reales))
        h_med_cuenca = float(np.mean(elevs_reales))

# 1.2 Renderizar la interfaz con los valores exactos (Conexión Geomorfológica)
with st.expander("⚙️ Características Físicas y Climáticas del Río", expanded=True):
    # ==============================================================================
    # 🚀 LECTURA TOPOGRÁFICA DESDE LA MATRIZ GEOMORFOLÓGICA
    # ==============================================================================
    df_geomorfo = consultar_matriz_sql_calidad("matriz_hidrogeomorfologica_maestra", nombre_seleccion, nivel_req, "Nivel")
    
    if not df_geomorfo.empty:
        # Extraemos los límites reales generados por el DEM
        h_min_cuenca = float(df_geomorfo.iloc[0].get('H_Minima', 1000))
        h_max_cuenca = float(df_geomorfo.iloc[0].get('H_Maxima', 3000))
        h_med_cuenca = float(df_geomorfo.iloc[0].get('H_Media', 2000))
        # Capturamos los coeficientes para el cálculo de área/caudal
        res_hipso_actual = {
            'c3': float(df_geomorfo.iloc[0].get('Coef_C3', 0)),
            'c2': float(df_geomorfo.iloc[0].get('Coef_C2', 0)),
            'c1': float(df_geomorfo.iloc[0].get('Coef_C1', 0)),
            'c0': float(df_geomorfo.iloc[0].get('Coef_C0', 0))
        }
    else:
        h_min_cuenca, h_max_cuenca, h_med_cuenca = 1000.0, 3000.0, 2000.0
        res_hipso_actual = None

    cr1, cr2, cr3 = st.columns(3)
    
    with cr1:
        st.markdown("##### 📍 Posición y Clima del Vertimiento")
        st.caption(f"**Topografía Real ({nombre_seleccion}):** Mín: {h_min_cuenca:.0f} m | Med: {h_med_cuenca:.0f} m | Máx: {h_max_cuenca:.0f} m")
        
        # 🚀 DETECCIÓN AUTOMÁTICA DEL VERTIMIENTO DE CABECERA
        h_real_vert = None
        caudal_real_lps = 0
        if nivel_sel_interno == "Municipal" and not df_vertimientos.empty:
            v_cabecera = df_vertimientos[
                (df_vertimientos['municipio_norm'] == normalizar_texto(nombre_seleccion)) & 
                (df_vertimientos['tipo_vertimiento'].str.contains('Domestico|Municipal', case=False, na=False))
            ].sort_values(by='caudal_vert_lps', ascending=False)

            if not v_cabecera.empty:
                h_real_vert = float(v_cabecera.iloc[0].get('Altitud_m', h_med_cuenca))
                caudal_real_lps = float(v_cabecera.iloc[0].get('caudal_vert_lps', 0))
                st.success(f"📍 **Tubo de Descarga Detectado:** {caudal_real_lps:.1f} L/s a **{h_real_vert:.0f} msnm**.")

        h_defecto = h_real_vert if h_real_vert else h_med_cuenca
        
        # El slider ahora está blindado por los límites reales de la montaña
        h_descarga = st.number_input(
            "Altitud de Descarga (H):", 
            min_value=float(h_min_cuenca), 
            max_value=float(h_max_cuenca), 
            value=float(h_defecto), 
            step=10.0,
            help="Sincronizado con la matriz geomorfológica maestra."
        )
        
        # 🌊 MOTOR HÍDRICO (CONEXIÓN A MATRIZ HIDROLÓGICA)
        df_hidro = consultar_matriz_sql_calidad("matriz_hidrologica_maestra", nombre_seleccion, nivel_req, "Jerarquia")
        q_medio_matriz, q_min_matriz = 0.0, 0.0
        if not df_hidro.empty:
            q_medio_matriz = float(df_hidro.iloc[0].get('Caudal_Medio_m3s', 0.0))
            q_min_matriz = float(df_hidro.iloc[0].get('Caudal_Minimo_m3s', 0.0))
            
        aleph_q_medio = q_medio_matriz if q_medio_matriz > 0 else st.session_state.get('aleph_q_rio_m3s', 0.0)
        aleph_q_min = q_min_matriz if q_min_matriz > 0 else st.session_state.get('aleph_q_min_m3s', 0.0)
        
        st.markdown("---")
        tipo_escenario = st.radio(
            "🌊 Escenario Hidrológico para Dilución:", 
            ["🏜️ Peor Escenario: Caudal de Estiaje / Mínimo", "🌊 Escenario Promedio: Caudal Medio"], 
            horizontal=True
        )
        
        q_base_cuenca = float(aleph_q_min) if "Estiaje" in tipo_escenario else float(aleph_q_medio)
        if q_base_cuenca <= 0: q_base_cuenca = 1.25 # Fallback
            
        # 🧮 CÁLCULO DE CAUDAL POR ALTITUD (MATEMÁTICA HIPSOMÉTRICA)
        frac_area, fuente_hipso, eq_hipso = calcular_area_inversa(h_descarga, res_hipso_actual)
        q_rio = max(0.01, q_base_cuenca * frac_area)
        
        # 🌡️ MOTOR TÉRMICO (GRADIENTE)
        t_sugerida = 28.0 - (0.006 * h_descarga)
        piso_termico = "Cálido" if h_descarga < 1000 else "Templado" if h_descarga < 2000 else "Frío" if h_descarga < 3000 else "Páramo"
        
        # --- EXHIBICIÓN DE LA CIENCIA ---
        c_m1, c_m2 = st.columns(2)
        with c_m1:
            st.caption(f"📉 {fuente_hipso}")
            st.latex(eq_hipso)
            st.metric("Caudal Local (Q)", f"{q_rio:.2f} m³/s", f"-{100-(frac_area*100):.1f}% vs Salida")
            
        with c_m2:
            st.caption("🌡️ Gradiente Térmico")
            st.latex(r"T = 28 - 0.006 \cdot H")
            st.metric("Temp. Natural", f"{t_sugerida:.1f} °C", f"Piso: {piso_termico}", delta_color="off")
            
        t_agua = st.slider("Temperatura del Agua (°C):", 
                          value=float(t_sugerida), 
                          min_value=float(t_sugerida-5), 
                          max_value=float(t_sugerida+5), step=0.5)
        
    with cr2:
        st.markdown("##### 🌊 Hidráulica")
        v_rio = st.slider("Velocidad del Flujo (m/s):", 0.1, 3.0, 0.5, 0.1)
        h_rio = st.slider("Profundidad Media (m):", 0.2, 5.0, 1.0, 0.2)
        
    with cr3:
        st.markdown("##### 🧪 Condición Inicial")
        od_rio_arriba = st.slider("OD Aguas Arriba (mg/L):", 0.0, 12.0, 7.5, 0.5)
        dist_sim = st.slider("Distancia a Simular (km):", 5, 150, 50, 5)
        
# =========================================================================
# 2. Balance de Masas (Mezcla Río + Vertimiento) y Vertimiento Hipotético
# =========================================================================

# 2.1 Calcular Carga Difusa Base (El "Fondo" del Río)
pob_u = st.session_state.get('aleph_pob_urbana', 0)
pob_r = st.session_state.get('aleph_pob_rural', 0)
bov = st.session_state.get('ica_bovinos_calc_met', 0)
por = st.session_state.get('ica_porcinos_calc_met', 0)
ave = st.session_state.get('ica_aves_calc_met', 0)

# Sincronización oficial con las constantes de wq (Convertido a kg/día)
dbo_hab_u = pob_u * (wq.DBO_HAB_URBANO / 1000.0)
dbo_hab_r = pob_r * (wq.DBO_HAB_RURAL / 1000.0)
dbo_bov = bov * (wq.DBO_VACA_ORDENO / 1000.0)
dbo_por = por * (wq.DBO_CERDO_CONFINADO / 1000.0)
dbo_ave = ave * 0.015  # Constante estándar avícola

# Carga Pecuaria Total y Carga Total de Fondo
dbo_hab_total = dbo_hab_u + dbo_hab_r
dbo_gan = dbo_bov + dbo_por + dbo_ave
carga_difusa_total_kg = dbo_hab_total + dbo_gan

with st.expander(f"🏭 Presión Antrópica y Vertimiento Hipotético en {nombre_seleccion}", expanded=True):
    st.markdown(f"##### 🐄🐖🐔 1. Carga Difusa Base (Cuenca Aguas Arriba de {nombre_seleccion})")
    
    # 🛡️ BISTURÍ: Slider de Calibración de Escorrentía
    factor_escorrentia = st.slider(
        "Atenuación Natural (% de carga difusa que llega al cauce):", 
        min_value=0.5, max_value=30.0, value=5.0, step=0.5,
        help="Los bosques y el suelo actúan como filtro. Ajusta este valor para calibrar la 'DBO de Fondo'."
    ) / 100.0
    
    # Calculamos la DBO de fondo en mg/L. Protegemos contra división por cero y fijamos mínimo 1.0 mg/L
    if q_rio > 0:
        dbo_rio_arriba_fondo = max(1.0, ((carga_difusa_total_kg * factor_escorrentia) * 1000) / (q_rio * 86400))
    else:
        dbo_rio_arriba_fondo = 1.0

    cd1, cd2, cd3 = st.columns(3)
    st.markdown("---")
    st.markdown(f"##### 🧪 2. Portafolio Espacial de Vertimientos en {nombre_seleccion}")
    st.caption("Añade descargas a diferentes altitudes. El simulador modelará la cascada de impactos a lo largo del perfil del río.")
    
    q_ini = caudal_real_lps / 1000 if 'caudal_real_lps' in locals() and caudal_real_lps > 0 else 0.150
    h_defecto = h_real_vert if 'h_real_vert' in locals() and h_real_vert else h_med_cuenca
    
    # Aseguramos que la tabla inicie de manera robusta
    df_descargas_base = pd.DataFrame([
        {"Activo": True, "Fuente": "PTAR Principal", "Altitud (m)": float(h_defecto), "Caudal (m3/s)": q_ini, "DBO (mg/L)": 250.0, "Temp (°C)": 24.0},
        {"Activo": False, "Fuente": "Industria Textil Hipotética", "Altitud (m)": float(max(h_min_cuenca, h_defecto - 50)), "Caudal (m3/s)": 0.050, "DBO (mg/L)": 600.0, "Temp (°C)": 30.0}
    ])
    
    # 🎛️ TABLA EDITABLE CON DIMENSIÓN ALTITUDINAL
    df_portafolio = st.data_editor(
        df_descargas_base, 
        num_rows="dynamic", 
        use_container_width=True,
        column_config={
            "Activo": st.column_config.CheckboxColumn("Activo", default=True),
            "Fuente": st.column_config.TextColumn("Nombre de la Fuente"),
            "Altitud (m)": st.column_config.NumberColumn("Altitud (m)", min_value=float(h_min_cuenca), max_value=float(h_max_cuenca), format="%.0f"),
            "Caudal (m3/s)": st.column_config.NumberColumn("Caudal (m³/s)", min_value=0.0, format="%.3f"),
            "DBO (mg/L)": st.column_config.NumberColumn("DBO (mg/L)", min_value=0.0),
            "Temp (°C)": st.column_config.NumberColumn("Temp (°C)", min_value=0.0)
        }
    )
    
    # 🧠 PRE-PROCESAMIENTO ESPACIAL PARA STREETER-PHELPS
    df_activas = df_portafolio[df_portafolio["Activo"] == True].copy()
    
    # FORZAMOS VALORES NUMÉRICOS (Seguro contra NaNs)
    if not df_activas.empty:
        df_activas["Altitud (m)"] = pd.to_numeric(df_activas["Altitud (m)"], errors='coerce').fillna(h_med_cuenca)
        df_activas["Caudal (m3/s)"] = pd.to_numeric(df_activas["Caudal (m3/s)"], errors='coerce').fillna(0)
        df_activas["DBO (mg/L)"] = pd.to_numeric(df_activas["DBO (mg/L)"], errors='coerce').fillna(0)
        df_activas["Temp (°C)"] = pd.to_numeric(df_activas["Temp (°C)"], errors='coerce').fillna(20)
        
        # Ordenamos las descargas desde la montaña hacia el valle (Mayor a Menor Altitud)
        df_activas = df_activas.sort_values(by="Altitud (m)", ascending=False)
        
        q_vert_total = df_activas["Caudal (m3/s)"].sum()
        carga_puntual_kg = (df_activas["Caudal (m3/s)"] * df_activas["DBO (mg/L)"]).sum() * 86.4
        
        st.info(f"🏔️ **Cascada Topográfica:** {len(df_activas)} descargas ordenadas | Caudal Total: **{q_vert_total:.3f} m³/s** | Carga: **{carga_puntual_kg:,.1f} kg DBO/día**")
        
        # Variables legado por seguridad de compatibilidad
        q_vertimiento = q_vert_total
        t_vertimiento = (df_activas["Caudal (m3/s)"] * df_activas["Temp (°C)"]).sum() / q_vert_total if q_vert_total > 0 else 20
        dbo_vert_mgL = (df_activas["Caudal (m3/s)"] * df_activas["DBO (mg/L)"]).sum() / q_vert_total if q_vert_total > 0 else 0
    else:
        q_vertimiento, t_vertimiento, dbo_vert_mgL, carga_puntual_kg = 0.0, 20.0, 0.0, 0.0
        st.warning("No hay descargas activas o caudales válidos en el portafolio.")
        
# =========================================================================
# 3. Ejecución del Motor Streeter-Phelps en Cascada (Multipunto)
# =========================================================================
paso_simulacion = 0.5  # Resolución de la gráfica en km

# El nuevo motor realiza los balances de masa automáticamente en cada tramo
# Pasamos dbo_rio_arriba_fondo que ahora es coherente con el módulo WQ
df_sag = calcular_streeter_phelps_multipunto(
    q_rio=q_rio, 
    t_rio=t_agua, 
    dbo_rio=dbo_rio_arriba_fondo, 
    od_rio=od_rio_arriba, 
    v_ms=v_rio, 
    H_m=h_rio, 
    dist_max_km=dist_sim, 
    paso_km=paso_simulacion, 
    df_descargas=df_activas, 
    eq_hipso_res=res_hipso_actual
)

# 4. Encontrar el Punto Crítico Global (Seguro contra NaN)
if not df_sag.empty and not df_sag['Oxigeno_Disuelto_mgL'].isna().all():
    idx_min = df_sag['Oxigeno_Disuelto_mgL'].idxmin()
    punto_critico = df_sag.loc[idx_min]
    od_minimo = float(punto_critico['Oxigeno_Disuelto_mgL'])
    km_critico = float(punto_critico['Distancia_km'])
    
    if q_vertimiento > 0:
        st.info(f"🧬 **Física de la Cascada:** El simulador procesó la caída topográfica inyectando {len(df_activas)} vertimientos. El punto más crítico de oxígeno (**{od_minimo:.1f} mg/L**) ocurre en el **Km {km_critico:.1f}**.")
else:
    od_minimo = float(od_rio_arriba)
    km_critico = 0.0
    st.warning("No se pudo calcular la curva de oxígeno. Verifique los datos de la tabla de vertimientos.")

# 5. Dibujar la Curva de Oxígeno
import plotly.graph_objects as go

fig_sag = go.Figure()

fig_sag.add_trace(go.Scatter(
    x=df_sag['Distancia_km'], 
    y=df_sag['Oxigeno_Disuelto_mgL'], 
    mode='lines', 
    name='Oxígeno Disuelto (OD)', 
    line=dict(color='#3498db', width=4)
))

fig_sag.add_trace(go.Scatter(
    x=df_sag['Distancia_km'], 
    y=df_sag['OD_Saturacion'], 
    mode='lines', 
    name='Saturación Ideal', 
    line=dict(color='rgba(52, 152, 219, 0.3)', width=2, dash='dash')
))

fig_sag.add_trace(go.Scatter(
    x=df_sag['Distancia_km'], 
    y=df_sag['Limite_Critico'], 
    mode='lines', 
    name='Límite Fauna Acuática (4 mg/L)', 
    line=dict(color='#e74c3c', width=2, dash='dot')
))

fig_sag.add_trace(go.Scatter(
    x=[km_critico], 
    y=[od_minimo], 
    mode='markers+text', 
    name='Punto Crítico',
    marker=dict(color='red', size=12, symbol='x'),
    text=[f"{od_minimo:.1f} mg/L"],
    textposition="bottom center"
))

# Usamos el valor máximo de saturación calculado por el nuevo motor
max_od_sat = float(df_sag['OD_Saturacion'].max()) if not df_sag.empty else 10.0

fig_sag.update_layout(
    title=f"Perfil Topográfico de Oxígeno Disuelto en Cascada (Streeter-Phelps)",
    xaxis_title="Distancia Aguas Abajo (km)",
    yaxis_title="Concentración (mg/L)",
    hovermode="x unified",
    height=450,
    yaxis=dict(range=[0, max_od_sat + 1])
)

# Mostrar métricas y gráfica de la Cascada
m_r1, m_r2, m_r3 = st.columns(3)

# Buscamos el pico máximo de contaminación en toda la simulación
max_dbo = float(df_sag['DBO_Remanente_mgL'].max()) if not df_sag.empty else 0.0

m_r1.metric("Pico Máx. Contaminación (DBO)", f"{max_dbo:.1f} mg/L")
estado_rio = "⚠️ Riesgo Ecológico" if od_minimo < 4.0 else "✅ Saludable"
m_r2.metric("OD Mínimo Crítico", f"{od_minimo:.1f} mg/L", delta=estado_rio, delta_color="inverse" if od_minimo < 4.0 else "normal")
m_r3.metric("Ubicación del Impacto", f"Km {km_critico:.1f}")

st.plotly_chart(fig_sag, use_container_width=True)

# ==========================================================
# 🧠 ACTUALIZACIÓN DEL ALEPH (Sincronización Total)
# ==========================================================
if 'max_od_sat' in locals() and max_od_sat > 0:
    oxigeno_salud_pct = (od_minimo / max_od_sat) * 100
else:
    oxigeno_salud_pct = 0.0

# Inyección al torrente sanguíneo de la aplicación con recuperación segura 
# para demanda_m3s en caso de que no venga del bloque superior
dem_total = st.session_state.get('demanda_total_m3s', 0.0)

st.session_state['carga_dbo_total_ton'] = float(carga_total_anual_ton) if 'carga_total_anual_ton' in locals() else st.session_state.get('carga_dbo_total_ton', 0.0)
st.session_state['calidad_oxigeno_pct'] = float(oxigeno_salud_pct)
st.session_state['calidad_dbo_salida_mgL'] = float(max_dbo) if 'max_dbo' in locals() else 0.0
st.session_state['demanda_total_m3s'] = float(dem_total)

# 3. Sincronizamos también la contaminación de pozos (Acuíferos)
if 'concentracion_acuifero' in locals():
    st.session_state['calidad_acuifero_mgL'] = float(concentracion_acuifero)

# Mensaje de confirmación silencioso para el usuario
st.toast(f"🧪 Calidad Sincronizada: OD {oxigeno_salud_pct:.1f}% | DBO {st.session_state['calidad_dbo_salida_mgL']:.1f} mg/L", icon="🔄")

# ------------------------------------------------------------------------------
# TAB 4: ESCENARIOS DE MITIGACIÓN (HOLÍSTICOS)
# ------------------------------------------------------------------------------
with tab_mitigacion:
    st.header("🛡️ Simulador de Escenarios de Intervención (CuencaVerde)")
    st.markdown("Combina infraestructura gris (PTAR) con soluciones basadas en la naturaleza y agroecología.")
    
    # --- 🛡️ EXTRACCIÓN SEGURA DE VARIABLES BASE ---
    _perd_dom = perd_dom if 'perd_dom' in locals() else 30.0
    _cob_ptar = cobertura_ptar if 'cobertura_ptar' in locals() else 15.0
    _efi_ptar = eficiencia_ptar if 'eficiencia_ptar' in locals() else 80.0
    _trat_porc = trat_porc if 'trat_porc' in locals() else 20.0
    _trat_ave = trat_ave if 'trat_ave' in locals() else 75.0
    
    _q_nec_dom = q_necesario_dom if 'q_necesario_dom' in locals() else st.session_state.get('demanda_domestica_lps', 100.0)
    _q_efec_dom = q_efectivo_dom if 'q_efectivo_dom' in locals() else (_q_nec_dom / (1 - (_perd_dom/100.0)) if _perd_dom < 100 else _q_nec_dom)
    
    _pob_u = pob_urbana if 'pob_urbana' in locals() else st.session_state.get('aleph_pob_urbana', 0)
    _cab_por = cab_por if 'cab_por' in locals() else st.session_state.get('ica_porcinos_calc_met', 0)
    _cab_ave = cab_ave if 'cab_ave' in locals() else st.session_state.get('ica_aves_calc_met', 0)
    _vol_suero = vol_suero if 'vol_suero' in locals() else 2000.0

    st.subheader("A. Saneamiento Urbano")
    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1: esc_perdidas = st.slider("Fugas Acueducto (%):", 0.0, 100.0, float(max(0, _perd_dom - 10)))
    with col_e2: esc_cobertura = st.slider("Cobertura PTAR Urbana (%):", 0.0, 100.0, float(min(100, _cob_ptar + 30)))
    with col_e3: esc_eficiencia = st.slider("Remoción DBO PTAR (%):", 0.0, 100.0, float(min(100, _efi_ptar + 10)))
        
    st.subheader("B. Intervención Rural y Agroindustrial")
    col_e4, col_e5, col_e6 = st.columns(3)
    with col_e4: esc_biodigestores = st.slider("Cerdos con Biodigestor (%):", 0.0, 100.0, float(min(100.0, _trat_porc + 40.0)))
    with col_e5: esc_gallinaza = st.slider("Manejo de Gallinaza (%):", 0.0, 100.0, float(min(100.0, _trat_ave + 20.0)))
    with col_e6: esc_suero = st.slider("Suero Recuperado (Economía Circular) %:", 0.0, 100.0, 50.0)

    st.divider()
    
    # ====================================================================
    # 🧮 CÁLCULOS DEL ESCENARIO (Integración Científica Módulo WQ)
    # ====================================================================
    q_efectivo_esc = _q_nec_dom / (1 - (esc_perdidas/100.0)) if esc_perdidas < 100 else _q_nec_dom
    
    # Motor Bioquímico (kilogramos diarios)
    dbo_urbana_esc = _pob_u * (wq.DBO_HAB_URBANO / 1000.0) * (1 - (esc_cobertura/100.0 * esc_eficiencia/100.0))
    dbo_porcinos_esc = _cab_por * (wq.DBO_CERDO_CONFINADO / 1000.0) * (1 - (esc_biodigestores/100.0))
    dbo_aves_esc = _cab_ave * 0.015 * (1 - (esc_gallinaza/100.0)) # Factor avícola estándar
    dbo_suero_esc = _vol_suero * (1 - (esc_suero/100.0)) * (wq.DBO_SUERO_LACTEO / 1000000.0)
    
    # Rescatamos las cargas inalteradas
    _dbo_rural = dbo_rural if 'dbo_rural' in locals() else (_pob_u * wq.DBO_HAB_RURAL / 1000.0) # Fallback estimado
    _dbo_bovinos = dbo_bovinos if 'dbo_bovinos' in locals() else 0.0
    _dbo_agricola = dbo_agricola if 'dbo_agricola' in locals() else 0.0
    _carga_base_dia = carga_total_dbo_dia if 'carga_total_dbo_dia' in locals() else (_dbo_rural + dbo_urbana_esc + _dbo_bovinos)
    
    carga_total_esc = dbo_urbana_esc + _dbo_rural + dbo_suero_esc + _dbo_bovinos + dbo_porcinos_esc + dbo_aves_esc + _dbo_agricola
    
    col_er1, col_er2 = st.columns([1, 1.5])
    with col_er1:
        st.metric("Agua extraída de la fuente", f"{q_efectivo_esc:,.1f} L/s", delta=f"{q_efectivo_esc - _q_efec_dom:,.1f} L/s (Agua conservada)", delta_color="inverse")
        st.metric("Materia Orgánica Total al Río", f"{carga_total_esc:,.1f} kg/día", delta=f"{carga_total_esc - _carga_base_dia:,.1f} kg/día (Contaminación evitada)", delta_color="inverse")
    
    with col_er2:
        df_esc = pd.DataFrame({
            "Escenario": ["1. Situación Actual", "1. Situación Actual", "2. Con Proyecto CuencaVerde", "2. Con Proyecto CuencaVerde"],
            "Variable": ["Extracción de Agua (L/s)", "Carga DBO (kg/día)", "Extracción de Agua (L/s)", "Carga DBO (kg/día)"],
            "Valor": [_q_efec_dom, _carga_base_dia, q_efectivo_esc, carga_total_esc]
        })
        fig_esc = px.bar(df_esc, x="Variable", y="Valor", color="Escenario", barmode="group", title="Impacto Integral del Proyecto", color_discrete_sequence=["#e74c3c", "#2ecc71"])
        st.plotly_chart(fig_esc, use_container_width=True)

    # -------------------------------------------------------------------------
    # PORTAFOLIO DE INVERSIÓN (SANEAMIENTO Y SbN) EN CALIDAD
    # -------------------------------------------------------------------------
    st.markdown("---")
    with st.expander("🎯 Portafolio de Inversión Financiera (Mitigación)", expanded=False):
        st.markdown("Simula el costo de alcanzar tus metas de reducción de carga contaminante (DBO/SST) combinando infraestructura gris (PTAR) e infraestructura verde (SbN).")
        
        col_m1, col_m2 = st.columns([1, 2.5])
        with col_m1:
            meta_remocion = st.slider("🎯 Meta de Remoción de Carga (%)", 10.0, 100.0, 85.0, 5.0)
            st.markdown("**Costos Unitarios (Millones COP):**")
            costo_ptar = st.number_input("PTAR Centralizada (por Ton/año removida):", value=150.0, step=10.0)
            costo_stam_calidad = st.number_input("STAM Rural (por Ton/año removida):", value=45.0, step=5.0)
            costo_sbn = st.number_input("SbN/Filtros Verdes (por Ton/año removida):", value=12.0, step=2.0)
        
        with col_m2:
            # 🚀 FIX: Leemos la llave correcta del Aleph para la carga actual
            carga_actual = st.session_state.get('carga_dbo_total_ton', 0.0)
            if carga_actual == 0.0: 
                carga_actual = 1000.0 # Fallback visual si no hay cálculos previos
                
            carga_a_remover = carga_actual * (meta_remocion / 100.0)
            
            st.info(f"⚖️ Para cumplir la meta del {meta_remocion}%, debes remover **{carga_a_remover:,.1f} Toneladas/año** del sistema hídrico.")
            
            st.markdown("🎚️ **Mix de Mitigación (% de aporte por estrategia):**")
            c_mix1, c_mix2, c_mix3 = st.columns(3)
            pct_ptar = c_mix1.number_input("% PTAR Urbana", 0, 100, 50)
            pct_stam = c_mix2.number_input("% STAM Rural", 0, 100, 30)
            pct_sbn = c_mix3.number_input("% SbN (Humedales/Riparios)", 0, 100, 20)
            
            if (pct_ptar + pct_stam + pct_sbn) != 100:
                st.error("La suma debe ser exactamente 100%.")
            else:
                ton_ptar = carga_a_remover * (pct_ptar / 100.0)
                ton_stam = carga_a_remover * (pct_stam / 100.0)
                ton_sbn = carga_a_remover * (pct_sbn / 100.0)
                
                inv_ptar = ton_ptar * costo_ptar
                inv_stam = ton_stam * costo_stam_calidad
                inv_sbn = ton_sbn * costo_sbn
                
                c_op1, c_op2, c_op3, c_op4 = st.columns(4)
                c_op1.metric("🏙️ PTAR Urbana", f"{ton_ptar:,.0f} Ton", f"${inv_ptar:,.0f} M")
                c_op2.metric("🏡 STAM Rural", f"{ton_stam:,.0f} Ton", f"${inv_stam:,.0f} M")
                c_op3.metric("🌿 SbN / Humedales", f"{ton_sbn:,.0f} Ton", f"${inv_sbn:,.0f} M")
                c_op4.metric("💰 INVERSIÓN TOTAL", f"${(inv_ptar+inv_stam+inv_sbn):,.0f} M", "Millones COP", delta_color="off")
                
# ------------------------------------------------------------------------------
# TAB 5: MAPA DE CALOR (VISOR ESPACIAL CON FONDO MAPBOX)
# ------------------------------------------------------------------------------
with tab_mapa:
    st.header("🗺️ Mapa de Calor y Análisis Espacial")
    st.info(f"📍 **Enfoque Espacial:** Mostrando datos para **{nivel_sel_interno} - {nombre_seleccion}**.")
    
    var_mapa = st.selectbox("Variable a cartografiar:", [
        "1. Densidad de Puntos de Concesión (Coordenadas)", 
        "2. Densidad de Puntos de Vertimiento (Coordenadas)",
        "3. Cargas Contaminantes DBO Teóricas (Topología por Municipio)",
        "4. Caudal Requerido Teórico (Topología por Municipio)"
    ])
    
    if "Teórica" in var_mapa:
        st.caption("Mapa de calor jerárquico (Treemap) basado en cálculos poblacionales.")
        df_agg = pd.DataFrame()
        
        # 🚀 FIX KEYERROR: Detector dinámico de la columna de año (o fallback si no existe)
        col_anio = next((c for c in ['año', 'Año', 'anio', 'Anio', 'year', 'Year'] if c in df_mpios.columns), None)
        anio_mapa = anio_analisis if 'anio_analisis' in locals() else 2025
        
        if col_anio:
            if anio_mapa not in df_mpios[col_anio].unique():
                anio_mapa = df_mpios[col_anio].max() # Si no existe 2025, usa el más reciente
            df_filtro_anio = df_mpios[df_mpios[col_anio] == anio_mapa]
        else:
            df_filtro_anio = df_mpios # Si el dataset ya está aplanado y no tiene año, lo usamos todo
            
        if nivel_sel_interno == "Nacional (Colombia)": df_m = df_filtro_anio.copy()
        elif nivel_sel_interno == "Jurisdicción Ambiental (CAR)":
            car_norm = normalizar_texto(nombre_seleccion.replace("CAR: ", ""))
            mpios_car = set()
            if not df_concesiones.empty: mpios_car.update(df_concesiones[df_concesiones['car_norm'] == car_norm]['municipio_norm'].unique())
            if not df_vertimientos.empty: mpios_car.update(df_vertimientos[df_vertimientos['car_norm'] == car_norm]['municipio_norm'].unique())
            df_m = df_filtro_anio[df_filtro_anio['municipio_norm'].isin(mpios_car)].copy()
        elif nivel_sel_interno == "Departamental": df_m = df_filtro_anio[df_filtro_anio['depto_nom'] == nombre_seleccion].copy()
        elif nivel_sel_interno == "Regional": df_m = df_filtro_anio[df_filtro_anio['region'] == nombre_seleccion].copy()
        elif nivel_sel_interno == "Municipal": df_m = df_filtro_anio[df_filtro_anio['municipio_norm'] == normalizar_texto(nombre_seleccion)].copy()
        else: df_m = df_filtro_anio.copy()
            
        if not df_m.empty:
            df_agg = df_m.groupby('municipio')['Poblacion'].sum().reset_index()
            df_agg['Poblacion_Proy'] = df_agg['Poblacion'] * 1.15 # Proyección proxy
            
            if "Caudal" in var_mapa:
                df_agg['Valor'] = (df_agg['Poblacion_Proy'] * dotacion) / 86400
                fig_tree = px.treemap(df_agg, path=[px.Constant(nombre_seleccion), 'municipio'], values='Valor', color='Valor', color_continuous_scale='Blues', title="Caudal Doméstico Requerido (L/s)")
            else:
                df_agg['Valor'] = df_agg['Poblacion_Proy'] * 0.050 * (1 - (cobertura_ptar/100 * eficiencia_ptar/100))
                fig_tree = px.treemap(df_agg, path=[px.Constant(nombre_seleccion), 'municipio'], values='Valor', color='Valor', color_continuous_scale='Reds', title="Carga Orgánica DBO (kg/día) aportada por Municipio")
                
            fig_tree.update_traces(textinfo="label+value")
            st.plotly_chart(fig_tree, use_container_width=True)
            
    else:
        st.caption("Mapa de densidad sobre cartografía base (Convierte MAGNA-SIRGAS a WGS84 en tiempo real).")
        
        # 🚀 CORRECCIÓN ESTRUCTURAL: Usamos los datos que ya pasaron por el filtro espacial (df_c y df_v)
        df_map = df_c.copy() if "Concesión" in var_mapa else df_v.copy()
        
        if not df_map.empty:
            # (Hemos eliminado los filtros de texto antiguos, porque df_map ya está recortado geográficamente)
            
            col_z = 'caudal_lps' if "Concesión" in var_mapa else 'caudal_vert_lps'
            df_map['coordenada_x'] = pd.to_numeric(df_map['coordenada_x'], errors='coerce')
            df_map['coordenada_y'] = pd.to_numeric(df_map['coordenada_y'], errors='coerce')
            df_map[col_z] = pd.to_numeric(df_map[col_z], errors='coerce')
            df_map = df_map.dropna(subset=['coordenada_x', 'coordenada_y', col_z])
            
            # ==============================================================================
            # 🛡️ ESCUDO PROTECTOR DE MEMORIA (MUESTREO VISUAL)
            # ==============================================================================
            LIMITE_PUNTOS = 2000
            total_puntos_reales = len(df_map)
            
            if total_puntos_reales > LIMITE_PUNTOS:
                df_map = df_map.sample(n=LIMITE_PUNTOS, random_state=42)
                st.warning(f"🗺️ **Optimización Visual Activada:** Hay {total_puntos_reales:,} puntos en esta zona. Para evitar colapsar la memoria, el mapa dibuja una muestra representativa de {LIMITE_PUNTOS} puntos. (Tus indicadores matemáticos de la parte superior están calculados con el 100% de los datos reales).")
            # ==============================================================================
            
            mask_magna = (df_map['coordenada_x'] > 100000) & (df_map['coordenada_y'] > 100000)
            mask_wgs84 = (df_map['coordenada_x'] < 0) & (df_map['coordenada_x'] > -85) & (df_map['coordenada_y'] > -5)
            
            df_plot = df_map[mask_magna | mask_wgs84].copy()
            
            if not df_plot.empty and df_plot[col_z].sum() > 0:
                try:
                    import pyproj
                    # EPSG:3116 es el origen Magna-Sirgas central
                    transformer = pyproj.Transformer.from_crs("epsg:3116", "epsg:4326", always_xy=True)
                    
                    def to_wgs84(row):
                        if row['coordenada_x'] > 100000: # Es plana
                            lon, lat = transformer.transform(row['coordenada_x'], row['coordenada_y'])
                            return pd.Series({'lon': lon, 'lat': lat})
                        else: # Ya es WGS84
                            return pd.Series({'lon': row['coordenada_x'], 'lat': row['coordenada_y']})
                            
                    with st.spinner("Proyectando coordenadas a satélite..."):
                        coords = df_plot.apply(to_wgs84, axis=1)
                        df_plot['lon'] = coords['lon']
                        df_plot['lat'] = coords['lat']
                        # Limpiar errores de proyección
                        df_plot = df_plot[(df_plot['lon'] >= -85) & (df_plot['lon'] <= -60) & (df_plot['lat'] >= -5) & (df_plot['lat'] <= 15)]
                    
                    fig_dens = px.density_mapbox(df_plot, lat='lat', lon='lon', z=col_z, radius=12,
                                                 center=dict(lat=df_plot['lat'].mean(), lon=df_plot['lon'].mean()),
                                                 zoom=8, mapbox_style="carto-positron", 
                                                 title=f"Densidad Espacial de Caudales (L/s)",
                                                 color_continuous_scale="Viridis")
                    st.plotly_chart(fig_dens, use_container_width=True)
                    
                except ImportError:
                    st.error("💡 Para habilitar el mapa de fondo real, debes instalar 'pyproj'. Usando mapa base temporal.")
                    fig_dens = px.density_contour(df_plot, x="coordenada_x", y="coordenada_y", z=col_z, histfunc="sum", title="Densidad (Sin mapa de fondo)")
                    fig_dens.update_traces(contours_coloring="fill", colorscale="Viridis")
                    st.plotly_chart(fig_dens, use_container_width=True)
            else:
                st.warning("No hay suficientes coordenadas válidas para generar un mapa en esta jurisdicción.")
        else:
            st.warning("No hay base de datos disponible en la zona seleccionada.")

# ------------------------------------------------------------------------------
# TAB 6: EXPLORADOR SIRENA Y VERTIMIENTOS
# ------------------------------------------------------------------------------
with tab_sirena:
    st.header("📊 Explorador Ambiental Avanzado")
    st.info(f"📍 **Contexto Global Activo:** Estás navegando la base de datos bajo la lupa de: **{nivel_sel_interno} - {nombre_seleccion}**.")
    
    if not df_concesiones.empty:
        df_exp = df_concesiones.copy()
        lugar_norm = normalizar_texto(nombre_seleccion.replace("CAR: ", ""))
        
        if nivel_sel_interno == "Jurisdicción Ambiental (CAR)" and 'car_norm' in df_exp.columns: df_exp = df_exp[df_exp['car_norm'] == lugar_norm]
        elif nivel_sel_interno == "Departamental" and 'departamento_norm' in df_exp.columns: df_exp = df_exp[df_exp['departamento_norm'] == normalizar_texto(nombre_seleccion)]
        elif nivel_sel_interno == "Regional" and 'region_norm' in df_exp.columns: df_exp = df_exp[df_exp['region_norm'] == normalizar_texto(nombre_seleccion)]
        elif nivel_sel_interno == "Municipal": df_exp = df_exp[df_exp['municipio_norm'] == lugar_norm]
        
        c_exp1, c_exp2 = st.columns([2, 1.5])
        with c_exp1:
            st.subheader(f"Registros Encontrados: {len(df_exp):,}")
            
            # 🛡️ ESCUDO DE TABLA (Transformamos a texto SOLO las 1000 que vamos a mostrar)
            if len(df_exp) > 1000:
                st.caption("⚠️ **Protección activada:** Se muestran los primeros 1,000 registros para evitar colapsar el navegador. El análisis lateral sí usa el 100% de la base.")
                df_view_exp = df_exp.head(1000).astype(str)
            else:
                df_view_exp = df_exp.astype(str)
                
            st.dataframe(df_view_exp, use_container_width=True)
            
        with c_exp2:
            if not df_exp.empty and df_exp['caudal_lps'].sum() > 0:
                agrupador = st.selectbox("Agrupar Concesiones por:", ["tipo_agua", "Sector_Sihcli", "uso_detalle", "estado"], index=0)
                df_agg = df_exp.groupby(agrupador)['caudal_lps'].sum().reset_index()
                fig_exp = px.pie(df_agg[df_agg['caudal_lps']>0], values='caudal_lps', names=agrupador, hole=0.4, title=f"Total: {df_agg['caudal_lps'].sum():,.1f} L/s")
                fig_exp.update_traces(textposition='inside', textinfo='value+label')
                st.plotly_chart(fig_exp, use_container_width=True)
    else:
        st.warning("No se ha cargado correctamente la base de datos oficial.")

# ------------------------------------------------------------------------------
# TAB 7: ANÁLISIS SECTORIAL DE EXTERNALIDADES (NIVEL AVANZADO)
# ------------------------------------------------------------------------------
with tab_extern:
    st.header("⚠️ Análisis Sectorial de Externalidades Ambientales")
    st.markdown(f"Evaluación avanzada de Huella de Carbono y Balance de Nutrientes (NPK) en **{nombre_seleccion}**.")
    
    # ==============================================================================
    # 1. ESCUDO DE EXTRACCIÓN Y RE-CÁLCULO (Sincronizado con Aleph y WQ)
    # ==============================================================================
    _pob_u = pob_urbana if 'pob_urbana' in locals() else st.session_state.get('aleph_pob_urbana', 0)
    _pob_r = pob_rural if 'pob_rural' in locals() else st.session_state.get('aleph_pob_rural', 0)
    _cab_bov = cab_bov if 'cab_bov' in locals() else st.session_state.get('ica_bovinos_calc_met', 0)
    _cab_por = cab_por if 'cab_por' in locals() else st.session_state.get('ica_porcinos_calc_met', 0)
    _cab_ave = cab_ave if 'cab_ave' in locals() else st.session_state.get('ica_aves_calc_met', 0)
    _ha_papa = ha_papa if 'ha_papa' in locals() else st.session_state.get('aleph_ha_agricola', 50.0)
    _ha_pastos = ha_pastos if 'ha_pastos' in locals() else st.session_state.get('aleph_ha_pastos', 200.0)
    _vol_suero = vol_suero if 'vol_suero' in locals() else 2000.0

    _cob_ptar = cobertura_ptar if 'cobertura_ptar' in locals() else 15.0
    _efi_ptar = eficiencia_ptar if 'eficiencia_ptar' in locals() else 80.0
    _trat_porc = trat_porc if 'trat_porc' in locals() else 20.0
    _trat_ave = trat_ave if 'trat_ave' in locals() else 75.0
    _f_vert_bov = f_vert_bov if 'f_vert_bov' in locals() else 0.15

    # 🧮 CÁLCULO SINCRONIZADO CON EL MÓDULO WQ
    dbo_domestica = _pob_u * (wq.DBO_HAB_URBANO/1000.0) * (1 - (_cob_ptar/100.0 * _efi_ptar/100.0)) + (_pob_r * (wq.DBO_HAB_RURAL/1000.0))
    dbo_bovinos_ext = _cab_bov * (wq.DBO_VACA_ORDENO/1000.0) * _f_vert_bov
    dbo_porcinos_ext = _cab_por * (wq.DBO_CERDO_CONFINADO/1000.0) * (1 - (_trat_porc/100.0))
    dbo_aves_ext = _cab_ave * 0.015 * (1 - (_trat_ave/100.0))
    dbo_agricola_ext = (_ha_papa + _ha_pastos) * 0.8
    dbo_agroind_ext = _vol_suero * (wq.DBO_SUERO_LACTEO/1000000.0)
    
    # 2. MODELO DE NUTRIENTES (Nitrógeno, Fósforo, Potasio)
    # Factores de excreción y escorrentía aproximados (kg/día)
    n_dom = _pob_u * 0.012 * (1 - (_cob_ptar/100.0 * 0.3)) + (_pob_r * 0.012)
    p_dom = _pob_u * 0.003 * (1 - (_cob_ptar/100.0 * 0.2)) + (_pob_r * 0.003)
    k_dom = _pob_u * 0.005 + (_pob_r * 0.005)
    
    n_bov = _cab_bov * 0.15 * 0.2 # Solo el 20% llega a cuerpos de agua (escorrentía)
    p_bov = _cab_bov * 0.04 * 0.2
    k_bov = _cab_bov * 0.12 * 0.2
    
    n_porc = _cab_por * 0.08 * (1 - (_trat_porc/100.0 * 0.5))
    p_porc = _cab_por * 0.02 * (1 - (_trat_porc/100.0 * 0.4))
    k_porc = _cab_por * 0.05 * (1 - (_trat_porc/100.0 * 0.2))

    n_aves = _cab_ave * 0.0015 * (1 - (_trat_ave/100.0 * 0.6))
    p_aves = _cab_ave * 0.0005 * (1 - (_trat_ave/100.0 * 0.6))
    k_aves = _cab_ave * 0.0006 * (1 - (_trat_ave/100.0 * 0.6))    
    
    n_agr = (_ha_papa * 1.5) + (_ha_pastos * 0.5)
    p_agr = (_ha_papa * 0.3) + (_ha_pastos * 0.1)
    k_agr = (_ha_papa * 1.2) + (_ha_pastos * 0.4)

    col_ext1, col_ext2 = st.columns([1, 1.2])
    
    with col_ext1:
        st.subheader("Huella de Carbono (Gases Efecto Invernadero)")
        st.caption("Conversión de Metano (CH4) y Óxido Nitroso (N2O) a Toneladas de CO2 equivalente (tCO2e/año).")
        
        # Factores IPCC: CH4 a CO2e (x 28), N2O a CO2e (x 265)
        # Doméstico: Principalmente CH4 de fosas sépticas y PTAR sin captura.
        co2e_dom = (dbo_domestica * 0.25 * 28 * 365) / 1000 
        # Bovinos: Fermentación entérica (muy alta) + estiércol
        co2e_bov = (_cab_bov * 0.16 * 28 * 365) / 1000 # ~60kg CH4/vaca/año
        # Porcinos: Manejo de estiércol (Lagunas anaerobias generan mucho CH4)
        co2e_porc = (dbo_porcinos_ext * 0.25 * 28 * 365) / 1000
        # Aves: Manejo de Gallinaza (Compost)
        co2e_aves = (dbo_aves_ext * 0.25 * 28 * 365) / 1000
        
        # Agrícola: Emisiones de N2O por fertilización nitrogenada
        co2e_agr = (n_agr * 0.01 * 265 * 365) / 1000 
        
        df_co2e = pd.DataFrame({
        "Sector": ["Saneamiento Doméstico", "Ganadería Bovina", "Porcicultura", "Avicultura", "Agricultura"],
        "tCO2e_año": [co2e_dom, co2e_bov, co2e_porc, co2e_aves, co2e_agr]
        })
        
        st.metric("Emisiones Totales del Territorio", f"{df_co2e['tCO2e_año'].sum():,.0f} Ton CO2e/año", help="Toneladas de Dióxido de Carbono equivalente al año.")
        
        fig_co2 = px.pie(df_co2e, values="tCO2e_año", names="Sector", hole=0.4, title="Distribución de la Huella de Carbono")
        fig_co2.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_co2, use_container_width=True)
        
    with col_ext2:
        st.subheader("Balance de Nutrientes (Eutrofización)")
        st.caption("Cargas diarias de N-P-K aportadas por cada sector a la cuenca.") 
        
        df_nutrientes = pd.DataFrame([
            {"Sector": "Doméstico", "Nutriente": "Nitrógeno (N)", "Carga_kg_dia": n_dom},
            {"Sector": "Doméstico", "Nutriente": "Fósforo (P)", "Carga_kg_dia": p_dom},
            {"Sector": "Doméstico", "Nutriente": "Potasio (K)", "Carga_kg_dia": k_dom},
            {"Sector": "Bovinos", "Nutriente": "Nitrógeno (N)", "Carga_kg_dia": n_bov},
            {"Sector": "Bovinos", "Nutriente": "Fósforo (P)", "Carga_kg_dia": p_bov},
            {"Sector": "Bovinos", "Nutriente": "Potasio (K)", "Carga_kg_dia": k_bov},
            {"Sector": "Porcinos", "Nutriente": "Nitrógeno (N)", "Carga_kg_dia": n_porc},
            {"Sector": "Porcinos", "Nutriente": "Fósforo (P)", "Carga_kg_dia": p_porc},
            {"Sector": "Porcinos", "Nutriente": "Potasio (K)", "Carga_kg_dia": k_porc},
            {"Sector": "Agrícola", "Nutriente": "Nitrógeno (N)", "Carga_kg_dia": n_agr},
            {"Sector": "Agrícola", "Nutriente": "Fósforo (P)", "Carga_kg_dia": p_agr},
            {"Sector": "Agrícola", "Nutriente": "Potasio (K)", "Carga_kg_dia": k_agr},
            {"Sector": "Avicultura", "Nutriente": "Nitrógeno (N)", "Carga_kg_dia": n_aves},
            {"Sector": "Avicultura", "Nutriente": "Fósforo (P)", "Carga_kg_dia": p_aves},
            {"Sector": "Avicultura", "Nutriente": "Potasio (K)", "Carga_kg_dia": k_aves}
        ])
    
        n_total = n_dom + n_bov + n_porc + n_aves + n_agr
        if n_total > 1000:
            st.error(f"🔴 **Riesgo Severo de Eutrofización:** Carga de Nitrógeno crítica ({n_total:,.0f} kg/día). Alta probabilidad de blooms algales e hipoxia en el cuerpo receptor.")
        elif n_total > 300:
            st.warning(f"⚠️ **Riesgo Moderado:** Carga de Nitrógeno de {n_total:,.0f} kg/día. Se requiere monitoreo de calidad del agua.")
        else:
            st.success(f"✅ **Carga Estable:** El aporte de nutrientes ({n_total:,.0f} kg N/día) está dentro de límites asimilables.")

    st.divider()
    st.subheader("📋 Consolidado de Externalidades")
    
    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        # Pivot table para una visualización elegante
        df_pivot = df_nutrientes.pivot(index='Sector', columns='Nutriente', values='Carga_kg_dia').reset_index()
        df_pivot = pd.merge(df_pivot, df_co2e, on='Sector', how='left')
        df_pivot.rename(columns={'tCO2e_año': 'Huella_Carbono (tCO2e/año)'}, inplace=True)
        df_pivot.fillna(0, inplace=True)
        
        # 🛡️ ESCUDO DE TABLA 3
        df_view_pivot = df_pivot.head(1000) if len(df_pivot) > 1000 else df_pivot
        
        st.dataframe(df_view_pivot.style.format({
            "Nitrógeno (N)": "{:,.1f}", "Fósforo (P)": "{:,.1f}", "Potasio (K)": "{:,.1f}", "Huella_Carbono (tCO2e/año)": "{:,.0f}"
        }), use_container_width=True)
        
    with col_t2:
        csv_ext = df_pivot.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Descargar Reporte de Externalidades (CSV)", data=csv_ext, file_name=f"Externalidades_NPK_CO2_{nombre_seleccion}.csv", mime='text/csv')

# ------------------------------------------------------------------------------
# TAB 9: ECONOMÍA CIRCULAR (VALORIZACIÓN DE LACTOSUERO)
# ------------------------------------------------------------------------------
with tab_lactosuero:
    st.header("🥛 Economía Circular: Valorización Industrial de Lactosueros")
    st.markdown(f"Evaluación del potencial tecnológico y mitigación ambiental para la cuenca lechera de **{nombre_seleccion}**.")
    
    # Extraer específicamente las vacas en edad de producción del censo ICA
    vacas_adultas = 0
    _mpios_activos = mpios_activos if 'mpios_activos' in locals() else []
    if 'df_bovinos' in locals() and not df_bovinos.empty and 'HEMBRAS>3AÑOS' in df_bovinos.columns:
        vacas_adultas = int(df_bovinos[df_bovinos['MUNICIPIO_NORM'].isin(_mpios_activos)]['HEMBRAS>3AÑOS'].sum())
    if vacas_adultas == 0: vacas_adultas = int(_cab_bov * 0.45) # Estimado si no hay datos
    
    col_l1, col_l2 = st.columns([1, 1.3])
    with col_l1:
        st.subheader("⚙️ Parámetros de la Industria Quesera")
        vacas_ordeno = st.number_input("Vacas en Ordeño (Hembras > 3 años - ICA):", min_value=0, value=int(vacas_adultas * 0.6), step=100, help="Asume un 60% de hembras adultas en lactancia activa.")
        litros_vaca = st.slider("Producción Media (L/vaca/día):", 5.0, 30.0, 12.0, step=0.5)
        pct_queso = st.slider("% de Leche destinada a Quesería Local:", 0.0, 100.0, 30.0, step=1.0)
        
        leche_total = vacas_ordeno * litros_vaca
        leche_queso = leche_total * (pct_queso / 100.0)
        # La regla de oro: 100L leche -> 10kg queso + 90L suero
        suero_generado = leche_queso * 0.90 
        
        st.markdown("---")
        st.metric("🥛 Leche Acopiada Estimada", f"{leche_total:,.0f} L/día")
        st.metric("🧀 Leche hacia Queserías", f"{leche_queso:,.0f} L/día")
        st.metric("⚠️ Lactosuero Generado", f"{suero_generado:,.0f} L/día", delta="Residuo Altamente Contaminante", delta_color="inverse")
        
        df_suero = pd.DataFrame({
            "Destino": ["Queso (Producto Útil)", "Lactosuero (Residuo/Subproducto)"],
            "Volumen (L/día)": [leche_queso * 0.10, suero_generado]
        })
        fig_suero = px.pie(df_suero, values="Volumen (L/día)", names="Destino", hole=0.5, color_discrete_sequence=["#f1c40f", "#e67e22"])
        fig_suero.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_suero, use_container_width=True)

    with col_l2:
        st.subheader("🔄 Potencial Tecnológico (Transformación a WPC)")
        st.info("El suero contiene proteínas y lactosa de altísimo valor. Mediante plantas de **Ultrafiltración (UF)** y secado, se obtiene Proteína de Suero Concentrada (WPC) comercial.")
        
        # Rendimiento tecnológico: 1000 L de suero generan aprox. 6.5 a 7 kg de WPC al 80%
        kg_proteina = (suero_generado / 1000.0) * 6.8 
        precio_kg_wpc = 8.5 # Precio internacional USD/kg aprox
        ingresos_anuales = (kg_proteina * precio_kg_wpc) * 365
        
        # Impacto Ambiental Evitado (Sincronizado con Módulo WQ)
        dbo_suero_evitada = suero_generado * (wq.DBO_SUERO_LACTEO/1000000.0) # kg/día
        
        c_res1, c_res2 = st.columns(2)
        c_res1.metric("Proteína Extraíble (WPC 80%)", f"{kg_proteina:,.1f} kg/día")
        c_res2.metric("Nuevos Ingresos Potenciales", f"${ingresos_anuales:,.0f} USD/año")
        
        st.success(f"🌱 **Impacto Hídrico Evitado:** Si este suero se procesa en lugar de arrojarse al campo o alcantarillado, el territorio evita la contaminación equivalente a **{dbo_suero_evitada:,.0f} kg de DBO/día**. ¡Esto equivale a las aguas residuales de una ciudad de {(dbo_suero_evitada/(wq.DBO_HAB_URBANO/1000.0)):,.0f} habitantes!")
        
        # Integración con el Aleph: Sincroniza el dato si quieren usarlo en la pestaña de inventario
        if st.button("🔌 Sincronizar suero con el Inventario General (Aleph)"):
            st.session_state['aleph_vol_suero'] = float(suero_generado)
            st.toast("✅ ¡Volumen de suero inyectado en la memoria global de Sihcli-Poter!")

# =========================================================================
# 5. VIAJE AL SUBSUELO: CALIDAD DEL AGUA SUBTERRÁNEA (LIXIVIACIÓN)
# =========================================================================
st.markdown("---")
with st.expander(f"⏬ 5. Vulnerabilidad y Calidad del Acuífero en {nombre_seleccion}", expanded=False):
    st.info("Simula el viaje de la contaminación difusa que escapa de la escorrentía superficial, se infiltra a través del perfil del suelo y llega al nivel freático, afectando eventualmente los nacimientos y el caudal base del río.")

with st.expander("🪨 Filtro del Suelo y Termodinámica de Recarga", expanded=True):
    cg1, cg2, cg3 = st.columns(3)
    
    with cg1:
        st.markdown("##### 🌧️ Recarga Hídrica (El Vehículo)")
        
        # 1. Recuperar datos del Aleph (Mapas Avanzados) o usar default regional
        recarga_mm_default = 350.0 
        if 'aleph_recarga_mm' in st.session_state:
            recarga_mm_default = float(st.session_state['aleph_recarga_mm'])
            
        recarga_anual_mm = st.number_input(
            "Recarga Real (mm/año):", 
            min_value=10.0, max_value=5000.0, value=recarga_mm_default, step=50.0,
            help="Lámina de agua que logra infiltrarse hasta el acuífero."
        )
        
        # 2. Calcular Área Aferente con Geometría Espacial Pura (A prueba de balas)
        if 'gdf_zona' in locals() and gdf_zona is not None and not gdf_zona.empty:
            area_km2 = gdf_zona.to_crs(epsg=3116).area.sum() / 1_000_000.0 # Metros cuadrados a km²
        else:
            area_km2 = 100.0 # Default
            
        st.caption(f"Área aferente de recarga (Cálculo Espacial): **{area_km2:,.1f} km²**")
        volumen_recarga_m3 = recarga_anual_mm * area_km2 * 1000.0
        st.metric("Volumen de Infiltración Anual", f"{volumen_recarga_m3:,.0f} m³")

    with cg2:
        st.markdown("##### 💩 Fracción de Lixiviación")
        factor_lixiviacion = st.slider(
            "% de Carga que llega al Acuífero:", 
            min_value=0.0, max_value=50.0, value=15.0, step=1.0,
            help="El suelo actúa como filtro natural. Arcillas retienen más, suelos arenosos y kársticos dejan pasar más contaminantes."
        ) / 100.0
        
        # 3. Recálculo Seguro de la Carga Difusa
        if 'carga_difusa_total_kg' in locals():
            carga_difusa_dia = carga_difusa_total_kg
        else:
            carga_difusa_dia = dbo_domestica + dbo_bovinos_ext + dbo_porcinos_ext + dbo_aves_ext
            
        carga_difusa_anual_kg = carga_difusa_dia * 365.0
        masa_lixiviada_kg = carga_difusa_anual_kg * factor_lixiviacion
        
        st.metric(
            "Masa Contaminante Infiltrada", 
            f"{masa_lixiviada_kg:,.0f} kg/año", 
            f"Filtro retuvo {(carga_difusa_anual_kg - masa_lixiviada_kg):,.0f} kg", 
            delta_color="normal"
        )

    with cg3:
        st.markdown("##### 🧪 Concentración Freática")
        
        # 4. Mezcla Subterránea: C = Masa / Volumen
        # Convertimos: (kg * 1,000,000 mg/kg) / (m3 * 1000 L/m3) = mg/L
        if volumen_recarga_m3 > 0:
            concentracion_acuifero = (masa_lixiviada_kg * 1000.0) / volumen_recarga_m3
        else:
            concentracion_acuifero = 0.0
            
        # Alertas de Calidad
        if concentracion_acuifero < 3.0:
            estado_pozos = "✅ Seguro / Base Limpia"
            color_alerta = "normal"
        elif concentracion_acuifero < 10.0:
            estado_pozos = "⚠️ Riesgo Moderado"
            color_alerta = "off"
        else:
            estado_pozos = "🚨 Acuífero Contaminado"
            color_alerta = "inverse"
            
        st.metric(
            "Impacto en Pozos/Nacimientos", 
            f"{concentracion_acuifero:.2f} mg/L", 
            estado_pozos, 
            delta_color=color_alerta
        )
        st.caption("Esta concentración es la que emergerá como **Caudal Base** en las épocas de estiaje, afectando el río de forma retardada.")
