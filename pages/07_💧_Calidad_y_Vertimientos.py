import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import unicodedata
import warnings
from modules.utils import normalizar_texto, leer_csv_robusto
from modules.demografia_tools import render_motor_demografico
import requests
import io
import geopandas as gpd
from modules import selectors

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Calidad y Vertimientos", page_icon="💧", layout="wide")

st.title("💧 Demanda, Calidad del Agua y Metabolismo Hídrico")
st.markdown("""
Modelo integral del ciclo hidrosocial: Simulación de demanda, cargas contaminantes, 
capacidad de asimilación, formalización y visor espacial de calor (Concesiones y Vertimientos).
""")
st.divider()

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
        
@st.cache_data
def cargar_municipios():
    ruta = "data/Pob_mpios_colombia.csv"
    if os.path.exists(ruta):
        df = leer_csv_robusto(ruta)
        if 'departamento' in df.columns: df.rename(columns={'departamento': 'depto_nom'}, inplace=True)
        if not df.empty and 'municipio' in df.columns:
            df.dropna(subset=['municipio'], inplace=True)
            df['municipio'] = df['municipio'].astype(str).str.strip().str.title()
            df['municipio_norm'] = df['municipio'].apply(normalizar_texto)
            return df
    return pd.DataFrame()

@st.cache_data
def cargar_veredas():
    ruta = "data/veredas_Antioquia.xlsx"
    return pd.read_excel(ruta) if os.path.exists(ruta) else pd.DataFrame()

# ==============================================================================
# 🌉 PUENTES DE COMPATIBILIDAD (Para no romper las gráficas existentes)
# ==============================================================================
@st.cache_data(show_spinner=False)
def cargar_vertimientos():
    import pandas as pd
    gdf = cargar_maestros_nube("vertimientos")
    if gdf.empty: return pd.DataFrame()
    
    # Adaptar los nombres de la base maestra a lo que espera la interfaz vieja
    df = pd.DataFrame(gdf)
    df['caudal_vert_lps'] = df['Caudal_Lps']
    df['municipio_norm'] = df['Municipio'].apply(normalizar_texto)
    df['tipo_vertimiento'] = df['Tipo_Vertimiento']
    df['car_norm'] = df['Autoridad'].apply(normalizar_texto)
    
    # EXTRACCIÓN SEGURA DE COORDENADAS (Evita el error de MultiPoints o Nulos)
    centroides = gdf.geometry.centroid
    df['coordenada_x'] = centroides.x.fillna(0)
    df['coordenada_y'] = centroides.y.fillna(0)
    
    return df

@st.cache_data(show_spinner=False)
def cargar_concesiones():
    import pandas as pd
    gdf = cargar_maestros_nube("concesiones")
    if gdf.empty: return pd.DataFrame()
    
    # Adaptar los nombres de la base maestra a lo que espera la interfaz vieja
    df = pd.DataFrame(gdf)
    df['caudal_lps'] = df['Caudal_Lps']
    df['municipio_norm'] = df['Municipio'].apply(normalizar_texto)
    df['tipo_agua'] = df['Tipo_Fuente']
    df['uso_detalle'] = df['Uso_Agua']
    df['estado'] = df['Estado']
    df['car_norm'] = df['Autoridad'].apply(normalizar_texto)
    
    # EXTRACCIÓN SEGURA DE COORDENADAS (Evita el error de MultiPoints o Nulos)
    centroides = gdf.geometry.centroid
    df['coordenada_x'] = centroides.x.fillna(0)
    df['coordenada_y'] = centroides.y.fillna(0)
    
    # Micro-clasificador de uso para las gráficas
    def clasificar_uso(u):
        u = normalizar_texto(u)
        if any(x in u for x in ['domestico', 'consumo humano', 'abastecimiento']): return 'Doméstico'
        elif any(x in u for x in ['agricola', 'pecuario', 'riego']): return 'Agrícola/Pecuario'
        elif any(x in u for x in ['industrial', 'mineria']): return 'Industrial'
        else: return 'Otros'
        
    df['Sector_Sihcli'] = df['uso_detalle'].apply(clasificar_uso)
    return df
    
@st.cache_data
def cargar_censo_bovino():
    ruta_xlsx = "data/censos_ICA/Censo_ICA_Bovinos_2023.xlsx"
    ruta_csv = "data/censos_ICA/Censo_ICA_Bovinos_2023.csv"
    df = pd.DataFrame()
    if os.path.exists(ruta_xlsx): df = pd.read_excel(ruta_xlsx)
    elif os.path.exists(ruta_csv): df = leer_csv_robusto(ruta_csv)
    if not df.empty:
        df.columns = df.columns.str.upper().str.replace(' ', '_').str.strip()
        df['MUNICIPIO_NORM'] = df['MUNICIPIO'].astype(str).apply(normalizar_texto)
    return df

@st.cache_data
def cargar_censo_porcino():
    ruta_xlsx = "data/censos_ICA/Censo_ICA_Porcinos_2023.xlsx"
    ruta_csv = "data/censos_ICA/Censo_ICA_Porcinos_2023.csv"
    df = pd.DataFrame()
    if os.path.exists(ruta_xlsx): df = pd.read_excel(ruta_xlsx)
    elif os.path.exists(ruta_csv): df = leer_csv_robusto(ruta_csv)
    if not df.empty:
        df.columns = df.columns.str.upper().str.replace(' ', '_').str.strip()
        df['MUNICIPIO_NORM'] = df['MUNICIPIO'].astype(str).apply(normalizar_texto)
    return df

@st.cache_data
def cargar_censo_aviar():
    ruta_xlsx = "data/censos_ICA/Censo_ICA_Aves_2025.xlsx"
    ruta_csv = "data/censos_ICA/Censo_ICA_Aves_2025.csv"
    df = pd.DataFrame()
    if os.path.exists(ruta_xlsx): df = pd.read_excel(ruta_xlsx)
    elif os.path.exists(ruta_csv): df = leer_csv_robusto(ruta_csv)
    if not df.empty:
        df.columns = df.columns.str.upper().str.replace(' ', '_').str.strip()
        df['MUNICIPIO_NORM'] = df['MUNICIPIO'].astype(str).apply(normalizar_texto)
    return df

@st.cache_data(show_spinner=False, ttl=86400)
def cargar_territorio_maestro():
    import pandas as pd
    import streamlit as st
    from supabase import create_client
    import io
    import unicodedata
    
    # Búsqueda exhaustiva de credenciales (A prueba de balas)
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
        return pd.DataFrame()
        
    try:
        cliente = create_client(url_sb, key_sb)
        bucket = "sihcli_maestros"
        archivo = "territorio_maestro.xlsx"
        
        # Descargamos el archivo directamente a la memoria de Python
        res = cliente.storage.from_(bucket).download(archivo)
        df_territorio = pd.read_excel(io.BytesIO(res))
        
        # Normalizador interno para cruces exactos
        def normalizar(texto):
            if pd.isna(texto): return ""
            return unicodedata.normalize('NFKD', str(texto).lower().strip()).encode('ascii', 'ignore').decode('utf-8')
            
        # Limpieza y estandarización
        df_territorio.columns = df_territorio.columns.str.lower().str.strip()
        
        if 'municipio' in df_territorio.columns:
            df_territorio['municipio_norm'] = df_territorio['municipio'].apply(normalizar)
            
        if 'region' in df_territorio.columns:
            df_territorio['region'] = df_territorio['region'].astype(str).str.title()
            
        if 'car' in df_territorio.columns:
            df_territorio['car'] = df_territorio['car'].astype(str).str.upper()
            
        return df_territorio
        
    except Exception as e:
        import streamlit as st
        st.error(f"❌ Error cargando territorio_maestro: {e}")
        return pd.DataFrame()

@st.cache_data
def cargar_cuencas_mpios():
    ruta = "data/cuencas_mpios_proporcion.csv"
    if os.path.exists(ruta):
        return leer_csv_robusto(ruta)
    return pd.DataFrame()

# ---------------------------------------------------------------------
# 🚀 INICIALIZACIÓN DE DATOS MAESTROS
# ---------------------------------------------------------------------
df_mpios = cargar_municipios()                                         
df_veredas = cargar_veredas()   

# Llamamos directamente a nuestros puentes de compatibilidad en la Nube
df_concesiones = cargar_concesiones()
df_vertimientos = cargar_vertimientos()
df_territorio = cargar_territorio_maestro()
df_bovinos = cargar_censo_bovino()
df_porcinos = cargar_censo_porcino()
df_aves = cargar_censo_aviar()
df_cuencas_mpios = cargar_cuencas_mpios()

# ==============================================================================
# 🌍 SELECTOR ESPACIAL Y RECEPTOR DE CONTEXTO (SÍNTESIS UNIFICADA)
# ==============================================================================
from modules import selectors

ids_seleccionados, nombre_seleccion, altitud_ref, gdf_zona = selectors.render_selector_espacial()

if gdf_zona is None or gdf_zona.empty:
    st.info("👈 Por favor, utiliza el menú lateral para seleccionar un territorio.")
    st.stop() 

# --- 🛑 DETECTOR DE FUGAS DE MEMORIA (STATE LEAKAGE) ---
lugar_en_memoria = st.session_state.get('aleph_lugar', '')
anio_analisis = st.session_state.get('aleph_anio', 2024)

# LÓGICA UNIFICADA: Si el territorio cambió, la población anterior es BASURA.
if lugar_en_memoria != nombre_seleccion:
    st.session_state['aleph_pob_total'] = 0.0 # Destruimos el dato falso
    lugar_en_memoria = nombre_seleccion # Forzamos la actualización de referencia
    
pob_memoria = float(st.session_state.get('aleph_pob_total', 0.0))

# Si la población es 0 (porque cambiamos de lugar o no se ha calculado), estimamos por área
if pob_memoria <= 0:
    area_km2 = gdf_zona.to_crs(epsg=3116).area.sum() / 1_000_000.0
    pob_total = area_km2 * 65.0  # Densidad rural/mixta base (65 hab/km2)
    pob_total = max(pob_total, 500.0) # Mínimo vital
    origen_dato = "Estimación Geoespacial (Densidad base)"
else:
    pob_total = pob_memoria
    origen_dato = "Motor Demográfico (Memoria)"

# --- INTERFAZ UNIFICADA ---
st.markdown("---")
with st.expander(f"📍 Contexto Territorial y Demográfico: {nombre_seleccion}", expanded=False):
    if pob_memoria > 0:
        st.success(f"🧠 **Contexto Sincronizado:** Analizando **{nombre_seleccion}** con datos del Motor Demográfico.")
    else:
        st.warning(f"⚠️ **Atención:** No hay proyección en memoria para **{nombre_seleccion}**. Se ha realizado una {origen_dato}.")
    
    # 🚀 RECUPERAMOS EL MOTOR DEMOGRÁFICO EN LA PÁGINA
    st.markdown("#### ⚙️ Proyectar Población (Motor Local)")
    from modules.demografia_tools import render_motor_demografico
    render_motor_demografico(lugar_defecto=nombre_seleccion)
    
    st.markdown("---")
    st.markdown("⚙️ **Ajuste Manual de Distribución Espacial**")
    col_p1, col_p2 = st.columns(2)
    with col_p1: 
        pob_urbana = st.number_input("Pob. Urbana (Cabecera):", min_value=0.0, value=float(pob_total * 0.70), step=1.0)
    with col_p2: 
        pob_rural = st.number_input("Pob. Rural (Resto):", min_value=0.0, value=float(pob_total * 0.30), step=1.0)
    
    pob_total = pob_urbana + pob_rural

st.success(f"📌 **SÍNTESIS ACTIVA |** 📍 Territorio: **{nombre_seleccion}** | 📅 Año: **{anio_analisis}** | 👥 Población Base: **{pob_total:,.0f} Hab.**")

# --- 🌉 ALIAS DE COMPATIBILIDAD PARA FUNCIONES ANTIGUAS ---
lugar_sel = nombre_seleccion
nivel_sel_interno = "Cuenca Hidrográfica" if any(x in nombre_seleccion.lower() for x in ['rio', 'río', 'quebrada']) else "Municipal"
nivel_sel_visual = nivel_sel_interno

# ==============================================================================
# 🚀 FILTRO GEOGRÁFICO AVANZADO (Modo Bajo Consumo de RAM)
# ==============================================================================
with st.spinner(f"Cruzando concesiones y vertimientos con la geometría de {nombre_seleccion}..."):
    import geopandas as gpd
    import gc # Importamos el recolector de basura de Python
    
    # Solo cruzamos con la columna geometry de la zona para ahorrar muchísima memoria
    gdf_zona_ligera = gdf_zona[['geometry']].to_crs(epsg=3116)
    
    # ---------------------------------------------------------
    # 1. VERTIMIENTOS
    # ---------------------------------------------------------
    if not df_vertimientos.empty:
        # Filtramos columnas estrictamente necesarias antes de volverlo espacial
        cols_v = [c for c in df_vertimientos.columns if c in ['caudal_vert_lps', 'tipo_vertimiento', 'coordenada_x', 'coordenada_y']]
        df_v_light = df_vertimientos[cols_v].copy()
        
        gdf_v = gpd.GeoDataFrame(df_v_light, geometry=gpd.points_from_xy(df_v_light['coordenada_x'], df_v_light['coordenada_y']), crs="EPSG:3116")
        gdf_v_filtrado = gpd.sjoin(gdf_v, gdf_zona_ligera, how="inner", predicate="intersects")
        df_v = pd.DataFrame(gdf_v_filtrado)
        
        # Limpieza de memoria manual
        del df_v_light, gdf_v, gdf_v_filtrado
    else:
        df_v = pd.DataFrame()

    # ---------------------------------------------------------
    # 2. CONCESIONES
    # ---------------------------------------------------------
    if not df_concesiones.empty:
        # Filtramos columnas estrictamente necesarias
        cols_c = [c for c in df_concesiones.columns if c in ['caudal_lps', 'tipo_agua', 'Sector_Sihcli', 'coordenada_x', 'coordenada_y']]
        df_c_light = df_concesiones[cols_c].copy()
        
        gdf_c = gpd.GeoDataFrame(df_c_light, geometry=gpd.points_from_xy(df_c_light['coordenada_x'], df_c_light['coordenada_y']), crs="EPSG:3116")
        gdf_c_filtrado = gpd.sjoin(gdf_c, gdf_zona_ligera, how="inner", predicate="intersects")
        df_c = pd.DataFrame(gdf_c_filtrado)
        
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
            
        # Limpieza de memoria manual
        del df_c_light, gdf_c, gdf_c_filtrado
    else:
        df_c = pd.DataFrame()

    # Forzamos a Python a vaciar la RAM inmediatamente
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

def obtener_censo_pecuario(nombre_seleccion, nivel_sel, anio_evaluacion):
    """Filtra los maestros por año y por territorio"""
    dfs_ica = cargar_maestros_pecuarios()
    anio_censo = max(2018, min(2025, int(anio_evaluacion)))
    
    def calcular_animales(df_censo, col_tot, mpios_lista=None, df_interseccion=None):
        if df_censo.empty or col_tot not in df_censo.columns: return 0.0
        df_y = df_censo[df_censo['AÑO'] == anio_censo]
        total_animales = 0.0
        
        if df_interseccion is not None:
            for _, row in df_interseccion.iterrows():
                # Normalizamos aquí por si acaso
                import unicodedata
                mpio_n = unicodedata.normalize('NFKD', str(row['Municipio']).upper()).encode('ascii', 'ignore').decode('utf-8')
                pct_area = row['Porcentaje'] / 100.0
                df_m = df_y[df_y['MUNICIPIO_NORM'] == mpio_n]
                if not df_m.empty: total_animales += (pd.to_numeric(df_m[col_tot], errors='coerce').sum() * pct_area)
        elif mpios_lista is not None:
            import unicodedata
            mpios_norm = [unicodedata.normalize('NFKD', str(m).upper()).encode('ascii', 'ignore').decode('utf-8') for m in mpios_lista]
            df_f = df_y[df_y['MUNICIPIO_NORM'].isin(mpios_norm)]
            if not df_f.empty: total_animales = pd.to_numeric(df_f[col_tot], errors='coerce').sum()
        else:
            total_animales = pd.to_numeric(df_y[col_tot], errors='coerce').sum()
            
        return float(total_animales)

    mpios_activos = None
    df_intersec = None
    
    if "Municipal" in nivel_sel: mpios_activos = [nombre_seleccion]
    elif "Cuenca" in nivel_sel and 'df_cuencas_mpios' in globals() and not df_cuencas_mpios.empty: 
        df_intersec = df_cuencas_mpios[df_cuencas_mpios['Subcuenca'] == nombre_seleccion]
    elif nivel_sel in ["Jurisdicción Ambiental (CAR)", "Regional", "Departamental"] and 'df_territorio' in globals() and not df_territorio.empty:
        if nivel_sel == "Jurisdicción Ambiental (CAR)": mpios_activos = df_territorio[df_territorio['car'] == nombre_seleccion.replace("CAR: ", "")]['municipio_norm'].tolist()
        elif nivel_sel == "Departamental": mpios_activos = df_territorio[df_territorio['depto_nom'].astype(str).str.title() == nombre_seleccion]['municipio_norm'].tolist()

    tot_bov = calcular_animales(dfs_ica["bovinos"], "TOTAL_BOVINOS", mpios_activos, df_intersec)
    tot_por = calcular_animales(dfs_ica["porcinos"], "TOTAL_CERDOS", mpios_activos, df_intersec)
    tot_ave = calcular_animales(dfs_ica["aves"], "TOTAL_AVES_CAP_OCUPADA_MAS_AVES_TRASPATIO", mpios_activos, df_intersec)
    
    return tot_bov, tot_por, tot_ave

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

# ------------------------------------------------------------------------------
# TAB 1: DEMANDA HÍDRICA Y EFICIENCIA
# ------------------------------------------------------------------------------
with tab_demanda:
    st.header(f"🚰 Demanda, Eficiencia de Sistemas y Formalización")
    
    # --- CONEXIÓN CON DEMOGRAFÍA (MEMORIA GLOBAL) ---
    # Si la página de Demografía envió un dato, lo usamos. Si no, usamos el del selector espacial actual.
    if 'poblacion_total' in st.session_state:
        pob_total = st.session_state['poblacion_total']
        st.info(f"👥 Población base inyectada desde el Modelo Demográfico: **{pob_total:,.0f} habitantes**.")
    # ------------------------------------------------
    
    col_d1, col_d2 = st.columns([1, 1.5])
    
    with col_d1:
        st.subheader("1. Demanda Teórica (Neto vs Bruto)")
        
        st.markdown("**A. Uso Doméstico**")
        col_d_dom1, col_d_dom2 = st.columns(2)
        with col_d_dom1: dotacion = st.number_input("Dotación Neta (L/hab/d):", value=120.0, step=5.0)
        with col_d_dom2: perd_dom = st.slider("Pérdidas del Acueducto (%):", 0.0, 100.0, 25.0, step=1.0)
        q_necesario_dom = (pob_total * dotacion) / 86400
        q_efectivo_dom = q_necesario_dom / (1 - (perd_dom/100)) if perd_dom < 100 else q_necesario_dom
        col_res1, col_res2 = st.columns(2)
        col_res1.metric("Neto (Necesario)", f"{q_necesario_dom:.2f} L/s")
        col_res2.metric("Bruto (Efectivo)", f"{q_efectivo_dom:.2f} L/s", delta=f"Pérdida: {(q_efectivo_dom - q_necesario_dom):.2f} L/s", delta_color="inverse")
        
        st.markdown("**B. Uso Agrícola / Pecuario**")
        
        # 1. Consultamos los animales en la nube para este territorio y año
        bov_dem, por_dem, ave_dem = obtener_censo_pecuario(nombre_seleccion, nivel_sel_interno, anio_analisis)
        
        # 2. Calculamos cuánto beben (L/día) y lo pasamos a L/s
        consumo_animales_lpd = (bov_dem * 40) + (por_dem * 15) + (ave_dem * 0.3)
        q_animales_ls = consumo_animales_lpd / 86400
        
        col_d_agr1, col_d_agr2 = st.columns(2)
        with col_d_agr1: 
            # Sumamos el agua de los animales a una base de cultivos (Ej. 45 L/s)
            q_necesario_agr = st.number_input("Demanda Neta Agrícola/Pecuaria (L/s):", value=float(q_animales_ls + 45.0), step=5.0)
        with col_d_agr2: 
            perd_agr = st.slider("Pérdidas Sist. de Riego (%):", 0.0, 100.0, 30.0, step=1.0)
            
        q_efectivo_agr = q_necesario_agr / (1 - (perd_agr/100)) if perd_agr < 100 else q_necesario_agr
        st.caption(f"Caudal Bruto Agrícola a captar: **{q_efectivo_agr:.2f} L/s**")
        
        st.markdown("**C. Uso Industrial**")
        col_d_ind1, col_d_ind2 = st.columns(2)
        with col_d_ind1: q_necesario_ind = st.number_input("Demanda Neta Industrial (L/s):", value=20.0, step=2.0)
        with col_d_ind2: perd_ind = st.slider("Pérdidas de Industria (%):", 0.0, 100.0, 10.0, step=1.0)
        q_efectivo_ind = q_necesario_ind / (1 - (perd_ind/100)) if perd_ind < 100 else q_necesario_ind
        st.caption(f"Caudal Bruto Industrial a captar: **{q_efectivo_ind:.2f} L/s**")
        
        st.markdown("---")
        st.subheader("2. Demanda Legal (Autorizada)")
        q_sup, q_sub, q_legal_agr, q_legal_ind = 0.0, 0.0, 0.0, 0.0
        df_usos_detalle = pd.DataFrame()
        
        if not df_concesiones.empty and nombre_seleccion != "N/A":
            lugar_norm = normalizar_texto(nombre_seleccion.replace("CAR: ", ""))
            
            if nivel_sel_interno == "Nacional (Colombia)": df_filtro_c = df_concesiones.copy()
            elif nivel_sel_interno == "Jurisdicción Ambiental (CAR)": df_filtro_c = df_concesiones[df_concesiones['car_norm'] == lugar_norm] if 'car_norm' in df_concesiones.columns else pd.DataFrame()
            elif nivel_sel_interno == "Departamental": df_filtro_c = df_concesiones[df_concesiones['departamento_norm'] == lugar_norm] if 'departamento_norm' in df_concesiones.columns else df_concesiones.copy()
            elif nivel_sel_interno == "Regional": df_filtro_c = df_concesiones[df_concesiones['region_norm'] == lugar_norm] if 'region_norm' in df_concesiones.columns else pd.DataFrame()
            elif nivel_sel_interno == "Municipal": df_filtro_c = df_concesiones[df_concesiones['municipio_norm'] == lugar_norm]
            elif nivel_sel_interno == "Veredal" and 'vereda_norm' in df_concesiones.columns: df_filtro_c = df_concesiones[df_concesiones['vereda_norm'] == lugar_norm]
            # 🌐 NUEVA LÓGICA: CRUCE ESPACIAL PARA CUENCAS
            elif nivel_sel_interno == "Cuenca Hidrográfica" and not df_cuencas_mpios.empty:
                mpios_en_cuenca = df_cuencas_mpios[df_cuencas_mpios['Subcuenca'] == nombre_seleccion]['Municipio'].apply(normalizar_texto).tolist()
                if mpios_en_cuenca:
                    df_filtro_c = df_concesiones[df_concesiones['municipio_norm'].isin(mpios_en_cuenca)]
                else:
                    df_filtro_c = pd.DataFrame()
            else: df_filtro_c = pd.DataFrame()
                
            if not df_filtro_c.empty:
                df_dom = df_filtro_c[df_filtro_c['Sector_Sihcli'] == 'Doméstico']
                q_sup = df_dom[df_dom['tipo_agua'] == 'Superficial']['caudal_lps'].sum()
                q_sub = df_dom[df_dom['tipo_agua'] == 'Subterránea']['caudal_lps'].sum()
                q_legal_agr = df_filtro_c[df_filtro_c['Sector_Sihcli'] == 'Agrícola/Pecuario']['caudal_lps'].sum()
                q_legal_ind = df_filtro_c[df_filtro_c['Sector_Sihcli'] == 'Industrial']['caudal_lps'].sum()
                
                df_usos_detalle = df_filtro_c.groupby(['uso_detalle', 'tipo_agua'])['caudal_lps'].sum().reset_index()
                df_usos_detalle.rename(columns={'uso_detalle':'Uso Específico', 'tipo_agua':'Fuente', 'caudal_lps':'Caudal (L/s)'}, inplace=True)
                df_usos_detalle = df_usos_detalle.sort_values(by='Caudal (L/s)', ascending=False)
                
        q_concesionado_dom = q_sup + q_sub
        st.write(f"- **Superficial Doméstico:** {q_sup:,.2f} L/s")
        st.write(f"- **Subterráneo Doméstico:** {q_sub:,.2f} L/s")
        st.write(f"- **Total Legal Doméstico:** {q_concesionado_dom:,.2f} L/s")
        
    with col_d2:
        st.subheader("📊 Análisis de Formalización (Uso Doméstico)")
        margen = 0.05 
        if q_concesionado_dom > q_efectivo_dom * (1 + margen): st.error(f"🔴 **Sobreconcesión:** Otorgado {q_concesionado_dom - q_efectivo_dom:,.1f} L/s por encima de la extracción bruta requerida.")
        elif q_concesionado_dom < q_efectivo_dom * (1 - margen): st.warning(f"⚠️ **Riesgo de Subregistro:** Se requiere evaluar el estado de legalidad de por lo menos {q_efectivo_dom - q_concesionado_dom:,.1f} L/s adicionales que no aparecen formalizados en la corporación.")
        else: st.success(f"✅ **Equilibrio Hídrico:** La concesión ({q_concesionado_dom:,.1f} L/s) cubre perfectamente la demanda y las pérdidas del sistema.")

        df_chart = pd.DataFrame([
            {"Categoría": "Demanda Efectiva (Bruta)", "Componente": "Consumo Neto", "Caudal (L/s)": q_necesario_dom},
            {"Categoría": "Demanda Efectiva (Bruta)", "Componente": "Pérdidas de Acueducto", "Caudal (L/s)": (q_efectivo_dom - q_necesario_dom)},
            {"Categoría": "Registro Oficial (Legal)", "Componente": "Concesión Superficial", "Caudal (L/s)": q_sup},
            {"Categoría": "Registro Oficial (Legal)", "Componente": "Concesión Subterránea", "Caudal (L/s)": q_sub}
        ])
        fig_sub = px.bar(df_chart, x="Categoría", y="Caudal (L/s)", color="Componente", color_discrete_map={"Consumo Neto": "#2980b9", "Pérdidas de Acueducto": "#e67e22", "Concesión Superficial": "#3498db", "Concesión Subterránea": "#2ecc71"}, title="Demanda Bruta vs Permisos Otorgados")
        fig_sub.add_hline(y=q_efectivo_dom, line_dash="dash", line_color="red", annotation_text="Límite Extracción Bruta")
        st.plotly_chart(fig_sub, use_container_width=True)
        
    st.divider()
    st.subheader("📋 Consolidado de Todos los Usos Registrados")
    if not df_usos_detalle.empty:
        c1, c2 = st.columns([2,1])
        with c1: st.dataframe(df_usos_detalle.astype(str), width="stretch")
        with c2:
            csv = df_usos_detalle.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Descargar Desglose (CSV)", data=csv, file_name=f'Usos_{nombre_seleccion}.csv', mime='text/csv')
    else: st.warning(f"⚠️ El territorio **{nombre_seleccion}** no registra datos formales.")

    # =========================================================================
    # NUEVA SECCIÓN: HUELLA HÍDRICA TERRITORIAL (CONSOLIDADO WRI)
    # =========================================================================
    st.divider()
    st.subheader("👣 Huella Hídrica Territorial (Metabolismo de Extracción)")
    st.markdown("Consolidación de las demandas brutas efectivas (Doméstica + Agrícola + Industrial) para evaluar el nivel de estrés del territorio.")
    
    # 1. Sumamos las demandas efectivas (lo que realmente se saca del río)
    caudal_total_efectivo_L_s = q_efectivo_dom + q_efectivo_agr + q_efectivo_ind
    
    # 2. Convertimos a m3/s para el estándar de los modelos topológicos y WRI
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
# TAB 2: INVENTARIO DE CARGAS (CONECTADO AL ALEPH)
# ------------------------------------------------------------------------------
with tab_fuentes:
    st.header(f"Inventario de Cargas Contaminantes ({anio_analisis})")
    
    # Lógica de recepción del Aleph (Blindada contra mayúsculas/minúsculas)
    aleph_activo = False
    if 'aleph_ha_pastos' in st.session_state and 'aleph_territorio_origen' in st.session_state:
        origen_aleph = normalizar_texto(st.session_state['aleph_territorio_origen'])
        destino_actual = normalizar_texto(nombre_seleccion)
        
        # Si el usuario está analizando el mismo municipio en ambas páginas
        if origen_aleph == destino_actual:
            aleph_activo = True

    if aleph_activo:
        st.success(f"🌐 **Conexión Aleph:** Las áreas agrícolas y de pastos para **{nombre_seleccion}** han sido extraídas automáticamente del modelo satelital de la página de Biodiversidad.")
    
    # Extraer valores del satélite, o usar manuales por defecto
    val_def_papa = float(st.session_state.get('aleph_ha_agricola', 50.0)) if aleph_activo else 50.0
    val_def_pastos = float(st.session_state.get('aleph_ha_pastos', 200.0)) if aleph_activo else 200.0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🏘️ Saneamiento")
        cobertura_ptar = st.slider("Cobertura de Tratamiento PTAR Urbana %:", 0, 100, 15)
        eficiencia_ptar = st.slider("Remoción DBO en PTAR %:", 0, 100, 80)
    with col2:
        st.markdown("### 🏭 Agroindustria")
        vol_suero = st.number_input("Sueros Lácteos Vertidos (L/día):", min_value=0, value=2000, step=500)
    with col3:
        st.markdown("### 🍓 Agricultura")
        ha_papa = st.number_input("Cultivos / Mosaico [Ha]:", min_value=0.0, value=val_def_papa, step=5.0, disabled=aleph_activo)
        ha_pastos = st.number_input("Pastos [Ha]:", min_value=0.0, value=val_def_pastos, step=10.0, disabled=aleph_activo)

    st.markdown("---")
    st.markdown(f"### 🐄🚜 Censo Pecuario ICA (2023) para: **{nombre_seleccion}**")
    
    # MOTOR DE CRUCE TERRITORIAL
    mpios_activos = []
    lugar_n = normalizar_texto(nombre_seleccion.replace("CAR: ", ""))
    if not df_territorio.empty:
        if nivel_sel_interno == "Jurisdicción Ambiental (CAR)": mpios_activos = df_territorio[df_territorio['car'].str.upper() == nombre_seleccion.replace("CAR: ", "").upper()]['municipio_norm'].tolist()
        elif nivel_sel_interno == "Departamental": mpios_activos = df_territorio[df_territorio['depto_nom'].apply(normalizar_texto) == lugar_n]['municipio_norm'].tolist()
        elif nivel_sel_interno == "Regional": mpios_activos = df_territorio[df_territorio['region'].apply(normalizar_texto) == lugar_n]['municipio_norm'].tolist()
        elif nivel_sel_interno == "Municipal": mpios_activos = [lugar_n]
    if not mpios_activos: mpios_activos = [lugar_n]

# ==============================================================================
    # EXTRACCIÓN DE BASES DE DATOS ICA (Ahora conectado a la Nube)
    # ==============================================================================
    # Llamamos a nuestra nueva función inteligente pasándole el año del Aleph
    total_bovinos, total_porcinos, total_aves = obtener_censo_pecuario(nombre_seleccion, nivel_sel_interno, anio_analisis)

    default_trat_porc = 20 # Eficiencia por defecto si no se ajusta manualmente

    # Valores por defecto en caso de que el municipio no tenga datos o esté en 0
    if total_bovinos == 0: total_bovinos = int(pob_rural * 1.5)
    if total_porcinos == 0: total_porcinos = int(pob_rural * 0.8)

    col_pec1, col_pec2, col_pec3 = st.columns(3)
    with col_pec1:
        st.subheader("Sector Bovino")
        cabezas_bovinos = st.number_input("Bovinos (Cabezas):", min_value=0, value=int(total_bovinos), step=100)
        sistema_bovino = st.radio("Sistema Bovino:", ["Extensivo", "Estabulado"])
        factor_dbo_bov = 0.8 if "Estabulado" in sistema_bovino else 0.15 
        
    with col_pec2:
        st.subheader("Sector Porcícola")
        cabezas_porcinos = st.number_input("Porcinos (Cabezas):", min_value=0, value=int(total_porcinos), step=100)
        tratamiento_porc = st.slider("Eficiencia Biodigestor %:", 0, 100, default_trat_porc)
        factor_dbo_porc = 0.150 * (1 - (tratamiento_porc/100))
        
    with col_pec3:
        st.subheader("Sector Avícola")
        cabezas_aves = st.number_input("Aves (Galpones/Cabezas):", min_value=0, value=int(total_aves), step=1000)
        tratamiento_aves = st.slider("Manejo Gallinaza %:", 0, 100, 75)
        factor_dbo_aves = 0.015 * (1 - (tratamiento_aves/100)) # Factor aprox 15g DBO/ave

    # --- CÁLCULOS DE CARGA ORGÁNICA (DBO5) ---
    dbo_urbana = pob_urbana * 0.050 * (1 - (cobertura_ptar/100 * eficiencia_ptar/100)) 
    dbo_rural = pob_rural * 0.040 
    dbo_suero = vol_suero * 0.035 
    dbo_agricola = (ha_papa + ha_pastos) * 0.8 
    dbo_bovinos = cabezas_bovinos * factor_dbo_bov
    dbo_porcinos = cabezas_porcinos * factor_dbo_porc
    dbo_aves = cabezas_aves * factor_dbo_aves
    
    carga_total_dbo = dbo_urbana + dbo_rural + dbo_suero + dbo_bovinos + dbo_porcinos + dbo_aves + dbo_agricola
    
    coef_retorno = 0.85
    q_efluente_lps = (q_necesario_dom * coef_retorno) + (q_necesario_ind * 0.8) + (vol_suero / 86400) + ((cabezas_porcinos * 40)/86400)
    conc_efluente_mg_l = (carga_total_dbo * 1_000_000) / (q_efluente_lps * 86400) if q_efluente_lps > 0 else 0

    df_cargas = pd.DataFrame({
        "Fuente": ["Urbana", "Rural", "Agroindustria", "Agricultura", "Bovinos", "Porcinos", "Avicultura"], 
        "DBO_kg_dia": [dbo_urbana, dbo_rural, dbo_suero, dbo_agricola, dbo_bovinos, dbo_porcinos, dbo_aves]
    })

    # --- GRÁFICAS Y RESULTADOS ---
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig_cargas = px.bar(df_cargas, x="DBO_kg_dia", y="Fuente", orientation='h', title=f"Aportes de DBO5 ({carga_total_dbo:,.1f} kg/día)", color="Fuente", color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig_cargas, use_container_width=True)
        csv_cargas = df_cargas.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Descargar Datos de Cargas (CSV)", data=csv_cargas, file_name=f"Inventario_Cargas_{nombre_seleccion}.csv", mime='text/csv')

    with col_g2:
        st.subheader("📈 Evolución de Carga Orgánica (Proyectada)")
        pob_u_evo = pob_urbana * factor_evo
        
        # Corrección: Se añade dbo_aves a la proyección futura
        dbo_evo = (pob_u_evo * 0.050 * (1 - (cobertura_ptar/100 * eficiencia_ptar/100))) + dbo_rural + dbo_suero + dbo_bovinos + dbo_porcinos + dbo_aves + dbo_agricola
        
        fig_dbo_evo = go.Figure()
        fig_dbo_evo.add_trace(go.Scatter(x=anios_evo, y=dbo_evo, mode='lines', fill='tozeroy', name='Carga DBO (kg/d)', line=dict(color='#e74c3c', width=3)))
        st.plotly_chart(fig_dbo_evo, use_container_width=True)
        
# =====================================================================
# 🌊 MÓDULO AVANZADO: ASIMILACIÓN Y CURVA DE OXÍGENO (STREETER-PHELPS)
# =====================================================================
st.markdown("---")
st.header("🌊 4. Capacidad de Asimilación del Río Receptor")
st.info("Modelo de Streeter-Phelps: Simula la caída y recuperación del Oxígeno Disuelto (OD) aguas abajo del vertimiento principal de la zona seleccionada.")

from modules.water_quality import calcular_streeter_phelps

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

# 1.2 Renderizar la interfaz con los valores exactos
with st.expander("⚙️ Características Físicas y Climáticas del Río", expanded=True):
    cr1, cr2, cr3 = st.columns(3)
    
    with cr1:
        st.markdown("##### 📍 Posición y Clima del Vertimiento")
        
        # Mostrar métricas topográficas
        st.caption(f"**Topografía ({nombre_c}):** Mín: {h_min_cuenca:.0f} m | Med: {h_med_cuenca:.0f} m | Máx: {h_max_cuenca:.0f} m")
        
        # Selector Hipsométrico
        h_descarga = st.number_input(
            "Altitud de Descarga (H):", 
            min_value=h_min_cuenca, 
            max_value=h_max_cuenca, 
            value=h_med_cuenca, 
            step=10.0, 
            help="A mayor altitud, menor área aportante (menos caudal) y menor temperatura del agua."
        )
        
        # 🌊 MOTOR HIPSOMÉTRICO
        q_base_cuenca = 5.0
        if 'aleph_q_rio_m3s' in st.session_state and st.session_state['aleph_q_rio_m3s'] > 0:
            q_base_cuenca = float(st.session_state['aleph_q_rio_m3s'])
            
        frac_area, fuente_hipso, eq_hipso = calcular_area_inversa(h_descarga, res_hipso_actual)
        q_rio = max(0.01, q_base_cuenca * frac_area)
        
        # 🌡️ MOTOR TÉRMICO (IDEAM)
        t_sugerida = 28.0 - (0.006 * h_descarga)
        
        if h_descarga < 1000: piso_termico = "Piso: Cálido"
        elif h_descarga < 2000: piso_termico = "Piso: Templado"
        elif h_descarga < 3000: piso_termico = "Piso: Frío"
        elif h_descarga < 4000: piso_termico = "Piso: Páramo"
        else: piso_termico = "Piso: Nival"
        
        # Ajuste dinámico de los límites del slider de temperatura (+/- 2 grados para permitir simulaciones extremas)
        t_min_posible = max(0.0, 28.0 - (0.006 * h_max_cuenca)) - 2.0
        t_max_posible = min(35.0, 28.0 - (0.006 * h_min_cuenca)) + 2.0
        
        # --- EXHIBICIÓN DE LA CIENCIA EN DOS COLUMNAS ---
        c_m1, c_m2 = st.columns(2)
        with c_m1:
            st.caption(fuente_hipso)
            st.latex(eq_hipso)
            reduccion = 100 - (frac_area * 100)
            st.metric(
                "Caudal Local (Q)", 
                f"{q_rio:.2f} m³/s", 
                f"-{reduccion:.1f}% vs Salida", 
                delta_color="normal"
            )
            
        with c_m2:
            st.caption("🌡️ Gradiente Térmico")
            st.latex(r"T = 28 - 0.006 \cdot H")
            st.metric(
                "Temp. Natural", 
                f"{t_sugerida:.1f} °C", 
                piso_termico, 
                delta_color="off"
            )
            
        # El slider obedece a la montaña
        t_agua = st.slider(
            "Temperatura del Agua (°C):", 
            min_value=float(np.floor(t_min_posible)), 
            max_value=float(np.ceil(t_max_posible)), 
            value=float(t_sugerida), 
            step=0.5, 
            help="Sugerida por la altitud. Puedes ajustarla para simular el efecto de una isla de calor urbano o descargas térmicas industriales."
        )
        
    with cr2:
        st.markdown("##### 🌊 Hidráulica")
        v_rio = st.slider("Velocidad del Flujo (m/s):", min_value=0.1, max_value=3.0, value=0.5, step=0.1)
        h_rio = st.slider("Profundidad Media (m):", min_value=0.2, max_value=5.0, value=1.0, step=0.2)
        
    with cr3:
        st.markdown("##### 🧪 Condición Inicial")
        od_rio_arriba = st.slider("OD Aguas Arriba (mg/L):", min_value=0.0, max_value=12.0, value=7.5, step=0.5)
        dist_sim = st.slider("Distancia a Simular (km):", min_value=5, max_value=150, value=50, step=5)
        
# =========================================================================
# 2. Balance de Masas (Mezcla Río + Vertimiento) Parámetros del Vertimiento Y VERTIMIENTO HIPOTÉTICO
# =========================================================================

# 2.1 Calcular Carga Difusa Base (El "Fondo" del Río)
pob_u, pob_r = pob_urbana, pob_rural
bov, por, ave = obtener_censo_pecuario(nombre_seleccion, nivel_sel_interno, anio_analisis)

# Factores de Emisión Típicos (kg DBO/día por individuo)
dbo_hab = (pob_u + pob_r) * 0.054
dbo_bov = bov * 0.600
dbo_por = por * 0.200
dbo_ave = ave * 0.015  # Nuevo aporte avícola

# Carga Pecuaria Total
dbo_gan = dbo_bov + dbo_por + dbo_ave
carga_difusa_total_kg = dbo_hab + dbo_gan

# Asumimos que solo un % de la carga difusa llega efectivamente al cauce en este punto (Atenuación natural)
factor_escorrentia = 0.15 
dbo_rio_arriba_fondo = max(1.0, ((carga_difusa_total_kg * factor_escorrentia) * 1000) / (q_rio * 86400))

with st.expander("🏭 Presión Antrópica y Vertimiento Hipotético", expanded=True):
    st.markdown("##### 🐄🐖🐔 1. Carga Difusa Base (Cuenca Aguas Arriba)")
    cd1, cd2, cd3 = st.columns(3)
    cd1.metric("Población Humana", f"{pob_u + pob_r:,.0f} hab", f"{dbo_hab:,.0f} kg DBO/d", delta_color="inverse")
    
    total_animales = bov + por + ave
    cd2.metric("Censo Pecuario Total", f"{total_animales:,.0f} animales", f"{dbo_gan:,.0f} kg DBO/d", delta_color="inverse")
    
    cd3.metric("Impacto en Río (Fondo)", f"{dbo_rio_arriba_fondo:.1f} mg/L DBO", "Concentración base antes del vertimiento puntual", delta_color="off")
    
    st.markdown("---")
    st.markdown("##### 🧪 2. Simulador de Vertimiento Hipotético (Puntual)")
    st.caption("Modela el impacto de una nueva industria, un tubo roto o una nueva PTAR.")
    
    cv1, cv2, cv3 = st.columns(3)
    with cv1:
        q_vertimiento = st.number_input(
            "Caudal del Vertimiento (m³/s):", 
            min_value=0.000, max_value=50.0, value=0.150, step=0.05, format="%.3f"
        )
    with cv2:
        t_vert_defecto = min(35.0, t_sugerida + 3.0) if 't_sugerida' in locals() else 25.0
        t_vertimiento = st.slider(
            "Temperatura del Efluente (°C):", 
            min_value=10.0, max_value=60.0, value=float(t_vert_defecto), step=0.5
        )
    with cv3:
        # Aquí el usuario decide qué tan sucia viene el agua (Ej: Agua cruda = 300 mg/L, PTAR = 40 mg/L)
        dbo_vert_mgL = st.number_input(
            "Concentración DBO (mg/L):",
            min_value=0.0, max_value=5000.0, value=250.0, step=10.0,
            help="Agua residual doméstica cruda ~250 mg/L. Agua tratada (PTAR) ~40 mg/L. Suero lácteo ~35,000 mg/L."
        )
        carga_puntual_kg = (dbo_vert_mgL * q_vertimiento * 86400) / 1000
        st.caption(f"Carga Inyectada: **{carga_puntual_kg:,.1f} kg/día**")

# =========================================================================
# 3. Termodinámica y Balance de Masas (Mezcla)
# =========================================================================
q_mezcla = q_rio + q_vertimiento

# Evitar división por cero si el usuario apaga todo
if q_mezcla > 0:
    t_mezcla = ((q_rio * t_agua) + (q_vertimiento * t_vertimiento)) / q_mezcla
    L0_mezcla = ((q_rio * dbo_rio_arriba_fondo) + (q_vertimiento * dbo_vert_mgL)) / q_mezcla
    od_mezcla = ((q_rio * od_rio_arriba) + (q_vertimiento * 0.0)) / q_mezcla
else:
    t_mezcla, L0_mezcla, od_mezcla = t_agua, dbo_rio_arriba_fondo, od_rio_arriba

od_sat = 14.652 - 0.41022 * t_mezcla + 0.007991 * (t_mezcla ** 2) - 0.000077774 * (t_mezcla ** 3)
D0_mezcla = max(0, od_sat - od_mezcla)

# Métrica de la Mezcla
if q_vertimiento > 0:
    st.info(f"🧬 **Física de la Mezcla:** Al inyectar el vertimiento, la DBO del río salta de {dbo_rio_arriba_fondo:.1f} a **{L0_mezcla:.1f} mg/L**, y la temperatura se ajusta a **{t_mezcla:.1f}°C**.")

# 4. Motor Streeter-Phelps
# Mantenemos tu llamada original asegurando que use las variables de mezcla
df_sag = calcular_streeter_phelps(
    L0=L0_mezcla, 
    D0=D0_mezcla, 
    T_agua=t_mezcla,  # Usamos la temperatura ya mezclada
    v_ms=v_rio, 
    H_m=h_rio, 
    dist_max_km=dist_sim, 
    paso_km=0.5
)

# 4. Encontrar el Punto Crítico (Donde el oxígeno es mínimo)
punto_critico = df_sag.loc[df_sag['Oxigeno_Disuelto_mgL'].idxmin()]
od_minimo = punto_critico['Oxigeno_Disuelto_mgL']
km_critico = punto_critico['Distancia_km']

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

fig_sag.update_layout(
    title=f"Curva de Oxígeno Disuelto - Río Receptor ({t_agua}°C)",
    xaxis_title="Distancia Aguas Abajo (km)",
    yaxis_title="Concentración (mg/L)",
    hovermode="x unified",
    height=450,
    yaxis=dict(range=[0, od_sat + 1])
)

# Mostrar métricas y gráfica
m_r1, m_r2, m_r3 = st.columns(3)
m_r1.metric("DBO Total Mezcla (L0)", f"{L0_mezcla:.1f} mg/L")
estado_rio = "⚠️ Anoxia / Riesgo Ecológico" if od_minimo < 4.0 else "✅ Saludable"
m_r2.metric("OD Mínimo Crítico", f"{od_minimo:.1f} mg/L", delta=estado_rio, delta_color="normal" if od_minimo >= 4.0 else "inverse")
m_r3.metric("Ubicación del Impacto Crítico", f"Km {km_critico:.1f}")

st.plotly_chart(fig_sag, width="stretch")

# ------------------------------------------------------------------------------
# TAB 4: ESCENARIOS DE MITIGACIÓN (HOLÍSTICOS)
# ------------------------------------------------------------------------------
with tab_mitigacion:
    st.header("🛡️ Simulador de Escenarios de Intervención (CuencaVerde)")
    st.markdown("Combina infraestructura gris (PTAR) con soluciones basadas en la naturaleza y agroecología.")
    
    st.subheader("A. Saneamiento Urbano")
    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1: esc_perdidas = st.slider("Fugas Acueducto (%):", 0.0, 100.0, float(max(0, perd_dom - 10)))
    with col_e2: esc_cobertura = st.slider("Cobertura PTAR Urbana (%):", 0.0, 100.0, float(min(100, cobertura_ptar + 30)))
    with col_e3: esc_eficiencia = st.slider("Remoción DBO PTAR (%):", 0.0, 100.0, float(min(100, eficiencia_ptar + 10)))
        
    st.subheader("B. Intervención Rural y Agroindustrial")
    col_e4, col_e5, col_e6 = st.columns(3)
    with col_e4: esc_biodigestores = st.slider("Cerdos con Biodigestor (%):", 0.0, 100.0, float(min(100, tratamiento_porc + 40)))
    with col_e5: esc_gallinaza = st.slider("Manejo de Gallinaza (%):", 0.0, 100.0, float(min(100, tratamiento_aves + 20)))
    with col_e6: esc_suero = st.slider("Suero Recuperado (Economía Circular) %:", 0.0, 100.0, 50.0)

    st.divider()
    
    # Cálculos del nuevo escenario
    q_efectivo_esc = q_necesario_dom / (1 - (esc_perdidas/100)) if esc_perdidas < 100 else q_necesario_dom
    
    dbo_urbana_esc = pob_urbana * 0.050 * (1 - (esc_cobertura/100 * esc_eficiencia/100))
    dbo_porcinos_esc = cabezas_porcinos * (0.150 * (1 - (esc_biodigestores/100)))
    dbo_aves_esc = cabezas_aves * (0.015 * (1 - (esc_gallinaza/100)))
    dbo_suero_esc = vol_suero * (1 - (esc_suero/100)) * 0.035
    
    carga_total_esc = dbo_urbana_esc + dbo_rural + dbo_suero_esc + dbo_bovinos + dbo_porcinos_esc + dbo_aves_esc + dbo_agricola
    
    col_er1, col_er2 = st.columns([1, 1.5])
    with col_er1:
        st.metric("Agua extraída de la fuente", f"{q_efectivo_esc:,.1f} L/s", delta=f"{q_efectivo_esc - q_efectivo_dom:,.1f} L/s (Agua conservada)", delta_color="inverse")
        st.metric("Materia Orgánica Total al Río", f"{carga_total_esc:,.1f} kg/día", delta=f"{carga_total_esc - carga_total_dbo:,.1f} kg/día (Contaminación evitada)", delta_color="inverse")
    
    with col_er2:
        df_esc = pd.DataFrame({
            "Escenario": ["1. Situación Actual", "1. Situación Actual", "2. Con Proyecto CuencaVerde", "2. Con Proyecto CuencaVerde"],
            "Variable": ["Extracción de Agua (L/s)", "Carga DBO (kg/día)", "Extracción de Agua (L/s)", "Carga DBO (kg/día)"],
            "Valor": [q_efectivo_dom, carga_total_dbo, q_efectivo_esc, carga_total_esc]
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
            # Asumimos que carga_total_ton viene calculada de las pestañas anteriores
            carga_actual = st.session_state.get('carga_total_ton', 1000.0)
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
        
        if nivel_sel_interno == "Nacional (Colombia)": df_m = df_mpios[df_mpios['año'] == anio_base].copy()
        elif nivel_sel_interno == "Jurisdicción Ambiental (CAR)":
            car_norm = normalizar_texto(nombre_seleccion.replace("CAR: ", ""))
            mpios_car = set()
            if not df_concesiones.empty: mpios_car.update(df_concesiones[df_concesiones['car_norm'] == car_norm]['municipio_norm'].unique())
            if not df_vertimientos.empty: mpios_car.update(df_vertimientos[df_vertimientos['car_norm'] == car_norm]['municipio_norm'].unique())
            df_m = df_mpios[(df_mpios['municipio_norm'].isin(mpios_car)) & (df_mpios['año'] == anio_base)].copy()
        elif nivel_sel_interno == "Departamental": df_m = df_mpios[(df_mpios['depto_nom'] == nombre_seleccion) & (df_mpios['año'] == anio_base)].copy()
        elif nivel_sel_interno == "Regional": df_m = df_mpios[(df_mpios['region'] == nombre_seleccion) & (df_mpios['año'] == anio_base)].copy()
        elif nivel_sel_interno == "Municipal": df_m = df_mpios[(df_mpios['municipio_norm'] == normalizar_texto(nombre_seleccion)) & (df_mpios['año'] == anio_base)].copy()
        else: df_m = df_mpios[df_mpios['año'] == anio_base].copy() 
            
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
        df_map = df_concesiones.copy() if "Concesión" in var_mapa else df_vertimientos.copy()
        
        if not df_map.empty:
            lugar_norm = normalizar_texto(nombre_seleccion.replace("CAR: ", ""))
            if nivel_sel_interno == "Jurisdicción Ambiental (CAR)" and 'car_norm' in df_map.columns: df_map = df_map[df_map['car_norm'] == lugar_norm]
            elif nivel_sel_interno == "Departamental" and 'departamento_norm' in df_map.columns: df_map = df_map[df_map['departamento_norm'] == normalizar_texto(nombre_seleccion)]
            elif nivel_sel_interno == "Regional" and 'region_norm' in df_map.columns: df_map = df_map[df_map['region_norm'] == normalizar_texto(nombre_seleccion)]
            elif nivel_sel_interno == "Municipal": df_map = df_map[df_map['municipio_norm'] == lugar_norm]
            elif nivel_sel_interno == "Veredal" and 'vereda_norm' in df_map.columns: df_map = df_map[df_map['vereda_norm'] == lugar_norm]
            
            col_z = 'caudal_lps' if "Concesión" in var_mapa else 'caudal_vert_lps'
            df_map['coordenada_x'] = pd.to_numeric(df_map['coordenada_x'], errors='coerce')
            df_map['coordenada_y'] = pd.to_numeric(df_map['coordenada_y'], errors='coerce')
            df_map[col_z] = pd.to_numeric(df_map[col_z], errors='coerce')
            df_map = df_map.dropna(subset=['coordenada_x', 'coordenada_y', col_z])
            
            mask_magna = (df_map['coordenada_x'] > 100000) & (df_map['coordenada_y'] > 100000)
            mask_wgs84 = (df_map['coordenada_x'] < 0) & (df_map['coordenada_x'] > -85) & (df_map['coordenada_y'] > -5)
            
            df_plot = df_map[mask_magna | mask_wgs84].copy()
            
            if not df_plot.empty and df_plot[col_z].sum() > 0:
                try:
                    import pyproj
                    # EPSG:3116 es el origen Magna-Sirgas central (muy común en Antioquia)
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
            st.warning("No hay base de datos disponible para esta variable.")

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
            st.subheader(f"Registros Encontrados: {len(df_exp)}")
            st.dataframe(df_exp.astype(str), width="stretch")
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
    
    # 1. RE-CÁLCULO INTERNO (Para evitar errores de variables huérfanas)
    dbo_domestica = pob_urbana * 0.050 * (1 - (cobertura_ptar/100 * eficiencia_ptar/100)) + (pob_rural * 0.040)
    dbo_bovinos_ext = cabezas_bovinos * factor_dbo_bov
    dbo_porcinos_ext = cabezas_porcinos * factor_dbo_porc
    dbo_agricola_ext = (ha_papa + ha_pastos) * 0.8
    dbo_agroind_ext = vol_suero * 0.035
    
    # 2. MODELO DE NUTRIENTES (Nitrógeno, Fósforo, Potasio)
    # Factores de excreción y escorrentía aproximados (kg/día)
    n_dom = pob_urbana * 0.012 * (1 - (cobertura_ptar/100 * 0.3)) + (pob_rural * 0.012)
    p_dom = pob_urbana * 0.003 * (1 - (cobertura_ptar/100 * 0.2)) + (pob_rural * 0.003)
    k_dom = pob_urbana * 0.005 + (pob_rural * 0.005)
    
    n_bov = cabezas_bovinos * 0.15 * 0.2 # Solo el 20% llega a cuerpos de agua (escorrentía)
    p_bov = cabezas_bovinos * 0.04 * 0.2
    k_bov = cabezas_bovinos * 0.12 * 0.2
    
    n_porc = cabezas_porcinos * 0.08 * (1 - (tratamiento_porc/100 * 0.5))
    p_porc = cabezas_porcinos * 0.02 * (1 - (tratamiento_porc/100 * 0.4))
    k_porc = cabezas_porcinos * 0.05 * (1 - (tratamiento_porc/100 * 0.2))

    n_aves = cabezas_aves * 0.0015 * (1 - (tratamiento_aves/100 * 0.6))
    p_aves = cabezas_aves * 0.0005 * (1 - (tratamiento_aves/100 * 0.6))
    k_aves = cabezas_aves * 0.0006 * (1 - (tratamiento_aves/100 * 0.6))    
    
    n_agr = (ha_papa * 1.5) + (ha_pastos * 0.5)
    p_agr = (ha_papa * 0.3) + (ha_pastos * 0.1)
    k_agr = (ha_papa * 1.2) + (ha_pastos * 0.4)

    col_ext1, col_ext2 = st.columns([1, 1.2])
    
    with col_ext1:
        st.subheader("Huella de Carbono (Gases Efecto Invernadero)")
        st.caption("Conversión de Metano (CH4) y Óxido Nitroso (N2O) a Toneladas de CO2 equivalente (tCO2e/año).")
        
        # Factores IPCC: CH4 a CO2e (x 28), N2O a CO2e (x 265)
        # Doméstico: Principalmente CH4 de fosas sépticas y PTAR sin captura.
        co2e_dom = (dbo_domestica * 0.25 * 28 * 365) / 1000 
        # Bovinos: Fermentación entérica (muy alta) + estiércol
        co2e_bov = (cabezas_bovinos * 0.16 * 28 * 365) / 1000 # ~60kg CH4/vaca/año
        # Porcinos: Manejo de estiércol (Lagunas anaerobias generan mucho CH4)
        co2e_porc = (dbo_porcinos_ext * 0.25 * 28 * 365) / 1000
        # Aves: Manejo de Gallinaza (Compost)
        co2e_aves = (dbo_aves * 0.25 * 28 * 365) / 1000
        
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
        
        st.dataframe(df_pivot.style.format({
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
    if not df_bovinos.empty and 'HEMBRAS>3AÑOS' in df_bovinos.columns:
        vacas_adultas = int(df_bovinos[df_bovinos['MUNICIPIO_NORM'].isin(mpios_activos)]['HEMBRAS>3AÑOS'].sum())
    if vacas_adultas == 0: vacas_adultas = int(cabezas_bovinos * 0.45) # Estimado si no hay datos
    
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
        kg_proteina = (suero_generado / 1000) * 6.8 
        precio_kg_wpc = 8.5 # Precio internacional USD/kg aprox
        ingresos_anuales = (kg_proteina * precio_kg_wpc) * 365
        
        # Impacto Ambiental Evitado (El suero tiene aprox. 35,000 mg/L de DBO)
        dbo_suero_evitada = suero_generado * 0.035 # kg/día
        
        c_res1, c_res2 = st.columns(2)
        c_res1.metric("Proteína Extraíble (WPC 80%)", f"{kg_proteina:,.1f} kg/día")
        c_res2.metric("Nuevos Ingresos Potenciales", f"${ingresos_anuales:,.0f} USD/año")
        
        st.success(f"🌱 **Impacto Hídrico Evitado:** Si este suero se procesa en lugar de arrojarse al campo o alcantarillado, el territorio evita la contaminación equivalente a **{dbo_suero_evitada:,.0f} kg de DBO/día**. ¡Esto equivale a las aguas residuales de una ciudad de {(dbo_suero_evitada/0.050):,.0f} habitantes!")
        
        # Integración con el Aleph: Sincroniza el dato si quieren usarlo en la pestaña de inventario
        if st.button("🔌 Sincronizar suero con el Inventario General (Aleph)"):
            st.session_state['aleph_vol_suero'] = float(suero_generado)
            st.toast("✅ ¡Volumen de suero inyectado en la memoria global de Sihcli-Poter!")

# =========================================================================
# 5. VIAJE AL SUBSUELO: CALIDAD DEL AGUA SUBTERRÁNEA (LIXIVIACIÓN)
# =========================================================================
st.markdown("---")
st.header("⏬ 5. Vulnerabilidad y Calidad del Acuífero")
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
        if gdf_zona is not None and not gdf_zona.empty:
            area_km2 = gdf_zona.to_crs(epsg=3116).area.sum() / 1_000_000.0 # Metros cuadrados a km²
        else:
            area_km2 = 100.0 # Default
            
        st.caption(f"Área aferente de recarga (Cálculo Espacial): **{area_km2:,.1f} km²**")
        volumen_recarga_m3 = recarga_anual_mm * area_km2 * 1000
        st.metric("Volumen de Infiltración Anual", f"{volumen_recarga_m3:,.0f} m³")

    with cg2:
        st.markdown("##### 💩 Fracción de Lixiviación")
        factor_lixiviacion = st.slider(
            "% de Carga que llega al Acuífero:", 
            min_value=0.0, max_value=50.0, value=15.0, step=1.0,
            help="El suelo actúa como filtro natural. Arcillas retienen más, suelos arenosos y kársticos dejan pasar más contaminantes."
        ) / 100.0
        
        # 3. Extraer la carga difusa total (calculada en la sección 2: Humanos + Vacas + Cerdos)
        # Asegurarnos de que la variable existe por si se ejecuta en otro orden
        carga_difusa_dia = carga_difusa_total_kg if 'carga_difusa_total_kg' in locals() else 1000.0
        
        carga_difusa_anual_kg = carga_difusa_dia * 365
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
            concentracion_acuifero = (masa_lixiviada_kg * 1000) / volumen_recarga_m3
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

