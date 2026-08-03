# pages/03_🗺️_Isoyetas_HD.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import text
import geopandas as gpd
import os
import sys

# --- IMPORTACIÓN ROBUSTA DE MÓDULOS ---
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules.config import Config
    from modules import db_manager, selectors
    from modules.interpolation import interpolador_maestro 
    try:
        from modules.data_processor import complete_series
    except ImportError:
        complete_series = None
except:
    from modules import db_manager, selectors
    from modules.config import Config
    from modules.interpolation import interpolador_maestro
    complete_series = None

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Isoyetas HD", page_icon="🗺️", layout="wide")
st.title("🗺️ Generador Avanzado de Isoyetas (Escenarios & Pronósticos)")

# ==========================================
# 📂 NUEVO: MENÚ DE NAVEGACIÓN PERSONALIZADO
# ==========================================
# Llama al menú expandible y resalta la página actual
selectors.renderizar_menu_navegacion("Isoyetas HD")

# ==========================================
# SECCIÓN DE UI: SELECTORES DE INTERPOLACIÓN
# ==========================================
st.sidebar.markdown("### ⚙️ Configuración del Modelo")

opciones_metodo = {
    "Kriging Ordinario": "kriging",
    "Kriging con Deriva Externa (KED)": "ked",
    "Spline (Thin Plate)": "spline",
    "Distancia Inversa (IDW)": "idw",
    "Tendencia Lineal": "trend"
}

metodo_seleccionado = st.sidebar.selectbox("Método de Interpolación:", options=list(opciones_metodo.keys()), index=0)
metodo_codigo = opciones_metodo[metodo_seleccionado]

modelo_var_codigo = 'spherical'
if "Kriging" in metodo_seleccionado:
    modelo_var_seleccionado = st.sidebar.selectbox("Modelo de Variograma:", options=["Esférico", "Exponencial", "Gaussiano"], index=0)
    mapa_variogramas = {"Esférico": "spherical", "Exponencial": "exponential", "Gaussiano": "gaussian"}
    modelo_var_codigo = mapa_variogramas[modelo_var_seleccionado]

# --- 3. SELECTOR ESPACIAL GLOBAL ---
ids_sel, nombre_zona, alt_ref, gdf_zona, nivel_jerarquico = selectors.render_selector_espacial()

# 🚀 FIX: Suavizamos la barrera. Dejamos pasar el polígono aunque venga sin estaciones.
if not nombre_zona or nombre_zona == "-- Seleccione --":
    st.info("👈 Seleccione un Territorio (Cuenca, Municipio o Región) en el menú lateral para iniciar.")
    st.stop()
    
# Nos aseguramos de que ids_sel sea al menos una tupla vacía para no romper la consulta SQL
if not ids_sel: 
    ids_sel = ('0',) # Un ID falso para que el SQL no colapse con IN ()

# --- 4. FUNCIONES DE SOPORTE ---
@st.cache_data(ttl=3600)
def load_geojson_cached(filename):
    possible_paths = [
        os.path.join("data", filename),
        os.path.join("..", "data", filename),
        os.path.join(os.path.dirname(__file__), '..', 'data', filename),
        os.path.join(os.getcwd(), "data", filename)
    ]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                gdf = gpd.read_file(path)
                if gdf.crs and gdf.crs.to_string() != "EPSG:4326": 
                    gdf = gdf.to_crs("EPSG:4326")
                return gdf
            except: continue
    # CORRECCIÓN DE CACHÉ: Se elimina st.toast para evitar CacheReplayClosureError
    print(f"Advertencia: No se encontró el archivo {filename}")
    return None

def detectar_columna(df, keywords):
    if df is None or df.empty: return None
    cols_orig = df.columns.tolist()
    for kw in keywords:
        kw_clean = kw.lower().replace('-', '').replace('_', '')
        for col in cols_orig:
            col_clean = col.lower().replace('-', '').replace('_', '')
            if kw_clean in col_clean:
                return col
    return None

@st.cache_data(ttl=600)
def obtener_estaciones_enriquecidas():
    try:
        engine = db_manager.get_engine()
        df_est = pd.read_sql("SELECT * FROM estaciones", engine)
        df_est['lat_calc'] = pd.to_numeric(df_est['latitud'], errors='coerce')
        df_est['lon_calc'] = pd.to_numeric(df_est['longitud'], errors='coerce')
        df_est = df_est.dropna(subset=['lat_calc', 'lon_calc'])
        gdf_est = gpd.GeoDataFrame(df_est, geometry=gpd.points_from_xy(df_est.lon_calc, df_est.lat_calc), crs="EPSG:4326")
        
        gdf_cuencas = load_geojson_cached("SubcuencasAinfluencia.geojson")
        if gdf_cuencas is not None:
            col_cuenca_geo = detectar_columna(gdf_cuencas, ['n-nss3', 'n_nss3', 'nnss3', 'nombre', 'subcuenca'])
            if col_cuenca_geo:
                if gdf_cuencas.crs != gdf_est.crs: gdf_cuencas = gdf_cuencas.to_crs(gdf_est.crs)
                gdf_joined = gpd.sjoin(gdf_est, gdf_cuencas[[col_cuenca_geo, 'geometry']], how='left', predicate='within')
                gdf_joined = gdf_joined.rename(columns={col_cuenca_geo: 'CUENCA_GIS'})
                gdf_joined['CUENCA_GIS'] = gdf_joined['CUENCA_GIS'].fillna('Fuera de Jurisdicción')
                return gdf_joined, True
        return gdf_est, False
    except Exception as e:
        return pd.DataFrame(), False

# 🚀 FIX V3.0: Descarga de municipios con reproyección nativa en PostGIS
@st.cache_data(ttl=3600)
def obtener_municipios_db():
    try:
        from modules import db_manager
        import geopandas as gpd
        
        engine = db_manager.get_engine()
        
        # 1. ST_Transform(geometry, 4326): Obliga a PostGIS a entregar los datos en grados (Lat/Lon).
        # 2. ST_Simplify(..., 0.005): Suaviza los bordes (aprox 500m) para que el mapa cargue súper rápido.
        sql_query = """
            SELECT nombre_municipio, 
                   ST_Simplify(ST_Transform(geometry, 4326), 0.005) as geom 
            FROM municipios 
            WHERE depto_nom ILIKE '%%Antioquia%%'
        """
        
        gdf = gpd.read_postgis(sql_query, engine, geom_col='geom')
        
        # Le pegamos la "etiqueta" oficial de WGS84 a GeoPandas por seguridad
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
            
        return gdf
    except Exception as e:
        print(f"Error espacial en capa municipios: {e}")
        return None

# 🚀 FIX V3.0: Función de renderizado espacial COMPLETA (Límite, Cuencas y Municipios)
def add_context_layers_robust(fig, gdf_zona_actual, show_cuencas=False, show_muni=False):
    # 1. SIEMPRE DIBUJAR LA FRONTERA DEL TERRITORIO SELECCIONADO (El Límite Fuerte)
    if gdf_zona_actual is not None and not gdf_zona_actual.empty:
        gdf_z = gdf_zona_actual.to_crs("EPSG:4326")
        for _, r in gdf_z.iterrows():
            geom = r.geometry
            polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
            for p in polys:
                x, y = p.exterior.xy
                # Límite principal en negro grueso para resaltar la zona de estudio
                fig.add_trace(go.Scatter(x=list(x), y=list(y), mode='lines', line=dict(width=3, color='rgba(0,0,0,0.8)'), name="Límite del Territorio", hoverinfo='skip'))
    
    # 2. DIBUJAR CAPA DE CUENCAS SI EL USUARIO LO ACTIVA
    if show_cuencas:
        gdf_cu = load_geojson_cached("SubcuencasAinfluencia.geojson")
        if gdf_cu is not None:
            gdf_cu = gdf_cu.to_crs("EPSG:4326")
            # Simplificación ligera para no saturar la memoria
            gdf_cu['geom_simp'] = gdf_cu.geometry.simplify(0.001)
            
            for _, r in gdf_cu.iterrows():
                # Rescatamos el nombre dinámicamente sin usar la función vieja
                name = r.get('n-nss3', r.get('n_nss3', r.get('nombre', 'Cuenca')))
                
                polys = [r['geom_simp']] if r['geom_simp'].geom_type == 'Polygon' else list(r['geom_simp'].geoms)
                for p in polys:
                    x, y = p.exterior.xy
                    # Cuencas con línea azul
                    fig.add_trace(go.Scatter(x=list(x), y=list(y), mode='lines', line=dict(width=1.5, color='rgba(0, 100, 255, 0.8)'), text=f"🌊 {name}", hoverinfo='text', showlegend=False))

    # 3. DIBUJAR CAPA DE MUNICIPIOS SI EL USUARIO LO ACTIVA (Desde PostGIS)
    if show_muni:
        gdf_m = obtener_municipios_db()
        if gdf_m is not None:
            gdf_m = gdf_m.to_crs("EPSG:4326")
            for _, r in gdf_m.iterrows():
                if r['geom'] is None: continue
                # Capturamos el nombre del municipio
                nom_muni = r.get('nombre_municipio', 'Municipio')
                polys = [r['geom']] if r['geom'].geom_type == 'Polygon' else list(r['geom'].geoms)
                
                for p in polys:
                    x, y = p.exterior.xy
                    # Magia de UI: Añadimos hovertemplate para que el texto siga al cursor en la frontera
                    fig.add_trace(go.Scatter(
                        x=list(x), y=list(y), mode='lines', 
                        line=dict(width=0.7, color='rgba(100, 100, 100, 0.5)', dash='dot'), 
                        name=nom_muni,
                        text=[nom_muni] * len(x),
                        hovertemplate="<b>%{text}</b><extra></extra>", 
                        showlegend=False
                    ))

# RESTAURADO: Funciones auxiliares
def calcular_pronostico(df_anual, target_year):
    proyecciones = []
    for station in df_anual['station_id'].unique():
        datos_est = df_anual[df_anual['station_id'] == station].dropna()
        if len(datos_est) >= 5: 
            try:
                x = datos_est['year'].values
                y = datos_est['total_anual'].values
                slope, intercept = np.polyfit(x, y, 1)
                pred = (slope * target_year) + intercept
                proyecciones.append({'station_id': station, 'valor': max(0, pred)}) 
            except: pass
    return pd.DataFrame(proyecciones)

def generar_analisis_texto_corregido(df_stats, tipo_analisis, config_text):
    if df_stats.empty: return "No hay datos suficientes."
    avg_val = df_stats['valor'].mean()
    min_val = df_stats['valor'].min()
    max_val = df_stats['valor'].max()
    diff = max_val - min_val
    
    try:
        est_max = df_stats.loc[df_stats['valor'].idxmax()]['nombre']
        est_min = df_stats.loc[df_stats['valor'].idxmin()]['nombre']
    except:
        est_max, est_min = "N/A", "N/A"
    
    if diff < 600: conclusion = "un comportamiento regional relativamente uniforme."
    elif diff < 1500: conclusion = "un gradiente de precipitación moderado."
    else: conclusion = "una **fuerte variabilidad orográfica**."
    
    return f"""
    ### 📝 Análisis Automático y Metadatos
    **⚙️ Parámetros de Modelación:**
    {config_text}

    **📊 Resultados Estadísticos:**
    * **Promedio Territorial:** {avg_val:,.0f} mm/año
    * **Rango de Variabilidad:** {diff:,.0f} mm
    * **Punto más Húmedo:** {est_max} ({max_val:,.0f} mm)
    * **Punto más Seco:** {est_min} ({min_val:,.0f} mm)
    * **Conclusión:** El territorio presenta {conclusion}
    """
    
def generar_raster_ascii(grid_z, minx, miny, cellsize, nrows, ncols):
    header = f"ncols        {ncols}\nnrows        {nrows}\nxllcorner    {minx}\nyllcorner    {miny}\ncellsize     {cellsize}\nNODATA_value -9999\n"
    grid_fill = np.nan_to_num(grid_z.T, nan=-9999)
    body = ""
    for row in np.flipud(grid_fill.T): 
        body += " ".join([f"{val:.2f}" for val in row]) + "\n"
    return header + body

# --- 5. SIDEBAR: CONFIGURACIÓN DEL MAPA ---
st.sidebar.header("⚙️ Configuración del Mapa")
tipo_analisis = st.sidebar.selectbox("📊 Modo de Análisis:", ["Año Específico", "Promedio Multianual", "Variabilidad Temporal", "Mínimo Histórico", "Máximo Histórico", "Pronóstico Futuro"])

params_analisis = {}
if tipo_analisis == "Año Específico":
    params_analisis['year'] = st.sidebar.selectbox("📅 Año:", range(2025, 1980, -1))
elif tipo_analisis in ["Promedio Multianual", "Variabilidad Temporal"]:
    params_analisis['start'], params_analisis['end'] = st.sidebar.slider("📅 Periodo:", 1980, 2025, (1990, 2020))
elif tipo_analisis == "Pronóstico Futuro":
    params_analisis['target'] = st.sidebar.slider("🔮 Proyección:", 2026, 2040, 2026)

paleta_colores = st.sidebar.selectbox("🎨 Escala de Color:", options=["YlGnBu", "Jet", "Portland", "Viridis", "RdBu"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("🗺️ Capas Vectoriales")
ver_cuencas = st.sidebar.checkbox("✅ Ver Capa de Cuencas", value=True)
ver_municipios = st.sidebar.checkbox("🏙️ Ver Capa de Municipios", value=False)

c1, c2 = st.sidebar.columns(2)
ignore_zeros = c1.checkbox("🚫 No Ceros", value=True)
ignore_nulls = c2.checkbox("🚫 No Nulos", value=True)

do_interp_temp = False
if complete_series: do_interp_temp = st.sidebar.checkbox("🔄 Interpolación Temporal", value=False)

# 🚀 NUEVO: Interruptor del Mapa de Incertidumbre
ver_error = st.sidebar.checkbox("📉 Ver Incertidumbre (Varianza)", value=False, help="Muestra las zonas de mayor error predictivo (solo disponible para Kriging).")

# --- NUEVAS HERRAMIENTAS V3.0 (Resolución, Suavizado e Info) ---
st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Herramientas de Renderizado")
grid_res = st.sidebar.slider("Resolución Espacial (Píxeles):", min_value=50, max_value=500, value=200, step=50, help="Mayor resolución = isoyetas más definidas pero carga más lenta.")
smooth_val = st.sidebar.slider("Suavizado de Curvas (Smooth):", min_value=0.0, max_value=1.3, value=1.0, step=0.1, help="0 = Cuadrículas crudas. 1.3 = Curvas muy fluidas.")

info_metodos = {
    "Kriging Ordinario": "Usa autocorrelación espacial. Ideal para modelar el clima regional.",
    "Kriging con Deriva Externa (KED)": "Permite usar la altitud como variable secundaria para mayor precisión en montañas.",
    "Spline (Thin Plate)": "Ajusta una superficie exacta por los puntos. Bueno para variaciones suaves.",
    "Distancia Inversa (IDW)": "Método clásico donde los pluviómetros cercanos tienen más peso.",
    "Tendencia Lineal": "Ajusta un plano general. Útil para ver grandes gradientes (ej. Norte-Sur)."
}
st.sidebar.info(f"💡 **Sobre {metodo_seleccionado}:**\n{info_metodos.get(metodo_seleccionado, '')}")

# --- 6. METADATOS Y ÁREA DE INFLUENCIA (BUFFER) ---
with st.spinner("Cargando catálogo de estaciones..."):
    gdf_meta, _ = obtener_estaciones_enriquecidas()

col_id = detectar_columna(gdf_meta, ['id_estacion', 'codigo']) or 'id_estacion'
col_nom = detectar_columna(gdf_meta, ['nombre', 'nom-est']) or 'nombre'
col_muni = detectar_columna(gdf_meta, ['municipio', 'mpio'])
col_alt = detectar_columna(gdf_meta, ['altitud' , 'alt_est'])
col_cuenca = 'CUENCA_GIS' if 'CUENCA_GIS' in gdf_meta.columns else None

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Área de Influencia (Buffer)")
buffer_km = st.sidebar.slider("Radio de Expansión (km):", min_value=0, max_value=50, value=15, step=5, 
                              help="Si el territorio está vacío, aumenta este radio para atrapar estaciones vecinas.")

# 🚀 FIX: Recalcular 'ids_sel' expandiendo el polígono geométricamente
if gdf_zona is not None and not gdf_zona.empty and not gdf_meta.empty:
    try:
        # Reproyectamos a MAGNA-SIRGAS (EPSG:3116) para medir kilómetros reales
        gdf_zona_metric = gdf_zona.to_crs(epsg=3116)
        gdf_meta_metric = gdf_meta.to_crs(epsg=3116)
        
        # Expandimos el polígono (convertimos km a metros)
        zona_buffered = gdf_zona_metric.buffer(buffer_km * 1000)
        
        # Encontramos cuáles estaciones caen dentro de la zona expandida
        gdf_buffer = gpd.GeoDataFrame(geometry=zona_buffered, crs="EPSG:3116")
        estaciones_dentro = gpd.sjoin(gdf_meta_metric, gdf_buffer, how="inner", predicate="intersects")
        
        # Actualizamos la lista oficial de estaciones para el SQL
        ids_sel = estaciones_dentro[col_id].unique().tolist()
        st.sidebar.success(f"📡 Estaciones en el radar: {len(ids_sel)}")
    except Exception as e:
        st.sidebar.error(f"Error en buffer espacial: {e}")

# Escudo Anti-Colapso: Si el buffer en 0km sigue vacío, ponemos un ID falso para no quebrar el SQL
if not ids_sel: ids_sel = ['0']

# --- 7. LÓGICA ESPACIAL SINCRONIZADA ---
tab_mapa, tab_datos = st.tabs(["🗺️ Visualización Espacial", "💾 Descargas GIS"])

with tab_mapa:
    try:
        df_agg = pd.DataFrame() # 🛡️ ESCUDO: Inicializamos vacío para evitar errores si no hay estaciones
        engine = db_manager.get_engine()
        
        ids_clean = [str(i).replace("'", "") for i in ids_sel] 
        ids_sql = "('" + "','".join(ids_clean) + "')"
        
        q_raw = text(f"SELECT p.id_estacion, p.fecha, p.valor FROM precipitacion p WHERE p.id_estacion IN {ids_sql}")
        df_raw = pd.read_sql(q_raw, engine)
        
        if not df_raw.empty:
            df_proc = df_raw.copy()
            df_proc['fecha'] = pd.to_datetime(df_proc['fecha'])
            df_proc = df_proc.groupby(['id_estacion', 'fecha'])['valor'].mean().reset_index()
            
            if do_interp_temp and complete_series:
                with st.spinner("Interpolando huecos temporales..."):
                    df_proc = complete_series(df_proc) 
            
            df_proc['year'] = df_proc['fecha'].dt.year
            
            if not do_interp_temp:
                estaciones_antes = df_proc['id_estacion'].nunique()
                year_counts = df_proc.groupby(['id_estacion', 'year'])['valor'].count().reset_index(name='count')
                valid_years = year_counts[year_counts['count'] >= 10]
                df_proc = pd.merge(df_proc, valid_years[['id_estacion', 'year']], on=['id_estacion', 'year'])
                estaciones_despues = df_proc['id_estacion'].nunique()
                
                if estaciones_despues < estaciones_antes:
                    st.warning(f"⚠️ Atención: {estaciones_antes - estaciones_despues} estaciones fueron descartadas porque tienen menos de 10 meses de datos válidos. **Activa 'Interpolación Temporal'** en el menú izquierdo para intentar rescatarlas.")

            df_annual_sums = df_proc.groupby(['id_estacion', 'year'])['valor'].sum().reset_index(name='total_anual')
            df_annual_sums = df_annual_sums.rename(columns={'id_estacion': 'station_id'})

            # --- FILTROS DE ANÁLISIS ---
            if tipo_analisis == "Año Específico":
                df_agg = df_annual_sums[df_annual_sums['year'] == params_analisis['year']].copy()
                df_agg = df_agg.rename(columns={'total_anual': 'valor'})
            elif tipo_analisis == "Promedio Multianual":
                mask = (df_annual_sums['year'] >= params_analisis['start']) & (df_annual_sums['year'] <= params_analisis['end'])
                df_agg = df_annual_sums[mask].groupby('station_id')['total_anual'].mean().reset_index(name='valor')
            elif tipo_analisis == "Pronóstico Futuro":
                df_agg = calcular_pronostico(df_annual_sums, params_analisis['target'])
            else:
                df_agg = df_annual_sums.groupby('station_id')['total_anual'].max().reset_index(name='valor')
                
            # --- GENERACIÓN DE ISOYETAS ---
            if not df_agg.empty:
                df_agg = df_agg.rename(columns={'station_id': col_id})
                
                # CORRECCIÓN DE TYPO: cols_finales ahora está correctamente declarada
                cols_finales = list(set([col_id, col_nom, 'lat_calc', 'lon_calc'] + ([col_muni] if col_muni else []) + ([col_alt] if col_alt else []) + ([col_cuenca] if col_cuenca else [])))
                df_final = pd.merge(df_agg, gdf_meta[cols_finales], on=col_id).groupby(['lat_calc', 'lon_calc']).first().reset_index()

                if ignore_zeros: df_final = df_final[df_final['valor'] > 1] 
                if ignore_nulls: df_final = df_final.dropna(subset=['valor'])
                
                if len(df_final) >= 3:
                    with st.spinner(f"Interpolando {len(df_final)} estaciones válidas..."):
                        
                        margin_lon = (df_final['lon_calc'].max() - df_final['lon_calc'].min()) * 0.15 or 0.1
                        margin_lat = (df_final['lat_calc'].max() - df_final['lat_calc'].min()) * 0.15 or 0.1
                        q_minx, q_maxx = df_final['lon_calc'].min() - margin_lon, df_final['lon_calc'].max() + margin_lon
                        q_miny, q_maxy = df_final['lat_calc'].min() - margin_lat, df_final['lat_calc'].max() + margin_lat
                        
                        gx_raw, gy_raw = np.mgrid[q_minx:q_maxx:complex(0, grid_res), q_miny:q_maxy:complex(0, grid_res)]
                        gdf_final = gpd.GeoDataFrame(df_final, geometry=gpd.points_from_xy(df_final.lon_calc, df_final.lat_calc), crs="EPSG:4326")
                        
                        try:
                            grid_z, _ = interpolador_maestro(df_puntos=gdf_final, col_val='valor', grid_x=gx_raw, grid_y=gy_raw, metodo=metodo_codigo, modelo_variograma=modelo_var_codigo)
                        except Exception as e:
                            st.error(f"Fallo en interpolación: {e}")
                            grid_z = np.zeros_like(gx_raw)

                        z_min, z_max = df_final['valor'].min(), df_final['valor'].max()
                        if z_max == z_min: z_max += 0.1
                        
                        fig = go.Figure()
                        tit = f"Isoyetas ({metodo_seleccionado}): {tipo_analisis} | {nombre_zona}"
                        
                        df_final['hover_val'] = df_final['valor'].apply(lambda x: f"{x:,.0f}")
                        
                        # Preparando datos enriquecidos para el tooltip del mapa
                        c_muni = df_final[col_muni].fillna('-') if col_muni else ["-"]*len(df_final)
                        c_alt = df_final[col_alt].fillna(0) if col_alt else [0]*len(df_final)
                        c_cuenca = df_final[col_cuenca].fillna('-') if col_cuenca else ["-"]*len(df_final)
                        custom_data = np.stack((c_muni, c_alt, c_cuenca, df_final['hover_val']), axis=-1)
                        
                        # Lógica para alternar entre Isoyetas y Mapa de Error
                        if ver_error and grid_z_var is not None and "Kriging" in metodo_seleccionado:
                            matriz_pintar = grid_z_var.T
                            titulo_color = "Error (mm)"
                            escala_color = "Reds"
                            tit_mapa = f"Incertidumbre ({metodo_seleccionado}) | {nombre_zona}"
                        else:
                            matriz_pintar = grid_z.T
                            titulo_color = "mm/año"
                            escala_color = paleta_colores
                            tit_mapa = f"Isoyetas ({metodo_seleccionado}): {tipo_analisis} | {nombre_zona}"

                        # El Contorno Maestro Dinámico
                        fig.add_trace(go.Contour(
                            z=matriz_pintar, x=np.linspace(q_minx, q_maxx, grid_res), y=np.linspace(q_miny, q_maxy, grid_res),
                            colorscale=escala_color, zmin=np.min(matriz_pintar), zmax=np.max(matriz_pintar), colorbar=dict(title=titulo_color),
                            contours=dict(coloring='heatmap', showlabels=True, labelfont=dict(size=10, color='white')),
                            opacity=0.8, connectgaps=True, line_smoothing=smooth_val
                        ))
                        
                        # 🚀 LLAMADO ACTUALIZADO AL LÍMITE Y MUNICIPIOS EN BASE DE DATOS
                        add_context_layers_robust(fig, gdf_zona, ver_cuencas, ver_municipios)
                        
                        fig.add_trace(go.Scatter(
                            x=df_final['lon_calc'], y=df_final['lat_calc'], mode='markers',
                            marker=dict(size=6, color='black', line=dict(width=1, color='white')),
                            text=df_final[col_nom], 
                            hovertemplate="<b>%{text}</b><br>Valor: %{customdata[3]} mm<br>🏙️: %{customdata[0]}<br>⛰️: %{customdata[1]} m<extra></extra>", 
                            customdata=custom_data, 
                            name="Estaciones"
                        ))
                        
                        fig.update_layout(title=tit, height=650, margin=dict(l=0,r=0,t=40,b=0), xaxis=dict(visible=False, scaleanchor="y", scaleratio=1), yaxis=dict(visible=False), plot_bgcolor='white', dragmode='pan')
                        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
                        
                        # 🚀 FIX: Capturar el estado del sidebar y calcular el área
                        txt_var = f" (Variograma: {modelo_var_seleccionado})" if "Kriging" in metodo_seleccionado else ""
                        
                        if tipo_analisis == "Año Específico": txt_tiempo = f"Año {params_analisis['year']}"
                        elif tipo_analisis in ["Promedio Multianual", "Variabilidad Temporal"]: txt_tiempo = f"Periodo {params_analisis['start']} - {params_analisis['end']}"
                        else: txt_tiempo = f"Proyección {params_analisis.get('target', '')}"
                        
                        # Cálculo matemático del área en km2 usando Magna-Sirgas
                        area_km2 = 0
                        if gdf_zona is not None and not gdf_zona.empty:
                            try:
                                # EPSG:3116 permite medir en metros reales en Colombia
                                area_km2 = gdf_zona.to_crs(epsg=3116).area.sum() / 1_000_000
                            except:
                                pass # Evita errores si la geometría viene rota
                        
                        # Construcción del texto consolidado
                        config_str = f"> *Método:* **{metodo_seleccionado}** {txt_var} | *Temporalidad:* **{txt_tiempo}** | *Radio:* **{buffer_km} km** | *Área de Estudio:* **{area_km2:,.1f} km²**"
                        
                        # Inyección de los datos a la interfaz
                        st.info(generar_analisis_texto_corregido(df_final, tipo_analisis, config_str))
                else:
                    st.warning("⚠️ Quedaron menos de 3 estaciones válidas después de aplicar los filtros de calidad temporal para este año.")
            
            else: 
                st.warning(f"⚠️ Las estaciones en esta zona no tienen registros consolidados para el modo seleccionado ({tipo_analisis}). Intenta con un año anterior o activa la 'Interpolación Temporal'.")
            # --------------------------------

        else:
            st.warning("No hay registros en la base de datos para esta zona y periodo.")
            
        with st.expander("🔍 Ver Datos Crudos", expanded=False):
            # 🚀 Escudo Anti-Errores
            if 'df_final' in locals() and not df_final.empty: 
                st.dataframe(df_final)

    except Exception as e:
        st.error(f"Error procesando datos: {e}")
        
# --- 8. DESCARGAS GIS ---
with tab_datos:
    if 'df_final' in locals() and not df_final.empty:
        st.subheader("💾 Descargas GIS")
        cols_show = [c for c in [col_id, col_nom, col_cuenca, 'valor'] if c in df_final.columns]
        st.dataframe(df_final[cols_show].head(50) if cols_show else df_final.head(50), use_container_width=True)
        
        c1, c2, c3 = st.columns(3)
        gdf_out = gpd.GeoDataFrame(df_final, geometry=gpd.points_from_xy(df_final.lon_calc, df_final.lat_calc), crs="EPSG:4326")
        c1.download_button("🌍 GeoJSON (Puntos)", gdf_out.to_json().encode('utf-8'), f"isoyetas_{tipo_analisis}.geojson", "application/json")
        
        if 'grid_z' in locals():
            asc = generar_raster_ascii(grid_z, q_minx, q_miny, (q_maxx-q_minx)/grid_res, grid_res, grid_res)
            c2.download_button("⬛ Raster (.asc)", asc, f"raster_{tipo_analisis}.asc", "text/plain")
        
        c3.download_button("📊 CSV (Excel)", df_final.to_csv(index=False).encode('utf-8'), f"datos_{tipo_analisis}.csv", "text/csv")
