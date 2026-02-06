# pages/05_🗺️_Isoyetas_HD.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import text
import geopandas as gpd
from scipy.interpolate import Rbf
import os
import sys

# --- IMPORTACIÓN ROBUSTA DE MÓDULOS ---
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules.config import Config
    from modules import db_manager # Usamos el gestor centralizado
    try:
        from modules.data_processor import complete_series
    except ImportError:
        complete_series = None
except:
    # Fallback por si la estructura de carpetas varía
    from modules import db_manager
    from modules.config import Config
    complete_series = None

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Isoyetas HD", page_icon="🗺️", layout="wide")
st.title("🗺️ Generador Avanzado de Isoyetas (Escenarios & Pronósticos)")

# --- 2. FICHA TÉCNICA ---
with st.expander("📘 Ficha Técnica: Metodología, Utilidad y Fuentes", expanded=False):
    st.markdown("""
    ### 1. Concepto y Utilidad
    Las **isoyetas** son líneas que unen puntos de igual precipitación. Este mapa permite visualizar la distribución espacial de la lluvia acumulada.

    ### 2. Metodología de Interpolación
    Se utiliza el algoritmo **RBF (Radial Basis Function)** con núcleo *Thin-Plate Spline*. 
    * Genera una superficie suave y continua, minimizando la curvatura total.
    * Ideal para campos de lluvia en zonas de montaña.

    ### 3. Fuentes de Información
    * **Datos:** Base de datos consolidada SIHCLI.
    * **Cartografía:** Límites oficiales IGAC y CuencaVerde.
    
    ### 4. Modos de Análisis
    * **📅 Año Específico:** Lluvia total acumulada.
    * **📉 Mínimo/Máximo Histórico:** Extremos climáticos.
    * **➗ Promedio Multianual:** Normal Climatológica.
    * **🔮 Pronóstico Futuro:** Proyección de tendencia lineal.
    """)

# --- 3. FUNCIONES DE SOPORTE ---
@st.cache_data(ttl=3600)
def load_geojson_cached(filename):
    """Carga GeoJSONs auxiliares con caché."""
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
    return None

def detectar_columna(df, keywords):
    """Busca columnas ignorando mayúsculas y caracteres especiales."""
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
    """
    PUENTE DE DATOS: Carga estaciones de la BD nueva y simula la estructura antigua.
    """
    try:
        engine = db_manager.get_engine()
        
        # 1. Consulta directa a las columnas reparadas (latitud, longitud)
        # Nota: Ya no usamos ST_Y(geom) porque 'geom' fallaba. Usamos los datos reparados.
        q = "SELECT * FROM estaciones"
        df_est = pd.read_sql(q, engine)
        
        # 2. Generar columnas calculadas para compatibilidad
        # Si 'latitud' es string por error, forzamos float
        df_est['latitud'] = pd.to_numeric(df_est['latitud'], errors='coerce')
        df_est['longitud'] = pd.to_numeric(df_est['longitud'], errors='coerce')
        
        # Creamos lat_calc/lon_calc que espera el resto de tu código
        df_est['lat_calc'] = df_est['latitud']
        df_est['lon_calc'] = df_est['longitud']
        
        # Filtrar las que no tienen coordenadas
        df_est = df_est.dropna(subset=['lat_calc', 'lon_calc'])

        # 3. Crear GeoDataFrame
        gdf_est = gpd.GeoDataFrame(
            df_est, 
            geometry=gpd.points_from_xy(df_est.lon_calc, df_est.lat_calc), 
            crs="EPSG:4326"
        )
        
        # 4. Enriquecimiento espacial (Cruce con Cuencas GIS si existe el archivo)
        gdf_cuencas = load_geojson_cached("SubcuencasAinfluencia.geojson")
        if gdf_cuencas is not None:
            col_cuenca_geo = detectar_columna(gdf_cuencas, ['n-nss3', 'n_nss3', 'nnss3', 'nombre', 'subcuenca'])
            if col_cuenca_geo:
                # Asegurar CRS igual
                if gdf_cuencas.crs != gdf_est.crs:
                    gdf_cuencas = gdf_cuencas.to_crs(gdf_est.crs)
                    
                gdf_joined = gpd.sjoin(gdf_est, gdf_cuencas[[col_cuenca_geo, 'geometry']], how='left', predicate='within')
                gdf_joined = gdf_joined.rename(columns={col_cuenca_geo: 'CUENCA_GIS'})
                gdf_joined['CUENCA_GIS'] = gdf_joined['CUENCA_GIS'].fillna('Fuera de Jurisdicción')
                return gdf_joined, True
                
        return gdf_est, False
    except Exception as e:
        # st.error(f"Error cargando estaciones: {e}") # Debug
        return pd.DataFrame(), False

def generar_raster_ascii(grid_z, minx, miny, cellsize, nrows, ncols):
    header = f"ncols        {ncols}\nnrows        {nrows}\nxllcorner    {minx}\nyllcorner    {miny}\ncellsize     {cellsize}\nNODATA_value -9999\n"
    grid_fill = np.nan_to_num(grid_z.T, nan=-9999)
    body = ""
    for row in np.flipud(grid_fill.T): 
        body += " ".join([f"{val:.2f}" for val in row]) + "\n"
    return header + body

def get_name_from_row_v2(row, type_layer):
    actual_cols = row.index.tolist()
    cols_map = {c.lower(): c for c in actual_cols}
    
    if type_layer == 'muni':
        targets = ['mpio_cnmbr', 'municipio', 'nombre', 'mpio_nomb']
    elif type_layer == 'cuenca':
        targets = ['n-nss3', 'n_nss3', 'subc_lbl', 'nom_cuenca', 'nombre']
    else:
        return "Zona"

    for t in targets:
        if t in cols_map:
            return row[cols_map[t]]
            
    return "Desconocido"

def add_context_layers_robust(fig, minx, miny, maxx, maxy):
    try:
        gdf_m = load_geojson_cached("MunicipiosAntioquia.geojson")
        gdf_cu = load_geojson_cached("SubcuencasAinfluencia.geojson")
        
        # 1. Capa Municipios
        if gdf_m is not None:
            gdf_m['geom_simp'] = gdf_m.geometry.simplify(0.001)
            for _, r in gdf_m.iterrows():
                name = get_name_from_row_v2(r, 'muni')
                geom = r['geom_simp']
                polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms) if geom.geom_type == 'MultiPolygon' else []
                for p in polys:
                    x, y = p.exterior.xy
                    fig.add_trace(go.Scatter(
                        x=list(x), y=list(y), 
                        mode='lines', 
                        line=dict(width=1.0, color='rgba(50, 50, 50, 0.6)', dash='dot'), 
                        text=f"🏙️ {name}", hoverinfo='text', showlegend=False
                    ))
        
        # 2. Capa Cuencas
        if gdf_cu is not None:
            gdf_cu['geom_simp'] = gdf_cu.geometry.simplify(0.001)
            for _, r in gdf_cu.iterrows():
                name = get_name_from_row_v2(r, 'cuenca')
                geom = r['geom_simp']
                polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms) if geom.geom_type == 'MultiPolygon' else []
                for p in polys:
                    x, y = p.exterior.xy
                    fig.add_trace(go.Scatter(
                        x=list(x), y=list(y), 
                        mode='lines', 
                        line=dict(width=1.5, color='rgba(0, 100, 255, 0.8)'), 
                        text=f"🌊 {name}", hoverinfo='text', showlegend=False
                    ))
        return True
    except Exception:
        return False

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

def generar_analisis_texto_corregido(df_stats, tipo_analisis):
    if df_stats.empty: return "No hay datos suficientes."
    avg_val = df_stats['valor'].mean()
    min_val = df_stats['valor'].min()
    max_val = df_stats['valor'].max()
    diff = max_val - min_val
    
    # Intento seguro de obtener nombres
    try:
        est_max = df_stats.loc[df_stats['valor'].idxmax()]['nombre']
        est_min = df_stats.loc[df_stats['valor'].idxmin()]['nombre']
    except:
        est_max = "N/A"
        est_min = "N/A"
    
    if diff < 600: conclusion = "un comportamiento regional relativamente uniforme."
    elif diff < 1500: conclusion = "un gradiente de precipitación moderado."
    else: conclusion = "una **fuerte variabilidad orográfica**."
    
    return f"""
    ### 📝 Análisis Automático
    * **Promedio:** {avg_val:,.0f} mm
    * **Rango:** {diff:,.0f} mm
    * **Conclusión:** El territorio presenta {conclusion}
    * **Máximo:** {est_max} ({max_val:,.0f} mm)
    * **Mínimo:** {est_min} ({min_val:,.0f} mm)
    """

# --- 4. SIDEBAR ---
st.sidebar.header("🔍 Configuración")

with st.spinner("Cargando catálogo..."):
    gdf_meta, exito_cruce = obtener_estaciones_enriquecidas()

if gdf_meta.empty:
    st.error("⚠️ No se pudieron cargar las estaciones. Verifica la conexión a BD.")
    st.stop()

# Continúa con la detección de columnas...
col_id = detectar_columna(gdf_meta, ['id_estacion', 'codigo']) or 'id_estacion'
col_nom = detectar_columna(gdf_meta, ['nombre', 'nom-est']) or 'nombre'
col_region = detectar_columna(gdf_meta, ['region', 'subregion', 'depto_region'])
col_muni = detectar_columna(gdf_meta, ['municipio', 'mpio'])
col_alt = detectar_columna(gdf_meta, ['altitud' , 'alt_est'])
# Buscamos si existe la columna de cuenca calculada en el paso anterior
col_cuenca = 'CUENCA_GIS' if 'CUENCA_GIS' in gdf_meta.columns else None
# ------------------------------------------------------

# --- CONTINUACIÓN PARTE 2: LÓGICA DE FILTRADO Y MAPEO ---

df_filtered_meta = gdf_meta.copy()

# Filtros Jerárquicos (Región -> Cuenca -> Municipio)
if col_region and col_region in df_filtered_meta.columns:
    regs = sorted(df_filtered_meta[col_region].dropna().astype(str).unique())
    sel_reg = st.sidebar.multiselect("📍 Región:", regs, key='filter_reg')
    if sel_reg: df_filtered_meta = df_filtered_meta[df_filtered_meta[col_region].isin(sel_reg)]

if col_cuenca and col_cuenca in df_filtered_meta.columns:
    cuencas = sorted(df_filtered_meta[col_cuenca].dropna().astype(str).unique())
    sel_cuenca = st.sidebar.multiselect("🌊 Cuenca:", cuencas, key='filter_cuenca')
    if sel_cuenca: df_filtered_meta = df_filtered_meta[df_filtered_meta[col_cuenca].isin(sel_cuenca)]

if col_muni and col_muni in df_filtered_meta.columns:
    munis = sorted(df_filtered_meta[col_muni].dropna().astype(str).unique())
    sel_muni = st.sidebar.multiselect("🏙️ Municipio:", munis, key='filter_muni')
    if sel_muni: df_filtered_meta = df_filtered_meta[df_filtered_meta[col_muni].isin(sel_muni)]

st.sidebar.markdown(f"**Estaciones en zona:** {len(df_filtered_meta)}")
st.sidebar.divider()

# Configuración de Análisis
tipo_analisis = st.sidebar.selectbox("📊 Modo de Análisis:", ["Año Específico", "Promedio Multianual", "Variabilidad Temporal", "Mínimo Histórico", "Máximo Histórico", "Pronóstico Futuro"], key='analisis_mode')

params_analisis = {}
if tipo_analisis == "Año Específico":
    params_analisis['year'] = st.sidebar.selectbox("📅 Año:", range(2025, 1980, -1), key='sel_year')
elif tipo_analisis in ["Promedio Multianual", "Variabilidad Temporal"]:
    rango = st.sidebar.slider("📅 Periodo:", 1980, 2025, (1990, 2020), key='sel_period')
    params_analisis['start'], params_analisis['end'] = rango
elif tipo_analisis == "Pronóstico Futuro":
    params_analisis['target'] = st.sidebar.slider("🔮 Proyección:", 2026, 2040, 2026, key='sel_proj')

paleta_colores = st.sidebar.selectbox("🎨 Escala de Color:", options=["YlGnBu", "Jet", "Portland", "Viridis", "RdBu"], index=0)
buffer_km = st.sidebar.slider("📡 Buffer Búsqueda (km):", 0, 100, 20, key='buff_km')
buffer_deg = buffer_km / 111.0

c1, c2 = st.sidebar.columns(2)
ignore_zeros = c1.checkbox("🚫 No Ceros", value=True, key='chk_zeros')
ignore_nulls = c2.checkbox("🚫 No Nulos", value=True, key='chk_nulls')

do_interp_temp = False
if complete_series: do_interp_temp = st.sidebar.checkbox("🔄 Interpolación Temporal", value=False, key='chk_interp')

suavidad = st.sidebar.slider("🖌️ Suavizado (RBF):", 0.0, 2.0, 0.1, key='slider_smooth')

# --- 5. LÓGICA ESPACIAL Y RENDERIZADO (CORREGIDO) ---
if len(df_filtered_meta) > 0:
    gdf_target = df_filtered_meta
    
    # 1. Obtener límites crudos
    bounds = gdf_target.total_bounds
    
    # 2. CORRECCIÓN CRÍTICA: Convertir de NumPy a Float de Python puro
    # Esto evita el error: schema "np" does not exist en SQL
    minx = float(bounds[0])
    miny = float(bounds[1])
    maxx = float(bounds[2])
    maxy = float(bounds[3])
    
    # 3. Aplicar buffer (ahora seguro porque son floats normales)
    q_minx = minx - buffer_deg
    q_miny = miny - buffer_deg
    q_maxx = maxx + buffer_deg
    q_maxy = maxy + buffer_deg

    tab_mapa, tab_datos = st.tabs(["🗺️ Visualización Espacial", "💾 Descargas GIS"])

    with tab_mapa:
        try:
            engine = db_manager.get_engine()
            df_agg = pd.DataFrame()
            
            # --- CONSULTA SQL OPTIMIZADA Y SEGURA ---
            # Filtramos estaciones por bounding box para no cargar toda la historia de lluvia
            q_raw = text(f"""
                SELECT p.id_estacion, p.fecha, p.valor
                FROM precipitacion p 
                JOIN estaciones e ON p.id_estacion = e.id_estacion
                WHERE e.longitud BETWEEN :mx AND :Mx 
                AND e.latitud BETWEEN :my AND :My
            """)
            
            # Ejecutamos con parámetros de seguridad
            df_raw = pd.read_sql(q_raw, engine, params={"mx":q_minx, "my":q_miny, "Mx":q_maxx, "My":q_maxy})
            
            if not df_raw.empty:
                df_proc = df_raw.copy()
                df_proc['fecha'] = pd.to_datetime(df_proc['fecha'])
                
                # Agrupación Inicial (Valor diario promedio si hay duplicados)
                df_proc = df_proc.groupby(['id_estacion', 'fecha'])['valor'].mean().reset_index()
                
                # Interpolación Temporal (Opcional)
                if do_interp_temp and complete_series:
                    with st.spinner("Interpolando huecos en series..."):
                        df_processed = complete_series(df_proc) 
                else:
                    df_processed = df_proc.copy()
                
                # Procesamiento Anual
                df_processed['year'] = df_processed['fecha'].dt.year
                
                # Filtro de Calidad (Mínimo 10 meses de datos por año para considerarlo válido)
                if not do_interp_temp:
                    year_counts = df_processed.groupby(['id_estacion', 'year'])['valor'].count().reset_index(name='count')
                    valid_years = year_counts[year_counts['count'] >= 10]
                    df_processed = pd.merge(df_processed, valid_years[['id_estacion', 'year']], on=['id_estacion', 'year'])
                
                # Total Anual
                df_annual_sums = df_processed.groupby(['id_estacion', 'year'])['valor'].sum().reset_index(name='total_anual')
                df_annual_sums = df_annual_sums.rename(columns={'id_estacion': 'station_id'})

                # --- SELECTOR DE MODO DE ANÁLISIS ---
                if tipo_analisis == "Año Específico":
                    df_agg = df_annual_sums[df_annual_sums['year'] == params_analisis['year']].copy()
                    df_agg = df_agg.rename(columns={'total_anual': 'valor'})
                
                elif tipo_analisis == "Promedio Multianual":
                    mask = (df_annual_sums['year'] >= params_analisis['start']) & (df_annual_sums['year'] <= params_analisis['end'])
                    df_agg = df_annual_sums[mask].groupby('station_id')['total_anual'].mean().reset_index(name='valor')
                
                elif tipo_analisis == "Máximo Histórico":
                    df_agg = df_annual_sums.groupby('station_id')['total_anual'].max().reset_index(name='valor')
                
                elif tipo_analisis == "Mínimo Histórico":
                    df_agg = df_annual_sums.groupby('station_id')['total_anual'].min().reset_index(name='valor')
                
                elif tipo_analisis == "Variabilidad Temporal":
                    mask = (df_annual_sums['year'] >= params_analisis['start']) & (df_annual_sums['year'] <= params_analisis['end'])
                    df_agg = df_annual_sums[mask].groupby('station_id')['total_anual'].std().reset_index(name='valor')

                elif tipo_analisis == "Pronóstico Futuro":
                    with st.spinner("Calculando tendencias..."):
                        df_agg = calcular_pronostico(df_annual_sums, params_analisis['target'])

            # --- GENERACIÓN DE ISOYETAS ---
            if not df_agg.empty:
                df_agg = df_agg.rename(columns={'station_id': col_id})
                
                # Merge: Unimos los valores calculados con las coordenadas (lat_calc/lon_calc)
                cols_merge = [col_id, col_nom, 'lat_calc', 'lon_calc']
                if col_muni: cols_merge.append(col_muni)
                if col_alt: cols_merge.append(col_alt)
                if col_cuenca: cols_merge.append(col_cuenca)
                
                # Usamos set() para evitar duplicados si col_nom == col_id
                cols_finales = list(set(cols_merge))
                df_final = pd.merge(df_agg, gdf_meta[cols_finales], on=col_id)
                
                # Limpieza de duplicados espaciales (promediar si hay 2 estaciones en el mismo pixel)
                df_final = df_final.groupby(['lat_calc', 'lon_calc']).agg({
                    col_id: 'first', col_nom: 'first', 'valor': 'mean', 
                    col_muni: 'first', col_alt: 'first', col_cuenca: 'first'
                }).reset_index()

                if ignore_zeros: df_final = df_final[df_final['valor'] > 1] # Filtro lluvia fantasma
                if ignore_nulls: df_final = df_final.dropna(subset=['valor'])
                
                if len(df_final) >= 3:
                    with st.spinner(f"Interpolando {len(df_final)} puntos..."):
                        grid_res = 200 # Resolución de la malla
                        
                        x_raw, y_raw, z_raw = df_final['lon_calc'].values, df_final['lat_calc'].values, df_final['valor'].values
                        
                        # Normalización Z-Score para estabilidad numérica del RBF
                        x_mean, x_std = x_raw.mean(), x_raw.std()
                        y_mean, y_std = y_raw.mean(), y_raw.std()
                        x_norm = (x_raw - x_mean) / x_std
                        y_norm = (y_raw - y_mean) / y_std
                        
                        # Malla destino
                        gx_raw, gy_raw = np.mgrid[q_minx:q_maxx:complex(0, grid_res), q_miny:q_maxy:complex(0, grid_res)]
                        gx_norm = (gx_raw - x_mean) / x_std
                        gy_norm = (gy_raw - y_mean) / y_std
                        
                        # Interpolación RBF
                        try:
                            rbf = Rbf(x_norm, y_norm, z_raw, function='thin_plate', smooth=suavidad)
                            grid_z = rbf(gx_norm, gy_norm)
                            grid_z = np.maximum(grid_z, 0) # Física: Lluvia no puede ser negativa
                        except Exception as e:
                            st.warning(f"Error matemático en interpolación (puntos superpuestos): {e}")
                            grid_z = np.zeros_like(gx_raw)

                        # Escala de Color Dinámica
                        z_min, z_max = df_final['valor'].min(), df_final['valor'].max()
                        if z_max == z_min: z_max += 0.1
                        
                        # --- GRAFICAR ---
                        fig = go.Figure()
                        tit = f"Isoyetas: {tipo_analisis}"
                        if tipo_analisis == "Año Específico": tit += f" ({params_analisis['year']})"
                        
                        # Datos hover
                        df_final['hover_val'] = df_final['valor'].apply(lambda x: f"{x:,.0f}")
                        c_muni = df_final[col_muni].fillna('-') if col_muni else ["-"]*len(df_final)
                        c_alt = df_final[col_alt].fillna(0) if col_alt else [0]*len(df_final)
                        c_cuenca = df_final[col_cuenca].fillna('-') if col_cuenca else ["-"]*len(df_final)
                        custom_data = np.stack((c_muni, c_alt, c_cuenca, df_final['hover_val']), axis=-1)
                        
                        # 1. Contornos Rellenos
                        fig.add_trace(go.Contour(
                            z=grid_z.T, x=np.linspace(q_minx, q_maxx, grid_res), y=np.linspace(q_miny, q_maxy, grid_res),
                            colorscale=paleta_colores, zmin=z_min, zmax=z_max,
                            colorbar=dict(title="mm/año"),
                            contours=dict(coloring='heatmap', showlabels=True, labelfont=dict(size=10, color='white')),
                            opacity=0.8, connectgaps=True, line_smoothing=1.3
                        ))
                        
                        # 2. Capas de Contexto (Municipios/Cuencas)
                        add_context_layers_robust(fig, q_minx, q_miny, q_maxx, q_maxy)
                        
                        # 3. Puntos Estaciones
                        fig.add_trace(go.Scatter(
                            x=df_final['lon_calc'], y=df_final['lat_calc'], mode='markers',
                            marker=dict(size=6, color='black', line=dict(width=1, color='white')),
                            text=df_final[col_nom], customdata=custom_data,
                            hovertemplate="<b>%{text}</b><br>Valor: %{customdata[3]} mm<br>🏙️: %{customdata[0]}<br>⛰️: %{customdata[1]} m<extra></extra>",
                            name="Estaciones"
                        ))
                        
                        # Configuración final
                        fig.update_layout(
                            title=tit, height=650, 
                            margin=dict(l=0,r=0,t=40,b=0),
                            xaxis=dict(visible=False, scaleanchor="y", scaleratio=1), # Aspect ratio 1:1 real
                            yaxis=dict(visible=False), 
                            plot_bgcolor='white', hovermode='closest'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Reporte Automático
                        st.info(generar_analisis_texto_corregido(df_final, tipo_analisis))
                        
                else:
                    st.warning("⚠️ Datos insuficientes en esta zona (se requieren al menos 3 estaciones). Intente aumentar el Buffer (km).")
            else:
                st.warning("No se encontraron registros de lluvia para los filtros seleccionados.")
            
            with st.expander("🔍 Ver Datos Crudos", expanded=False):
                if not df_agg.empty: st.dataframe(df_final)

        except Exception as e:
            st.error(f"Error general en proceso: {e}")

    # --- PESTAÑA DESCARGAS ---
    with tab_datos:
        if 'df_final' in locals() and not df_final.empty:
            st.subheader("💾 Descargas GIS")
            
            # Tabla previa
            cols_show = [col_id, col_nom, 'valor']
            if col_cuenca in df_final.columns: cols_show.append(col_cuenca)
            st.dataframe(df_final[cols_show].head(50), use_container_width=True)
            
            c1, c2, c3 = st.columns(3)
            
            # GeoJSON
            gdf_out = gpd.GeoDataFrame(df_final, geometry=gpd.points_from_xy(df_final.lon_calc, df_final.lat_calc), crs="EPSG:4326")
            c1.download_button("🌍 GeoJSON (Puntos)", gdf_out.to_json(), f"isoyetas_{tipo_analisis}.geojson", "application/json")
            
            # Raster ASC
            if 'grid_z' in locals():
                asc = generar_raster_ascii(grid_z, q_minx, q_miny, (q_maxx-q_minx)/grid_res, grid_res, grid_res)
                c2.download_button("⬛ Raster (.asc)", asc, f"raster_{tipo_analisis}.asc", "text/plain")
            
            # CSV
            csv = df_final.to_csv(index=False).encode('utf-8')
            c3.download_button("📊 CSV (Excel)", csv, f"datos_{tipo_analisis}.csv", "text/csv")

else:
    st.info("👈 Seleccione una Región o Cuenca en el menú lateral para iniciar.")