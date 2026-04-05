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
ids_sel, nombre_zona, alt_ref, gdf_zona = selectors.render_selector_espacial()

if not ids_sel or gdf_zona is None or gdf_zona.empty:
    st.info("👈 Seleccione un Territorio (Cuenca, Municipio o Región) en el menú lateral para iniciar.")
    st.stop()

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

def get_name_from_row_v2(row, type_layer):
    cols_map = {c.lower(): c for c in row.index.tolist()}
    targets = ['mpio_cnmbr', 'municipio', 'nombre', 'mpio_nomb'] if type_layer == 'muni' else ['n-nss3', 'n_nss3', 'subc_lbl', 'nom_cuenca', 'nombre']
    for t in targets:
        if t in cols_map: return row[cols_map[t]]
    return "Desconocido"

def add_context_layers_robust(fig, show_cuencas=True, show_muni=False):
    if show_muni:
        gdf_m = load_geojson_cached("MunicipiosAntioquia.geojson")
        if gdf_m is not None:
            gdf_m['geom_simp'] = gdf_m.geometry.simplify(0.001)
            for _, r in gdf_m.iterrows():
                name = get_name_from_row_v2(r, 'muni')
                polys = [r['geom_simp']] if r['geom_simp'].geom_type == 'Polygon' else list(r['geom_simp'].geoms)
                for p in polys:
                    x, y = p.exterior.xy
                    fig.add_trace(go.Scatter(x=list(x), y=list(y), mode='lines', line=dict(width=1.0, color='rgba(50, 50, 50, 0.6)', dash='dot'), text=f"🏙️ {name}", hoverinfo='text', showlegend=False))
    
    if show_cuencas:
        gdf_cu = load_geojson_cached("SubcuencasAinfluencia.geojson")
        if gdf_cu is not None:
            gdf_cu['geom_simp'] = gdf_cu.geometry.simplify(0.001)
            for _, r in gdf_cu.iterrows():
                name = get_name_from_row_v2(r, 'cuenca')
                polys = [r['geom_simp']] if r['geom_simp'].geom_type == 'Polygon' else list(r['geom_simp'].geoms)
                for p in polys:
                    x, y = p.exterior.xy
                    fig.add_trace(go.Scatter(x=list(x), y=list(y), mode='lines', line=dict(width=1.5, color='rgba(0, 100, 255, 0.8)'), text=f"🌊 {name}", hoverinfo='text', showlegend=False))

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

def generar_analisis_texto_corregido(df_stats, tipo_analisis):
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
    ### 📝 Análisis Automático
    * **Promedio:** {avg_val:,.0f} mm
    * **Rango:** {diff:,.0f} mm
    * **Conclusión:** El territorio presenta {conclusion}
    * **Máximo:** {est_max} ({max_val:,.0f} mm)
    * **Mínimo:** {est_min} ({min_val:,.0f} mm)
    """

def generar_raster_ascii(grid_z, minx, miny, cellsize, nrows, ncols):
    header = f"ncols        {ncols}\nnrows        {nrows}\nxllcorner    {minx}\nyllcorner    {miny}\ncellsize     {cellsize}\nNODATA_value -9999\n"
    grid_fill = np.nan_to_num(grid_z.T, nan=-9999)
    body = ""
    for row in np.flipud(grid_fill.T): 
        body += " ".join([f"{val:.2f}" for val in row]) + "\n"
    return header + body

# pages/03_🗺️_Isoyetas_HD.py (BLOQUE 2)

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

# --- 6. METADATOS ---
with st.spinner("Cargando catálogo..."):
    gdf_meta, _ = obtener_estaciones_enriquecidas()

col_id = detectar_columna(gdf_meta, ['id_estacion', 'codigo']) or 'id_estacion'
col_nom = detectar_columna(gdf_meta, ['nombre', 'nom-est']) or 'nombre'
col_muni = detectar_columna(gdf_meta, ['municipio', 'mpio'])
col_alt = detectar_columna(gdf_meta, ['altitud' , 'alt_est'])
col_cuenca = 'CUENCA_GIS' if 'CUENCA_GIS' in gdf_meta.columns else None

# --- 7. LÓGICA ESPACIAL SINCRONIZADA ---
tab_mapa, tab_datos = st.tabs(["🗺️ Visualización Espacial", "💾 Descargas GIS"])

with tab_mapa:
    try:
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
                        grid_res = 200 
                        
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
                        
                        fig.add_trace(go.Contour(
                            z=grid_z.T, x=np.linspace(q_minx, q_maxx, grid_res), y=np.linspace(q_miny, q_maxy, grid_res),
                            colorscale=paleta_colores, zmin=z_min, zmax=z_max, colorbar=dict(title="mm/año"),
                            contours=dict(coloring='heatmap', showlabels=True, labelfont=dict(size=10, color='white')),
                            opacity=0.8, connectgaps=True, line_smoothing=1.3
                        ))
                        
                        add_context_layers_robust(fig, ver_cuencas, ver_municipios)
                        
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
                        
                        st.info(generar_analisis_texto_corregido(df_final, tipo_analisis))
                else:
                    st.warning("⚠️ Quedaron menos de 3 estaciones válidas después de aplicar los filtros de calidad temporal para este año.")
            
            else: 
                st.warning(f"⚠️ Las estaciones en esta zona no tienen registros consolidados para el modo seleccionado ({tipo_analisis}). Intenta con un año anterior o activa la 'Interpolación Temporal'.")
            # --------------------------------

        else:
            st.warning("No hay registros en la base de datos para esta zona y periodo.")
            
        with st.expander("🔍 Ver Datos Crudos", expanded=False):
            if not df_agg.empty and 'df_final' in locals(): st.dataframe(df_final)

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
