# pages/13_🕵️_Detective.py

import os
import sys
import json
import re

import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import box
from sqlalchemy import create_engine, text

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats

# --- 1. CONFIGURACIÓN DE PÁGINA (SIEMPRE PRIMERO) ---
st.set_page_config(page_title="Centro de Diagnóstico", page_icon="🕵️", layout="wide")

# --- 📂 IMPORTACIÓN ROBUSTA DE MÓDULOS ---
try:
    from modules import selectors
    from modules.db_manager import get_engine
except ImportError:
    # Fallback de rutas por si hay problemas de lectura entre carpetas
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules import selectors
    try:
        from modules.db_manager import get_engine
    except ImportError:
        # Último recurso si falla la base de datos
        def get_engine(): return create_engine(st.secrets["DATABASE_URL"])

st.subheader("🕵️ Radiografía de Matrices (Borrar después de usar)")

try:
    engine_sql = get_engine()
    
    q = text("""
        SELECT table_name, column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name IN ('matriz_maestra_demografica', 'matriz_maestra_pecuaria', 'matriz_hidrologica_maestra')
    """)
    
    df_esquema = pd.read_sql(q, engine_sql)
    st.dataframe(df_esquema, use_container_width=True)
    
except Exception as e:
    st.error(f"Error conectando a la BD: {e}")

# ==========================================
# 📂 NUEVO: MENÚ DE NAVEGACIÓN PERSONALIZADO
# ==========================================
# Llama al menú expandible y resalta la página actual
selectors.renderizar_menu_navegacion("Detective")

# ==============================================================================
# 🔒 MURO DE SEGURIDAD GLOBAL (ACCESO BETA)
# ==============================================================================
def muro_de_acceso_beta():
    if "beta_unlocked" not in st.session_state:
        st.session_state["beta_unlocked"] = False
        
    if not st.session_state["beta_unlocked"]:
        st.title("🔒 Sihcli-Poter: Fase de Pruebas (Beta)")
        st.info("Esta plataforma científica se encuentra en fase de acceso restringido. Por favor, ingresa la credencial proporcionada por el equipo de investigación.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            clave_beta = st.text_input("Credencial de Acceso:", type="password")
            if st.button("Ingresar al Gemelo Digital", type="primary", use_container_width=True):
                # 💡 La contraseña por defecto es "Agua2026"
                if clave_beta == st.secrets.get("CLAVE_BETA", "AdminPoter"):
                    st.session_state["beta_unlocked"] = True
                    st.rerun() # Recarga la página y muestra todo el contenido
                else:
                    st.error("❌ Credencial incorrecta. Acceso denegado.")
        
        # 🛑 st.stop() es la magia: evita que Python siga leyendo el código hacia abajo
        st.stop() 

# Llamamos a la función para activar el escudo ANTES de mostrar el contenido
muro_de_acceso_beta()

# ==============================================================================
# --- CONTENIDO DE LA PÁGINA (SOLO VISIBLE SI PASAN EL MURO) ---
# ==============================================================================
st.title("🕵️ Centro de Diagnóstico y Detective")
st.markdown("Herramientas forenses para administrador: Evaluación de coordenadas, proyecciones espaciales y auditoría de la base de datos.")
st.divider()

engine = get_engine()

# --- PESTAÑAS PARA ORGANIZAR TODO EL SISTEMA FORENSE ---
tab_coord, tab_dem, tab_bd = st.tabs([
    "🏥 Salud de Coordenadas (Estaciones)", 
    "⛰️ Diagnóstico DEM vs Cuencas", 
    "🔍 Explorador de Tablas (BD)"
])

# ==============================================================================
# TAB 1: BÚSQUEDA DE COORDENADAS (ANTIGUA PÁGINA 13)
# ==============================================================================
with tab_coord:
    st.header("🏥 Análisis de Integridad Espacial de Estaciones")
    
    st.subheader("1. Conteo de Salud")
    try:
        # Contamos cuántas tienen coordenadas y cuántas no
        df_count = pd.read_sql("""
            SELECT 
                COUNT(*) as total,
                COUNT(latitud) as con_latitud,
                COUNT(longitud) as con_longitud
            FROM estaciones
        """, engine)
        
        total = df_count.iloc[0]['total']
        validas = df_count.iloc[0]['con_latitud']
        
        c1, c2 = st.columns(2)
        c1.metric("Total Estaciones en BD", total)
        c2.metric("Con Coordenadas Válidas", validas)
        
        if validas == 0:
            st.error("🚨 ¡CERO! Ninguna estación tiene coordenadas en las columnas 'latitud'/'longitud'.")
        elif validas < total:
            st.warning(f"⚠️ Hay {total - validas} estaciones huérfanas sin coordenadas.")
        else:
            st.success(f"✅ Excelente. Las {validas} estaciones tienen coordenadas.")

    except Exception as e:
        st.error(str(e))

    st.subheader("2. Inspección de Columnas (Vista Cruda)")
    try:
        # Traemos las primeras 5 filas COMPLETAS
        df_all = pd.read_sql("SELECT * FROM estaciones LIMIT 5", engine)
        st.write("Verifica si tus coordenadas están ocultas en alguna columna con otro nombre:")
        st.dataframe(df_all)
        
        cols = df_all.columns.tolist()
        st.caption(f"**Columnas detectadas:** {cols}")

    except Exception as e:
        st.error(str(e))

# ==============================================================================
# TAB 2: DIAGNÓSTICO DEM vs CUENCAS
# ==============================================================================
with tab_dem:
    st.header("🗺️ Detective Espacial: Conflicto de Proyecciones")
    st.info("Verifica si el archivo DEM y las Cuencas en BD están 'viviendo' en el mismo sistema de coordenadas para evitar mapas en blanco.")
    
    PATH_DEM = "data/DemAntioquia_EPSG3116.tif"

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("1. Análisis del DEM (Raster)")
        try:
            if not rasterio:
                st.error("Librería rasterio no instalada.")
            else:
                try:
                    with rasterio.open(PATH_DEM) as src:
                        st.success(f"✅ DEM Cargado: {PATH_DEM}")
                        dem_crs = src.crs
                        dem_bounds = src.bounds
                        st.code(f"CRS DEM:\n{dem_crs}\n\nLímites:\nIzquierda: {dem_bounds.left:,.0f}\nAbajo:     {dem_bounds.bottom:,.0f}\nDerecha:   {dem_bounds.right:,.0f}\nArriba:    {dem_bounds.top:,.0f}")
                        
                        # Diagnóstico de Origen
                        if dem_bounds.left > 4000000:
                            st.info("ℹ️ TIPO: MAGNA ORIGEN NACIONAL (CTM12)")
                        elif dem_bounds.left > 800000:
                            st.info("ℹ️ TIPO: MAGNA BOGOTÁ (EPSG:3116)")
                        else:
                            st.info("ℹ️ TIPO: Probablemente Grados (WGS84)")
                except FileNotFoundError:
                    st.error(f"❌ No se encontró el archivo: {PATH_DEM}")
        except Exception as e:
            st.error(f"Error analizando DEM: {e}")

    with c2:
        st.subheader("2. Análisis de Cuenca (Vectorial)")
        try:
            gdf_test = gpd.read_postgis("SELECT * FROM cuencas LIMIT 1", engine, geom_col="geometry")
            
            if not gdf_test.empty:
                st.success(f"✅ Cuenca cargada: {gdf_test.iloc[0].get('nombre_cuenca', gdf_test.iloc[0].get('subc_lbl', 'Sin Nombre'))}")
                st.write(f"**CRS Original en BD:** {gdf_test.crs}")
                
                if 'dem_crs' in locals():
                    try:
                        gdf_reproj = gdf_test.to_crs(dem_crs)
                        poly_bounds = gdf_reproj.total_bounds
                        st.code(f"Límites Cuenca (Reproyectada al CRS del DEM):\nIzquierda: {poly_bounds[0]:,.0f}\nAbajo:     {poly_bounds[1]:,.0f}\nDerecha:   {poly_bounds[2]:,.0f}\nArriba:    {poly_bounds[3]:,.0f}")
                        
                        dem_box = box(*dem_bounds)
                        cuenca_box = box(*poly_bounds)
                        
                        if dem_box.intersects(cuenca_box):
                            st.success("🎉 ¡HAY INTERSECCIÓN! Los datos se tocan físicamente.")
                        else:
                            st.error("❌ NO SE TOCAN. Están en lugares diferentes.")
                            
                            dist_x = abs(dem_bounds.left - poly_bounds[0])
                            st.write(f"Distancia en X entre ellos: {dist_x:,.0f} metros")
                            if 3500000 < dist_x < 4500000:
                                st.error("⚠️ La diferencia es ~4,000,000m. Conflicto Origen Nacional vs Bogotá confirmado.")
                    except Exception as e:
                        st.error(f"Error reproyectando: {e}")
            else:
                st.warning("La tabla 'cuencas' está vacía.")

        except Exception as e:
            st.error(f"Error consultando Cuenca: {e}")

# ==============================================================================
# TAB 3: EXPLORADOR BD (MODO FORENSE)
# ==============================================================================
with tab_bd:
    st.header("🔍 Explorador de Tablas de la Base de Datos")
    
    with st.container():
        try:
            with engine.connect() as conn:
                tables = pd.read_sql("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'", conn)
                
                if not tables.empty:
                    table_list = tables['table_name'].tolist()
                    selected_table = st.selectbox("Selecciona la tabla a investigar:", table_list)
                else:
                    st.error("No se encontraron tablas en la base de datos.")
                    selected_table = None
        except Exception as e:
            st.error(f"Error conectando a BD: {e}")
            selected_table = None

    if selected_table:
        st.markdown(f"### 🔬 Analizando: `{selected_table}`")
        
        try:
            with engine.connect() as conn:
                count = pd.read_sql(text(f"SELECT count(*) as total FROM {selected_table}"), conn).iloc[0]['total']
                cols_df = pd.read_sql(text(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{selected_table}'"), conn)
                
                c1, c2 = st.columns(2)
                c1.metric("Filas Totales", count)
                with c2:
                    with st.expander("Ver Columnas y Tipos de Dato"):
                        st.dataframe(cols_df, hide_index=True)

                st.markdown("#### 🌍 Auditoría de Geometría")
                geom_col = "geom" 
                if "geometry" in cols_df['column_name'].values: geom_col = "geometry"
                
                if geom_col in cols_df['column_name'].values:
                    try:
                        q_geo = text(f"""
                            SELECT 
                                ST_SRID({geom_col}) as srid_detectado, 
                                ST_AsText({geom_col}) as ejemplo_coordenada,
                                ST_IsValid({geom_col}) as es_valido
                            FROM {selected_table} 
                            WHERE {geom_col} IS NOT NULL LIMIT 1
                        """)
                        geo_sample = pd.read_sql(q_geo, conn)
                        
                        if not geo_sample.empty:
                            geo_sample = geo_sample.iloc[0]
                            st.write("**Sistema de Referencia (SRID) en BD:**", f"`{geo_sample['srid_detectado']}`")
                            st.write("**Ejemplo de Coordenada:**", f"`{geo_sample['ejemplo_coordenada']}`")
                            
                            coord_text = str(geo_sample['ejemplo_coordenada'])
                            if "POINT" in coord_text or "POLYGON" in coord_text:
                                nums = re.findall(r"[-+]?\d*\.\d+|\d+", coord_text)
                                if nums:
                                    first_num = float(nums[0])
                                    if abs(first_num) <= 180:
                                        st.success("✅ **DIAGNÓSTICO:** Las coordenadas parecen ser **GRADOS (Lat/Lon - WGS84)**.")
                                    else:
                                        st.error(f"🚨 **DIAGNÓSTICO:** Las coordenadas parecen ser **METROS** (Ej. {first_num:,.0f}).")
                        else:
                            st.warning("La columna geométrica está vacía en todos los registros.")
                    except Exception as e:
                        st.warning(f"No se pudo analizar la geometría: {e}")
                else:
                    st.info("Esta tabla no parece tener columna espacial.")

                st.markdown("#### 📄 Vista Previa de Datos Crudos (Primeras 7 filas)")
                cols_safe = [c for c in cols_df['column_name'] if c != geom_col]
                cols_query = ", ".join([f'"{c}"' for c in cols_safe])
                
                try:
                    df_preview = pd.read_sql(text(f"SELECT {cols_query} FROM {selected_table} LIMIT 7"), conn)
                    st.dataframe(df_preview)
                except Exception as e:
                    st.error(f"Error cargando vista previa: {e}")
        except Exception as e:
             st.error(f"Error en análisis de tabla: {e}")

    st.markdown("---")
    st.subheader("🛠️ Consola SQL Manual")
    query = st.text_area("Ejecutar SQL personalizado:", f"SELECT * FROM {selected_table if selected_table else 'cuencas'} LIMIT 10")
    if st.button("Ejecutar Query"):
        with engine.connect() as conn:
            try:
                res = pd.read_sql(text(query), conn)
                st.dataframe(res)
            except Exception as e:
                st.error(f"Error SQL: {e}")

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Diagnóstico Forense", page_icon="🕵️‍♂️", layout="wide")

st.title("🕵️‍♂️ Escáner Forense: Buscando el 'Abultamiento' 2010-2025")
st.markdown("Este módulo aísla los 3 archivos crudos y compara sus promedios anuales puros para detectar si alguna fuente tiene un error de escala (ej. comas por puntos) o un sesgo satelital de sobreestimación.")

col1, col2, col3 = st.columns(3)
with col1:
    file_m = st.file_uploader("1. Histórico (< 2010)", type=['csv'])
with col2:
    file_p1 = st.file_uploader("2. Institucional (2010-2025)", type=['csv'])
with col3:
    file_p2 = st.file_uploader("3. Satelital (Delta Copérnicus)", type=['csv'])

def procesar_archivo(file_obj, nombre_fuente):
    """Lee un archivo ancho, limpia fechas, y saca el promedio puro mensual y anual de TODA la red de ese archivo."""
    if file_obj is None: return None
    
    try:
        # Lectura segura
        try:
            df = pd.read_csv(file_obj, sep=';', encoding='utf-8')
        except UnicodeDecodeError:
            file_obj.seek(0)
            df = pd.read_csv(file_obj, sep=';', encoding='latin1')
            
        # Homologar fecha
        if 'date' in df.columns: df.rename(columns={'date': 'fecha'}, inplace=True)
        if 'fecha' not in df.columns: return None
        
        df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
        df = df.dropna(subset=['fecha'])
        df.set_index('fecha', inplace=True)
        
        # Filtrar solo columnas de estaciones (numéricas)
        cols_estaciones = [c for c in df.columns if str(c).isnumeric()]
        
        # ⚠️ AUDITORÍA DE DECIMALES (No forzamos la coma a punto, solo convertimos para ver qué pasa)
        for col in cols_estaciones:
            if df[col].dtype == object:
                # Si el archivo original traía comas, aquí las cambiamos para que Python las lea
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Calcular el promedio espacial (cuánto llovió en promedio en toda la red ese mes)
        # mean(axis=1) calcula el promedio de todas las estaciones en una misma fila (mes)
        df_mensual = df[cols_estaciones].mean(axis=1).reset_index()
        df_mensual.columns = ['fecha', 'promedio_red_mes']
        
        # Sumar los meses para obtener el total anual
        df_anual = df_mensual.groupby(df_mensual['fecha'].dt.year)['promedio_red_mes'].sum().reset_index()
        df_anual.columns = ['año', 'precipitacion_total_anual']
        df_anual['fuente'] = nombre_fuente
        
        return df_anual
    except Exception as e:
        st.error(f"Error procesando {nombre_fuente}: {e}")
        return None

if file_m or file_p1 or file_p2:
    if st.button("🚀 Ejecutar Diagnóstico", type="primary"):
        with st.spinner("Analizando matrices..."):
            res_m = procesar_archivo(file_m, "Histórico")
            res_p1 = procesar_archivo(file_p1, "Institucional (2010-2025)")
            res_p2 = procesar_archivo(file_p2, "Satelital (Copérnicus)")
            
            # Unir los resultados en una sola tabla
            lista_dfs = [df for df in [res_m, res_p1, res_p2] if df is not None]
            
            if lista_dfs:
                df_total = pd.concat(lista_dfs, ignore_index=True)
                
                # Graficar superposiciones
                st.subheader("📈 Comparativa Anual Pura (Sin Imputaciones)")
                fig = px.line(df_total, x='año', y='precipitacion_total_anual', color='fuente', markers=True,
                              title="Precipitación Total Anual Promedio (Por Fuente de Datos)",
                              labels={'precipitacion_total_anual': 'Precipitación Anual (mm)', 'año': 'Año'},
                              template="plotly_white")
                
                fig.update_layout(hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                ### 🕵️‍♂️ ¿Cómo leer este gráfico?
                1. **El Factor X (Escala):** Si la línea de 'Institucional' o 'Satelital' está en 3,000+ mm y el 'Histórico' está en 1,500 mm, significa que los archivos modernos están en una escala diferente (probablemente un problema de miles vs. decimales en Excel).
                2. **El Sesgo Satelital:** Si el satélite (Copérnicus) está consistentemente un 30% o 40% por encima de los otros dos, confirma que el radar satelital está sobreestimando las lluvias orográficas en los Andes.
                3. **Completitud:** Si un año se desploma a casi cero (como 2026), es simplemente porque no ha terminado, no te preocupes por esos valles en los bordes.
                """)
                
                st.dataframe(df_total.pivot(index='año', columns='fuente', values='precipitacion_total_anual').style.format("{:.1f}"))

# Laboratorio de Correlación: IDEAM vs Copernicus

st.set_page_config(page_title="Correlación Satélite-Tierra", page_icon="🔬", layout="wide")

st.title("🔬 Laboratorio de Correlación: IDEAM vs Copernicus")
st.markdown("Este módulo cruza el periodo de solapamiento para descubrir la verdadera relación matemática entre el sensor terrestre y el radar satelital, filtrando únicamente estaciones con datos válidos.")

col1, col2 = st.columns(2)
with col1:
    file_ideam = st.file_uploader("📥 Subir Histórico IDEAM", type=['csv'])
with col2:
    file_copernicus = st.file_uploader("🛰️ Subir Satelital (Copernicus Delta)", type=['csv'])

if file_ideam and file_copernicus:
    if st.button("🚀 Ejecutar Análisis de Correlación", type="primary"):
        with st.spinner("Purgando columnas vacías y calculando regresiones reales..."):
            try:
                # 1. Lectura de datos
                df_ideam = pd.read_csv(file_ideam, sep=';', decimal=',', encoding='utf-8')
                df_cop = pd.read_csv(file_copernicus, sep=';', decimal=',', encoding='utf-8')
                
                # Estandarización de fechas
                for df in [df_ideam, df_cop]:
                    if 'date' in df.columns: df.rename(columns={'date': 'fecha'}, inplace=True)
                    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
                    df.set_index('fecha', inplace=True)
                    
                # 2. PURGA ESTRICTA: Eliminar columnas 100% vacías antes de comparar
                df_ideam_clean = df_ideam.dropna(axis=1, how='all')
                df_cop_clean = df_cop.dropna(axis=1, how='all')
                    
                # 3. Encontrar estaciones realmente comunes con datos potenciales
                cols_ideam = set([c for c in df_ideam_clean.columns if str(c).isnumeric()])
                cols_cop = set([c for c in df_cop_clean.columns if str(c).isnumeric()])
                estaciones_comunes = list(cols_ideam.intersection(cols_cop))
                
                # 4. Construir Matriz de Solapamiento
                datos_cruzados = []
                MIN_MESES_SOLAPE = 12 # Exigimos al menos un año de datos conjuntos para evitar estadísticas falsas
                
                for est in estaciones_comunes:
                    s_ideam = pd.to_numeric(df_ideam_clean[est], errors='coerce')
                    s_cop = pd.to_numeric(df_cop_clean[est], errors='coerce')
                    
                    # Unir por fecha y eliminar filas donde falte alguna de las dos fuentes
                    df_cruce = pd.DataFrame({'IDEAM': s_ideam, 'Copernicus': s_cop}).dropna()
                    
                    # Candado: Solo correlacionamos si hay suficientes datos
                    if len(df_cruce) >= MIN_MESES_SOLAPE:
                        # Filtrar para no dañar la regresión con ceros absolutos si es necesario
                        # df_cruce = df_cruce[(df_cruce['IDEAM'] > 0) & (df_cruce['Copernicus'] > 0)]
                        
                        slope, intercept, r_value, p_value, std_err = stats.linregress(df_cruce['Copernicus'], df_cruce['IDEAM'])
                        
                        datos_cruzados.append({
                            'Estacion': est,
                            'Meses_Solapados': len(df_cruce),
                            'Media_IDEAM': df_cruce['IDEAM'].mean(),
                            'Media_Copernicus': df_cruce['Copernicus'].mean(),
                            'Correlacion_R': r_value,
                            'R2': r_value**2,
                            'Pendiente_m': slope,
                            'Intercepto_b': intercept,
                            'df_plot': df_cruce 
                        })
                
                # 5. Resultados Globales
                df_resultados = pd.DataFrame(datos_cruzados)
                
                if df_resultados.empty:
                    st.error(f"Ninguna estación cumplió el mínimo de {MIN_MESES_SOLAPE} meses de datos superpuestos válidos.")
                    st.stop()
                    
                df_resultados = df_resultados.sort_values(by='R2', ascending=False).drop(columns=['df_plot'])
                
                st.success(f"✅ Se encontraron {len(df_resultados)} estaciones con solapamiento estadístico válido (>{MIN_MESES_SOLAPE} meses).")
                
                st.markdown("### 📊 Tabla de Relaciones Matemáticas (Y = mX + b)")
                st.markdown("**Y** = Lluvia Real (IDEAM) | **X** = Lluvia Satélite (Copernicus) | **R²** = Confiabilidad del modelo")
                st.dataframe(df_resultados.style.background_gradient(subset=['R2', 'Correlacion_R'], cmap='viridis'))
                
                # 6. Visualizador Interactivo
                st.markdown("---")
                st.markdown("### 📈 Análisis de Dispersión por Estación")
                est_seleccionada = st.selectbox("Seleccione una estación para ver su correlación:", [d['Estacion'] for d in datos_cruzados])
                
                datos_est = next(item for item in datos_cruzados if item["Estacion"] == est_seleccionada)
                df_grafico = datos_est['df_plot']
                
                fig = px.scatter(
                    df_grafico, x='Copernicus', y='IDEAM', 
                    title=f"Estación {est_seleccionada} | R² = {datos_est['R2']:.3f} | Meses: {datos_est['Meses_Solapados']}",
                    labels={'Copernicus': 'Copernicus (mm) [Satélite]', 'IDEAM': 'IDEAM (mm) [Pluviómetro]'},
                    trendline="ols", trendline_color_override="red",
                    opacity=0.7
                )
                
                max_val = max(df_grafico['Copernicus'].max(), df_grafico['IDEAM'].max())
                fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val, line=dict(color="gray", dash="dash"))
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"**💡 Ecuación de Calibración Sugerida:**\n`Lluvia_Real = ({datos_est['Pendiente_m']:.3f} * Lluvia_Copernicus) + {datos_est['Intercepto_b']:.3f}`")
                
                # Exportar matriz de calibración
                csv_calibracion = df_resultados.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig')
                st.download_button(
                    label="📥 Descargar Matriz de Ecuaciones (Para el Generador)",
                    data=csv_calibracion,
                    file_name="Ecuaciones_Calibracion_Copernicus.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"Error procesando: {e}")
