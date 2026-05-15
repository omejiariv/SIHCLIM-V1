# modules/selectors.py

import streamlit as st
import geopandas as gpd
import pandas as pd
from sqlalchemy import text
import io
import requests
import time

from modules import db_manager
from modules.config import Config

# ====================================================================
# --- 1. REPARADOR DE TILDES (MOJIBAKE) ---
# ====================================================================
def decodificar_tildes(texto):
    """Corrige errores de codificación en BD como 'Abriaquã' -> 'Abriaquí'"""
    if not isinstance(texto, str): return texto
    try:
        if 'Ã' in texto or 'ã' in texto or '\x8d' in texto:
            return texto.encode('latin1').decode('utf-8')
    except: pass
    return texto

# ====================================================================
# --- 2. TELEMETRÍA DEL ALEPH (CON CLIMA ENSO) ---
# ====================================================================
def renderizar_telemetria_aleph():
    """
    Panel de control universal que monitorea las variables de estado en tiempo real.
    Se inyecta en la barra lateral de todas las páginas.
    """
    st.sidebar.markdown("---")
    
    # 🧠 EL ALEPH: Monitor de Memoria Activa
    with st.sidebar.expander("🧠 Telemetría del Aleph", expanded=True):
        
        # 1. Monitoreo Demográfico
        pob_viva = st.session_state.get('aleph_pob_total', st.session_state.get('poblacion_servida', 0))
        if pob_viva > 0:
            st.markdown(f"👥 **Población:** <span style='color:#2ecc71'>Sincronizada ({pob_viva:,.0f} hab)</span>", unsafe_allow_html=True)
        else:
            st.markdown("👥 **Población:** <span style='color:#e74c3c'>Vacía (Default)</span>", unsafe_allow_html=True)
            
        # 2. Monitoreo de Amenazas Físicas (Lodos)
        lodo_vivo = st.session_state.get('eco_lodo_total_m3', 0.0)
        if lodo_vivo > 0:
            st.markdown(f"⛈️ **Tormenta:** <span style='color:#e67e22'>Activa ({lodo_vivo:,.0f} m³ Lodo)</span>", unsafe_allow_html=True)
        else:
            st.markdown("⛈️ **Tormenta:** <span style='color:#95a5a6'>Inactiva</span>", unsafe_allow_html=True)
            
        # 3. Monitoreo Químico (Carga Orgánica)
        dbo_viva = st.session_state.get('carga_dbo_total_ton', 0.0)
        if dbo_viva > 0:
            st.markdown(f"☣️ **Carga DBO:** <span style='color:#8e44ad'>Registrada ({dbo_viva:,.0f} Ton)</span>", unsafe_allow_html=True)
        else:
            st.markdown("☣️ **Carga DBO:** <span style='color:#95a5a6'>Inactiva</span>", unsafe_allow_html=True)
            
        # ==========================================================
        # 🌍 4. MONITOREO CLIMÁTICO GLOBAL (ALEPH CLIMÁTICO)
        # ==========================================================
        st.markdown("---")
        
        # 📡 AUTO-FETCH INTELIGENTE (Búsqueda por Trimestre AMJ)
        if 'aleph_iri_nino' not in st.session_state:
            try:
                from modules.climate_api import get_iri_enso_forecast
                df_enso, _ = get_iri_enso_forecast()
                
                if df_enso is not None and not df_enso.empty:
                    # 🎯 BUSCADOR: Intentamos encontrar AMJ, si no, tomamos el actual
                    df_target = df_enso[df_enso['Trimestre'].str.contains('AMJ', na=False)]
                    
                    if not df_target.empty:
                        fila_actual = df_target.iloc[0]
                    else:
                        fila_actual = df_enso.iloc[0]
                    
                    p_nina = float(fila_actual.get('La Niña', 0))
                    p_nino = float(fila_actual.get('El Niño', 0))
                    p_neutro = float(fila_actual.get('Neutral', 0))
                    trim_txt = str(fila_actual.get('Trimestre', 'Actual'))
                    
                    # Lógica de fase basada en los números reales
                    if p_nina > 50: estado = "Niña 🌧️"
                    elif p_nino > 50: estado = "Niño ☀️"
                    else: estado = "Neutro ⚖️"
                    
                    # Actualizamos la memoria global
                    st.session_state['enso_fase'] = estado
                    st.session_state['aleph_iri_nino'] = int(p_nino)
                    st.session_state['aleph_iri_neutro'] = int(p_neutro)
                    st.session_state['aleph_iri_nina'] = int(p_nina)
                    st.session_state['aleph_iri_trimestre'] = trim_txt
                    st.session_state['aleph_iri_tendencia'] = "Sincronización AMJ Exitosa 📡"
            except Exception as e:
                st.session_state['aleph_iri_tendencia'] = f"Error: {str(e)}"
                
        # --- RENDERIZADO DINÁMICO EN EL PANEL ---
        enso_global = st.session_state.get('enso_fase', 'Desconocido ⚠️')
        color_enso = "#3498db" if "Niña" in enso_global else "#e74c3c" if "Niño" in enso_global else "#f39c12" if "Desconocido" in enso_global else "#2ecc71"
        st.markdown(f"🌍 **Clima ENSO:** <span style='color:{color_enso}'><b>{enso_global}</b></span>", unsafe_allow_html=True)
        
        st.caption("📡 **Pronóstico IRI (Memoria Activa)**")
        
        p_nino = st.session_state.get('aleph_iri_nino', 0)
        p_neutro = st.session_state.get('aleph_iri_neutro', 0)
        p_nina = st.session_state.get('aleph_iri_nina', 0)
        trim = st.session_state.get('aleph_iri_trimestre', 'Sin datos')
        tendencia = st.session_state.get('aleph_iri_tendencia', '')
        
        # Solo dibujamos las barras de progreso si hay datos reales
        if (p_nino + p_neutro + p_nina) > 0:
            st.caption(f"Probabilidad ({trim}):")
            st.progress(p_nino, text=f"☀️ El Niño ({p_nino}%)")
            st.progress(p_neutro, text=f"⚖️ Neutro ({p_neutro}%)")
            st.progress(p_nina, text=f"🌧️ La Niña ({p_nina}%)")
        
        if tendencia:
            st.caption(f"📈 **Estado:** {tendencia}")

        st.markdown("---")
        # 🚀 MEJORA: El botón ahora destruye tanto el Session State como el Data Cache
        if st.button("🧹 Purgar Memoria y Caché", use_container_width=True):
            st.session_state.clear()
            st.cache_data.clear()
            st.rerun()

# ====================================================================
# --- 3. NAVEGACIÓN GLOBAL ---
# ====================================================================
def renderizar_menu_navegacion(pagina_actual):
    titulo_expander = f"📂 Navegación | Actual: {pagina_actual}"
    with st.sidebar.expander(titulo_expander, expanded=False):
        st.page_link("app.py", label="Inicio", icon="🏠")
        st.page_link("pages/01_🌦️_Clima_e_Hidrologia.py", label="Clima e Hidrología", icon="🌦️")
        st.page_link("pages/02_💧_Aguas_Subterraneas.py", label="Aguas Subterráneas", icon="💧")
        st.page_link("pages/03_🗺️_Isoyetas_HD.py", label="Isoyetas HD", icon="🗺️")
        st.page_link("pages/04_🍃_Biodiversidad.py", label="Biodiversidad", icon="🌱")
        st.page_link("pages/05_🏔️_Geomorfologia.py", label="Geomorfología", icon="⛰️")
        st.page_link("pages/06_🐄_Modelo_Pecuario.py", label="Modelo Pecuario", icon="🐄")
        st.page_link("pages/06_📈_Modelo_Demografico.py", label="Modelo Demográfico", icon="👥")
        st.page_link("pages/07_💧_Calidad_y_Vertimientos.py", label="Calidad y Vertimientos", icon="🧪")
        st.page_link("pages/08_🔗_Sistemas_Hidricos_Territoriales.py", label="Sistemas Hídricos", icon="🌊")
        st.page_link("pages/09_🧠_Toma_de_Decisiones.py", label="Toma de Decisiones", icon="🧠")
        st.page_link("pages/10_👑_Panel_Administracion.py", label="Panel Administración", icon="⚙️")
        st.page_link("pages/11_⚙️_Generador.py", label="Generador", icon="✨")
        st.page_link("pages/12_📚_Ayuda_y_Docs.py", label="Ayuda y Docs", icon="📚")
        st.page_link("pages/13_🕵️_Detective.py", label="Detective", icon="🕵️")
        st.page_link("pages/14_🛰️_Satelite_En_Vivo.py", label="Satélite en Vivo", icon="🛰️")
    
    # 🚀 AQUÍ OCURRE LA MAGIA: Llamamos a la telemetría para que se dibuje
    renderizar_telemetria_aleph()

# ====================================================================
# --- 4. CONEXIÓN A SUPABASE (Para guardar los JSON) ---
# ====================================================================
@st.cache_resource
def get_supabase_client():
    try:
        from supabase import create_client
        
        url_sb = st.secrets.get("SUPABASE_URL")
        key_sb = st.secrets.get("SUPABASE_KEY")
        
        if not url_sb: url_sb = st.secrets.get("supabase", {}).get("SUPABASE_URL")
        if not key_sb: key_sb = st.secrets.get("supabase", {}).get("SUPABASE_KEY")
            
        if not url_sb: url_sb = st.secrets.get("supabase", {}).get("url")
        if not key_sb: key_sb = st.secrets.get("supabase", {}).get("key")

        if url_sb and key_sb: return create_client(url_sb, key_sb)
        else: return "NO_SECRETS"
            
    except ImportError: return "NO_LIBRARY"
    except Exception as e: return str(e)

# ====================================================================
# --- 5. GESTOR DE SNAPSHOTS (GUARDAR, CARGAR Y ELIMINAR ESCENARIOS) ---
# ====================================================================
def renderizar_gestor_escenarios(nombre_zona_actual):
    st.sidebar.markdown("---")
    with st.sidebar.expander("💾 Gestor de Escenarios (Snapshots)", expanded=False):
        supabase = get_supabase_client()
        
        if supabase == "NO_SECRETS" or supabase == "NO_LIBRARY" or isinstance(supabase, str) or not supabase:
            st.error("Error de conexión a Supabase. Revisa tus credenciales.")
            return

        tab_guardar, tab_cargar = st.tabs(["Guardar", "Cargar / Eliminar"])
        
        with tab_guardar:
            nombre_escenario = st.text_input("Nombre del Proyecto/Escenario:", placeholder="Ej: Río Buey - Plan 2030")
            if st.button("💾 Guardar Sesión", use_container_width=True):
                if nombre_escenario:
                    with st.spinner("Empaquetando sesión en JSON..."):
                        estado_limpio = {k: v for k, v in st.session_state.items() if isinstance(v, (int, float, str, bool, list, dict)) and not k.startswith('FormSubmitter')}
                        try:
                            supabase.table("escenarios_guardados").insert({
                                "nombre_escenario": nombre_escenario,
                                "territorio": nombre_zona_actual,
                                "estado_json": estado_limpio
                            }).execute()
                            st.success("✅ ¡Escenario guardado en la Nube!")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error guardando en tabla: {e}")
                else:
                    st.warning("⚠️ Debes darle un nombre al escenario.")

        with tab_cargar:
            try:
                res = supabase.table("escenarios_guardados").select("id, nombre_escenario, territorio, fecha_creacion").order("fecha_creacion", desc=True).execute()
                escenarios = res.data
                if escenarios:
                    opciones = {f"{e['nombre_escenario']} ({e['territorio']})": e['id'] for e in escenarios}
                    seleccion = st.selectbox("Selecciona un proyecto:", list(opciones.keys()))
                    
                    col_c, col_d = st.columns(2)
                    with col_c:
                        if st.button("📂 Cargar", type="primary", use_container_width=True):
                            id_sel = opciones[seleccion]
                            res_json = supabase.table("escenarios_guardados").select("estado_json").eq("id", id_sel).execute()
                            estado_recuperado = res_json.data[0]['estado_json']
                            with st.spinner("Inyectando variables..."):
                                for k, v in estado_recuperado.items(): st.session_state[k] = v
                                st.success("✅ Interfaz restaurada.")
                                time.sleep(1)
                                st.rerun()
                                
                    with col_d:
                        if st.button("🗑️ Borrar", type="secondary", use_container_width=True):
                            id_del = opciones[seleccion]
                            with st.spinner("Eliminando de la nube..."):
                                supabase.table("escenarios_guardados").delete().eq("id", id_del).execute()
                                st.warning("🗑️ Proyecto eliminado.")
                                time.sleep(1)
                                st.rerun()
                else:
                    st.info("No hay escenarios guardados aún.")
            except Exception as e:
                st.error(f"Error consultando base: {e}")

# ====================================================================
# --- 6. MOTORES DE CARGA ESPACIAL (EL MOTOR 4x4 CORREGIDO) ---
# ====================================================================
@st.cache_data(show_spinner=False, ttl=3600)
def cargar_maestro_cuencas():
    try:
        engine = db_manager.get_engine()
        # 🛠️ CORRECCIÓN: Folium exige EPSG:4326 (Latitud/Longitud) para dibujar
        return gpd.read_postgis("SELECT * FROM cuencas", engine, geom_col="geometry").to_crs("EPSG:4326")
    except Exception as e:
        st.error(f"Error cargando cuencas: {e}")
        return None

@st.cache_data(show_spinner=False, ttl=3600)
def cargar_maestro_municipios():
    try:
        engine = db_manager.get_engine()
        # 🛠️ CORRECCIÓN: Devolvemos EPSG:4326 para que el Gemelo Digital no se rompa
        gdf = gpd.read_postgis("SELECT * FROM municipios", engine, geom_col="geometry").to_crs("EPSG:4326")
        col_map = {}
        for col in gdf.columns:
            if col.lower() in ['mpio_cnmbr', 'municipio']: col_map[col] = 'MPIO_CNMBR'
            if col.lower() == 'dpto_cnmbr': col_map[col] = 'DPTO_CNMBR'
        if col_map: gdf = gdf.rename(columns=col_map)
        return gdf
    except Exception as e:
        return None

@st.cache_data(show_spinner=False, ttl=3600)
def cargar_estaciones_geometria():
    try:
        engine = db_manager.get_engine()
        q = text("SELECT id_estacion, altitud, ST_SetSRID(ST_MakePoint(CAST(longitud AS FLOAT), CAST(latitud AS FLOAT)), 4326) as geometry FROM estaciones WHERE latitud IS NOT NULL")
        # 🛠️ CORRECCIÓN: Mantenemos EPSG:4326
        return gpd.read_postgis(q, engine, geom_col="geometry").to_crs("EPSG:4326")
    except: return None

def encontrar_estaciones_en_mapa(gdf_zona, buffer_km):
    gdf_est = cargar_estaciones_geometria()
    if gdf_zona is None or gdf_zona.empty or gdf_est is None or gdf_est.empty:
        return [], 1500
    
    # 🛠️ MAGIA ESPACIAL: Convertimos a Metros (3116) SOLO para calcular el círculo exacto
    zona_metros = gdf_zona.to_crs("EPSG:3116")
    est_metros = gdf_est.to_crs("EPSG:3116")
    
    area_busqueda = zona_metros.geometry.unary_union.buffer(buffer_km * 1000)
    est_finales = est_metros[est_metros.geometry.within(area_busqueda)]
    
    ids = est_finales['id_estacion'].astype(str).tolist()
    alt = est_finales['altitud'].mean() if not est_finales.empty else 1500
    return ids, alt
    
# ====================================================================
# 🧠 MOTORES DE DESCARGA MAESTRA (DESDE SUPABASE STORAGE)
# ====================================================================
@st.cache_data(show_spinner=False, ttl=86400) # Descarga una vez al día (86400 seg)
def obtener_matriz_maestra_csv(url):
    """
    Descarga el CSV directamente desde el Storage de Supabase.
    Estandariza automáticamente todas las columnas a MAYÚSCULAS para evitar errores.
    """
    import pandas as pd
    import streamlit as st
    try:
        df = pd.read_csv(url)
        # Limpieza extrema: Quitamos espacios y pasamos todo a mayúsculas
        df.columns = df.columns.str.strip().str.upper()
        # Normalizamos la columna territorio para cruces perfectos
        if 'TERRITORIO' in df.columns:
            df['TERRITORIO_NORM'] = df['TERRITORIO'].astype(str).str.strip().str.upper()
        return df
    except Exception as e:
        st.sidebar.error(f"Error conectando al Storage de Supabase: {e}")
        return pd.DataFrame()

# ====================================================================
# 🧠 AUTO-CARGADOR DEL ALEPH (VERSIÓN 4.0 - SINGLE SOURCE OF TRUTH)
# ====================================================================
def auto_cargar_matrices_al_aleph(nombre_zona, nivel_agregacion):
    """
    Lee directamente los archivos maestros en la nube. Cero consultas SQL.
    A prueba de caídas, esquemas rotos o problemas de mayúsculas.
    """
    import streamlit as st
    
    url_demo = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/Matriz_Maestra_Demografica.csv"
    url_pecu = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/Matriz_Maestra_Pecuaria.csv"
    
    zona_upper = str(nombre_zona).strip().upper()

    # --- 1. CARGA DEMOGRÁFICA ---
    df_demo = obtener_matriz_maestra_csv(url_demo)
    
    if not df_demo.empty and 'TERRITORIO_NORM' in df_demo.columns:
        filtro_demo = df_demo[df_demo['TERRITORIO_NORM'] == zona_upper]
        
        if not filtro_demo.empty:
            fila = filtro_demo.iloc[0]
            # Usamos .get() buscando los nombres más lógicos, y si no están, pone 0
            st.session_state['aleph_pob_total'] = int(fila.get('POB_TOTAL', fila.get('POBLACION', 0)))
            st.session_state['aleph_pob_urbana'] = int(fila.get('POB_URBANA', fila.get('URBANA', 0)))
            st.session_state['aleph_pob_rural'] = int(fila.get('POB_RURAL', fila.get('RURAL', 0)))
        else:
            st.session_state['aleph_pob_total'], st.session_state['aleph_pob_urbana'], st.session_state['aleph_pob_rural'] = 0, 0, 0
    else:
        st.session_state['aleph_pob_total'], st.session_state['aleph_pob_urbana'], st.session_state['aleph_pob_rural'] = 0, 0, 0

    # --- 2. CARGA PECUARIA ---
    df_pecu = obtener_matriz_maestra_csv(url_pecu)
    
    st.session_state['ica_bovinos_calc_met'] = 0
    st.session_state['ica_porcinos_calc_met'] = 0
    st.session_state['ica_aves_calc_met'] = 0
    
    if not df_pecu.empty and 'TERRITORIO_NORM' in df_pecu.columns:
        filtro_pecu = df_pecu[df_pecu['TERRITORIO_NORM'] == zona_upper]
        
        if not filtro_pecu.empty:
            # Detectar si el CSV está ordenado por filas (Especie) o por columnas (Bovinos, Porcinos...)
            if 'ESPECIE' in filtro_pecu.columns:
                for _, row in filtro_pecu.iterrows():
                    esp = str(row['ESPECIE']).upper()
                    # Buscar la columna de cantidad (suele llamarse CABEZAS_ACTUALES, CANTIDAD, o TOTAL)
                    cabezas = int(row.get('CABEZAS_ACTUALES', row.get('CANTIDAD', row.get('TOTAL', 0))))
                    
                    if 'BOVINO' in esp: st.session_state['ica_bovinos_calc_met'] = cabezas
                    elif 'PORCINO' in esp: st.session_state['ica_porcinos_calc_met'] = cabezas
                    elif 'AVE' in esp: st.session_state['ica_aves_calc_met'] = cabezas
            else:
                # Si el CSV tiene una columna para cada animal
                fila_pec = filtro_pecu.iloc[0]
                st.session_state['ica_bovinos_calc_met'] = int(fila_pec.get('BOVINOS', 0))
                st.session_state['ica_porcinos_calc_met'] = int(fila_pec.get('PORCINOS', 0))
                st.session_state['ica_aves_calc_met'] = int(fila_pec.get('AVES', 0))
                
# ====================================================================
# --- 7. SELECTOR ESPACIAL PRINCIPAL ---
# ====================================================================
def render_selector_espacial():
    st.sidebar.markdown("### 📍 Filtros Geográficos Principales")

    def selectbox_seguro(label, opciones, clave):
        lista = list(opciones)
        if not lista: lista = ["-- NO HAY DATOS --"]
        idx = 0
        if clave in st.session_state and st.session_state[clave] in lista:
            idx = lista.index(st.session_state[clave])
        return st.sidebar.selectbox(label, lista, index=idx, key=clave)

    opciones_agr = ["Por Cuenca", "Por Municipio", "Por Región", "Departamento"]
    idx_agr = opciones_agr.index(st.session_state.get('mem_nivel_agregacion', "Por Cuenca")) if st.session_state.get('mem_nivel_agregacion') in opciones_agr else 0
    nivel_agregacion = st.sidebar.radio("Nivel de Agregación:", opciones_agr, index=idx_agr, key="radio_agregacion_mem")
    st.session_state['mem_nivel_agregacion'] = nivel_agregacion

    nombre_zona = "Sin Selección"
    gdf_zona = None
    
    if nivel_agregacion == "Por Cuenca":
        df_c = cargar_maestro_cuencas()
        if df_c is not None and not df_c.empty:
            ruta = selectbox_seguro("Ruta de Búsqueda:", ["Hidrología", "Administrativo"], "mem_ruta_busqueda")
            if ruta == "Hidrología":
                nivel = selectbox_seguro("1. Nivel a Evaluar:", ["NSS1", "NSS2", "NSS3", "SZH", "ZH", "AH"], "mem_nivel_hidro")
                col = {"NSS1":"nom_nss1", "NSS2":"nom_nss2", "NSS3":"nom_nss3", "SZH":"nom_szh", "ZH":"nomzh", "AH":"nomah"}.get(nivel)
                territorio = selectbox_seguro(f"🎯 Territorio ({nivel}):", sorted(df_c[col].dropna().unique()), "mem_terr_hidro")
                if territorio != "-- NO HAY DATOS --":
                    gdf_zona = df_c[df_c[col] == territorio]
                    nombre_zona = territorio
            else:
                nivel = selectbox_seguro("1. Nivel a Evaluar:", ["CAR", "Subregión"], "mem_nivel_admin")
                col = "CorpoAmb" if nivel == "CAR" else "depto_regi"
                territorio = selectbox_seguro(f"🎯 Territorio ({nivel}):", sorted(df_c[col].dropna().unique()), "mem_terr_admin")
                if territorio != "-- NO HAY DATOS --":
                    gdf_zona = df_c[df_c[col] == territorio]
                    nombre_zona = territorio

    else:
        df_m = cargar_maestro_municipios()
        if df_m is not None and not df_m.empty:
            if nivel_agregacion == "Por Municipio":
                mun = selectbox_seguro("Municipio:", sorted(df_m['MPIO_CNMBR'].dropna().unique()), "mem_mun_sel")
                if mun != "-- NO HAY DATOS --":
                    gdf_zona = df_m[df_m['MPIO_CNMBR'] == mun]
                    nombre_zona = mun
            elif nivel_agregacion == "Por Región" and 'subregion' in df_m.columns:
                reg = selectbox_seguro("Región:", sorted(df_m['subregion'].dropna().unique()), "mem_reg_sel")
                if reg != "-- NO HAY DATOS --":
                    gdf_zona = df_m[df_m['subregion'] == reg]
                    nombre_zona = reg
            elif nivel_agregacion == "Departamento" and 'DPTO_CNMBR' in df_m.columns:
                dep = selectbox_seguro("Departamento:", sorted(df_m['DPTO_CNMBR'].dropna().unique()), "mem_dep_sel")
                if dep != "-- NO HAY DATOS --":
                    gdf_zona = df_m[df_m['DPTO_CNMBR'] == dep]
                    nombre_zona = dep

    st.sidebar.markdown("---")
    buffer_km = st.sidebar.slider("Buffer (km):", 0.0, 100.0, float(st.session_state.get('buffer_global_km', 25.0)), 5.0, key="slider_buffer_mem")
    st.session_state['buffer_global_km'] = buffer_km
    st.session_state['aleph_lugar'] = decodificar_tildes(nombre_zona)

    ids_estaciones, alt_ref = encontrar_estaciones_en_mapa(gdf_zona, buffer_km)
    
    # 🚀 EJECUCIÓN DEL AUTO-CARGADOR RESTAURADO
    auto_cargar_matrices_al_aleph(decodificar_tildes(nombre_zona), nivel_agregacion)

    # Renderizamos el Gestor de Escenarios con el nombre de la zona actual
    renderizar_gestor_escenarios(decodificar_tildes(nombre_zona))

    return ids_estaciones, decodificar_tildes(nombre_zona), alt_ref, gdf_zona
