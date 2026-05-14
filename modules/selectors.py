# modules/selectors.py

import streamlit as st
import geopandas as gpd
import pandas as pd
from sqlalchemy import text
from shapely.geometry import box
import json
import time
import unicodedata

from modules import db_manager
from modules.config import Config

# ====================================================================
# --- REPARADOR DE TILDES (MOJIBAKE) ---
# ====================================================================
def decodificar_tildes(texto):
    """Corrige errores de codificación en BD como 'Abriaquã' -> 'Abriaquí'"""
    if not isinstance(texto, str): return texto
    try:
        if 'Ã' in texto or 'ã' in texto or '\x8d' in texto:
            return texto.encode('latin1').decode('utf-8')
    except: pass
    return texto

def renderizar_telemetria_aleph():
    """
    Panel de control universal que monitorea las variables de estado en tiempo real.
    Se inyecta en la barra lateral de todas las páginas.
    """
    st.sidebar.markdown("---")
    
    # 🧠 EL ALEPH: Monitor de Memoria Activa
    with st.sidebar.expander("🧠 Telemetría del Aleph", expanded=True):
        
        # 1. Monitoreo Demográfico (CORREGIDO: Ahora busca ambas "cajas")
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
                    
                    # Lógica de fase basada en los números reales (80% Neutro -> Neutro)
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
            st.cache_data.clear() # <- Esta es la línea mágica que rompe el congelamiento
            st.rerun()
            
# ====================================================================
# 📂 NAVEGACIÓN GLOBAL
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
        st.page_link("pages/09_🧠_Toma_de_Decisiones.py", label="Toma de Decisiones", icon="🧠") # <-- Actualizado el icono y nombre
        st.page_link("pages/10_👑_Panel_Administracion.py", label="Panel Administración", icon="⚙️")
        st.page_link("pages/11_⚙️_Generador.py", label="Generador", icon="✨")
        st.page_link("pages/12_📚_Ayuda_y_Docs.py", label="Ayuda y Docs", icon="📚")
        st.page_link("pages/13_🕵️_Detective.py", label="Detective", icon="🕵️")
        st.page_link("pages/14_🛰️_Satelite_En_Vivo.py", label="Satélite en Vivo", icon="🛰️")
    
    # 🚀 AQUÍ OCURRE LA MAGIA: Llamamos a la telemetría para que se dibuje
    renderizar_telemetria_aleph()

# ====================================================================
# ☁️ CONEXIÓN A SUPABASE (Para guardar los JSON)
# ====================================================================
@st.cache_resource
def get_supabase_client():
    try:
        from supabase import create_client
        
        # 1. Buscamos primero si están sueltas (sin corchetes)
        url_sb = st.secrets.get("SUPABASE_URL")
        key_sb = st.secrets.get("SUPABASE_KEY")
        
        # 2. Si no están sueltas, buscamos dentro de [supabase] con mayúsculas
        if not url_sb:
            url_sb = st.secrets.get("supabase", {}).get("SUPABASE_URL")
        if not key_sb:
            key_sb = st.secrets.get("supabase", {}).get("SUPABASE_KEY")
            
        # 3. Por si acaso, buscamos dentro de [supabase] con minúsculas (url / key)
        if not url_sb:
            url_sb = st.secrets.get("supabase", {}).get("url")
        if not key_sb:
            key_sb = st.secrets.get("supabase", {}).get("key")

        # Intentamos conectar si encontramos ambas
        if url_sb and key_sb:
            return create_client(url_sb, key_sb)
        else:
            return "NO_SECRETS"
            
    except ImportError:
        return "NO_LIBRARY"
    except Exception as e:
        return str(e)

# ====================================================================
# 💾 GESTOR DE SNAPSHOTS (GUARDAR, CARGAR Y ELIMINAR ESCENARIOS)
# ====================================================================
def renderizar_gestor_escenarios(nombre_zona_actual):
    st.sidebar.markdown("---")
    with st.sidebar.expander("💾 Gestor de Escenarios (Snapshots)", expanded=False):
        supabase = get_supabase_client()
        
        if supabase == "NO_SECRETS" or supabase == "NO_LIBRARY" or isinstance(supabase, str) or not supabase:
            st.error("Error de conexión a Supabase. Revisa tus credenciales.")
            return

        tab_guardar, tab_cargar = st.tabs(["Guardar", "Cargar / Eliminar"])
        
        # --- TAB GUARDAR ---
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

        # --- TAB CARGAR / ELIMINAR ---
        with tab_cargar:
            try:
                res = supabase.table("escenarios_guardados").select("id, nombre_escenario, territorio, fecha_creacion").order("fecha_creacion", desc=True).execute()
                escenarios = res.data
                if escenarios:
                    opciones = {f"{e['nombre_escenario']} ({e['territorio']})": e['id'] for e in escenarios}
                    seleccion = st.selectbox("Selecciona un proyecto:", list(opciones.keys()))
                    
                    # 🔥 NUEVO: Dos botones paralelos para Cargar o Borrar
                    col_c, col_d = st.columns(2)
                    
                    with col_c:
                        if st.button("📂 Cargar", type="primary", use_container_width=True):
                            id_sel = opciones[seleccion]
                            res_json = supabase.table("escenarios_guardados").select("estado_json").eq("id", id_sel).execute()
                            estado_recuperado = res_json.data[0]['estado_json']
                            with st.spinner("Inyectando variables..."):
                                for k, v in estado_recuperado.items():
                                    st.session_state[k] = v
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
# 🌍 SELECTOR ESPACIAL MAESTRO (SQL ESTRICTO - GEOMETRÍAS)
# ====================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def cargar_mapa_cuencas():
    from modules.db_manager import get_engine
    import geopandas as gpd
    engine = get_engine()
    return gpd.read_postgis("SELECT * FROM cuencas", engine, geom_col="geometry")

@st.cache_data(ttl=3600, show_spinner=False)
def cargar_mapa_municipios():
    from modules.db_manager import get_engine
    import geopandas as gpd
    engine = get_engine()
    return gpd.read_postgis("SELECT * FROM municipios", engine, geom_col="geometry")

# ====================================================================
# --- MOTORES DE CARGA DE METADATOS (PUNTO CERO - SQL DIRECTO) ---
# ====================================================================
@st.cache_data(show_spinner=False, ttl=3600)
def cargar_maestro_cuencas():
    """Carga el inventario de subcuencas directamente desde PostgreSQL."""
    try:
        from modules.db_manager import get_engine
        import pandas as pd
        engine = get_engine()
        return pd.read_sql("SELECT * FROM cuencas", engine)
    except Exception as e:
        st.error(f"Error conectando a BD de cuencas: {e}")
        return None

@st.cache_data(show_spinner=False, ttl=3600)
def cargar_maestro_municipios():
    """Carga el inventario de municipios directamente desde PostgreSQL."""
    try:
        from modules.db_manager import get_engine
        import pandas as pd
        engine = get_engine()
        df = pd.read_sql("SELECT * FROM municipios", engine)
        
        # 🛡️ Escudo de Compatibilidad para el Punto Cero
        col_map = {}
        for col in df.columns:
            if col.lower() == 'mpio_cnmbr': col_map[col] = 'MPIO_CNMBR'
            if col.lower() == 'dpto_cnmbr': col_map[col] = 'DPTO_CNMBR'
        
        if col_map:
            df = df.rename(columns=col_map)
            
        return df
    except Exception as e:
        st.error(f"Error conectando a BD de municipios: {e}")
        return None

# ====================================================================

def render_selector_espacial():
    """
    Renderiza el panel lateral para filtrar estaciones base por Cuencas o Administrativo.
    🔥 MICROCIRUGÍA APLICADA: Mantiene toda la lógica original, pero añade memoria.
    """
    # renderizar_telemetria_aleph()  # Comentado si ya se llama en la navegación para evitar error de duplicados
    st.sidebar.markdown("### 📍 Filtros Geográficos Principales")

    # =========================================================
    # 🧠 ESCUDO DE MEMORIA INTELIGENTE (No rompe listas vacías)
    # =========================================================
    def selectbox_seguro(label, opciones, clave):
        lista = list(opciones)
        if not lista: lista = ["-- NO HAY DATOS --"]
        idx = 0
        if clave in st.session_state and st.session_state[clave] in lista:
            idx = lista.index(st.session_state[clave])
        return st.sidebar.selectbox(label, lista, index=idx, key=clave)

    # 1. Nivel de Agregación
    opciones_agr = ["Por Cuenca", "Por Municipio", "Por Región", "Departamento"]
    idx_agr = 0
    if 'mem_nivel_agregacion' in st.session_state and st.session_state['mem_nivel_agregacion'] in opciones_agr:
        idx_agr = opciones_agr.index(st.session_state['mem_nivel_agregacion'])
    
    nivel_agregacion = st.sidebar.radio("Nivel de Agregación:", opciones_agr, index=idx_agr, key="radio_agregacion_mem")
    st.session_state['mem_nivel_agregacion'] = nivel_agregacion

    ids_estaciones = []
    nombre_zona = "Sin Selección"
    altitud_ref = 1500
    gdf_zona = None

    # --- LÓGICA POR CUENCAS ---
    if nivel_agregacion == "Por Cuenca":
        df_cuencas = cargar_maestro_cuencas()
        if df_cuencas is not None and not df_cuencas.empty:
            
            ruta_busqueda = selectbox_seguro("Ruta de Búsqueda:", ["Hidrología", "Administrativo"], "mem_ruta_busqueda")
            
            if ruta_busqueda == "Hidrología":
                nivel_evaluar = selectbox_seguro("1. Nivel a Evaluar:", ["NSS1", "NSS2", "NSS3", "SZH", "ZH", "AH"], "mem_nivel_evaluar_hidro")

                with st.sidebar.expander("Filtros Opcionales de Búsqueda:", expanded=False):
                    lista_ah = ["-- TODAS --"] + sorted(df_cuencas['nomah'].dropna().unique().tolist())
                    filtro_ah = selectbox_seguro("Filtro AH:", lista_ah, "mem_filtro_ah")
                    
                    df_opciones = df_cuencas if filtro_ah == "-- TODAS --" else df_cuencas[df_cuencas['nomah'] == filtro_ah]
                    
                    lista_zh = ["-- TODAS --"] + sorted(df_opciones['nomzh'].dropna().unique().tolist())
                    filtro_zh = selectbox_seguro("Filtro ZH:", lista_zh, "mem_filtro_zh")
                    
                    if filtro_zh != "-- TODAS --": df_opciones = df_opciones[df_opciones['nomzh'] == filtro_zh]
                    
                    lista_szh = ["-- TODAS --"] + sorted(df_opciones['nom_szh'].dropna().unique().tolist())
                    filtro_szh = selectbox_seguro("Filtro SZH:", lista_szh, "mem_filtro_szh")
                    
                    if filtro_szh != "-- TODAS --": df_opciones = df_opciones[df_opciones['nom_szh'] == filtro_szh]

                columna_nombre = {
                    "NSS1": "nom_nss1", "NSS2": "nom_nss2", "NSS3": "nom_nss3",
                    "SZH": "nom_szh", "ZH": "nomzh", "AH": "nomah"
                }
                
                col_busqueda = columna_nombre.get(nivel_evaluar, "nom_nss1")
                lista_territorios = sorted(df_opciones[col_busqueda].dropna().unique().tolist())
                
                territorio_sel = selectbox_seguro(f"🎯 2. Territorio Exacto ({nivel_evaluar}):", lista_territorios, "mem_terr_exacto_hidro")
                
                if territorio_sel and territorio_sel != "-- NO HAY DATOS --":
                    nombre_zona = decodificar_tildes(territorio_sel)
                    df_final = df_cuencas[df_cuencas[col_busqueda] == territorio_sel]
                    
                    ids_str = df_final['estaciones_id'].dropna().astype(str).tolist()
                    ids_estaciones = []
                    for item in ids_str:
                        if item.strip():
                            ids_estaciones.extend([x.strip() for x in item.split(',') if x.strip()])
                    ids_estaciones = list(set(ids_estaciones))
                    altitud_ref = df_final['altitud_media'].mean() if 'altitud_media' in df_final.columns else 1500
                    
                    from modules.db_manager import get_engine
                    try:
                        engine = get_engine()
                        col_geom = col_busqueda.replace('nom_', '')
                        query = text(f"SELECT geometry FROM cuencas WHERE {col_busqueda} = :val")
                        gdf_z = gpd.read_postgis(query, engine, params={"val": territorio_sel}, geom_col="geometry")
                        if not gdf_z.empty: gdf_zona = gdf_z.dissolve()
                    except: pass

            elif ruta_busqueda == "Administrativo":
                nivel_evaluar = selectbox_seguro("1. Nivel a Evaluar:", ["CAR", "Subregión", "Departamento"], "mem_nivel_evaluar_admin")
                
                col_busqueda = {"CAR": "CorpoAmb", "Subregión": "depto_regi", "Departamento": "departamen"}.get(nivel_evaluar, "CorpoAmb")
                lista_territorios = sorted(df_cuencas[col_busqueda].dropna().unique().tolist())
                
                territorio_sel = selectbox_seguro(f"🎯 2. Territorio Exacto ({nivel_evaluar}):", lista_territorios, "mem_terr_exacto_admin")
                
                if territorio_sel and territorio_sel != "-- NO HAY DATOS --":
                    nombre_zona = decodificar_tildes(territorio_sel)
                    df_final = df_cuencas[df_cuencas[col_busqueda] == territorio_sel]
                    
                    ids_str = df_final['estaciones_id'].dropna().astype(str).tolist()
                    ids_estaciones = []
                    for item in ids_str:
                        if item.strip():
                            ids_estaciones.extend([x.strip() for x in item.split(',') if x.strip()])
                    ids_estaciones = list(set(ids_estaciones))
                    altitud_ref = df_final['altitud_media'].mean() if 'altitud_media' in df_final.columns else 1500

    # --- LÓGICA POR MUNICIPIO ---
    elif nivel_agregacion == "Por Municipio":
        df_mun = cargar_maestro_municipios()
        if df_mun is not None and not df_mun.empty:
            
            lista_municipios = sorted(df_mun['MPIO_CNMBR'].dropna().unique().tolist())
            municipio_sel = selectbox_seguro("Municipio:", lista_municipios, "mem_municipio_sel")
            
            if municipio_sel and municipio_sel != "-- NO HAY DATOS --":
                nombre_zona = decodificar_tildes(municipio_sel)
                df_final = df_mun[df_mun['MPIO_CNMBR'] == municipio_sel]
                
                ids_str = df_final['estaciones_id'].dropna().astype(str).tolist()
                ids_estaciones = []
                for item in ids_str:
                    if item.strip():
                        ids_estaciones.extend([x.strip() for x in item.split(',') if x.strip()])
                ids_estaciones = list(set(ids_estaciones))
                altitud_ref = df_final['altitud_media'].mean() if 'altitud_media' in df_final.columns else 1500

                try:
                    from modules.db_manager import get_engine
                    engine = get_engine()
                    query = text("SELECT geometry FROM municipios WHERE \"MPIO_CNMBR\" = :val")
                    gdf_z = gpd.read_postgis(query, engine, params={"val": municipio_sel}, geom_col="geometry")
                    if not gdf_z.empty: gdf_zona = gdf_z.dissolve()
                except: pass

    # --- LÓGICA POR REGIÓN ---
    elif nivel_agregacion == "Por Región":
        df_mun = cargar_maestro_municipios()
        if df_mun is not None and not df_mun.empty:
            
            lista_regiones = sorted(df_mun['subregion'].dropna().unique().tolist())
            region_sel = selectbox_seguro("Región:", lista_regiones, "mem_region_sel")
            
            if region_sel and region_sel != "-- NO HAY DATOS --":
                nombre_zona = decodificar_tildes(region_sel)
                df_final = df_mun[df_mun['subregion'] == region_sel]
                
                ids_str = df_final['estaciones_id'].dropna().astype(str).tolist()
                ids_estaciones = []
                for item in ids_str:
                    if item.strip():
                        ids_estaciones.extend([x.strip() for x in item.split(',') if x.strip()])
                ids_estaciones = list(set(ids_estaciones))
                altitud_ref = df_final['altitud_media'].mean() if 'altitud_media' in df_final.columns else 1500

    # --- LÓGICA POR DEPARTAMENTO ---
    elif nivel_agregacion == "Departamento":
        df_mun = cargar_maestro_municipios()
        if df_mun is not None and not df_mun.empty:
            
            lista_deptos = sorted(df_mun['DPTO_CNMBR'].dropna().unique().tolist())
            depto_sel = selectbox_seguro("Departamento:", lista_deptos, "mem_depto_sel")
            
            if depto_sel and depto_sel != "-- NO HAY DATOS --":
                nombre_zona = decodificar_tildes(depto_sel)
                df_final = df_mun[df_mun['DPTO_CNMBR'] == depto_sel]
                
                ids_str = df_final['estaciones_id'].dropna().astype(str).tolist()
                ids_estaciones = []
                for item in ids_str:
                    if item.strip():
                        ids_estaciones.extend([x.strip() for x in item.split(',') if x.strip()])
                ids_estaciones = list(set(ids_estaciones))
                altitud_ref = df_final['altitud_media'].mean() if 'altitud_media' in df_final.columns else 1500

    st.sidebar.markdown("---")
    
    # 🧠 MEMORIA DEL SLIDER DE BUFFER
    buffer_def = float(st.session_state.get('buffer_global_km', 25.0))
    buffer_km = st.sidebar.slider("Buffer (km):", 0.0, 100.0, buffer_def, 5.0, key="slider_buffer_mem")
    st.session_state['buffer_global_km'] = buffer_km
    
    # Guardamos el nombre final
    st.session_state['aleph_lugar'] = nombre_zona

    return ids_estaciones, nombre_zona, altitud_ref, gdf_zona
