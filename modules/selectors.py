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
        
        # 📡 AUTO-FETCH SIN DATOS QUEMADOS (Sincronizado con enso_iri_prob.json)
        if 'aleph_iri_nino' not in st.session_state:
            try:
                # Importamos del motor IRI
                from modules.iri_api import get_iri_enso_forecast
                df_enso, _ = get_iri_enso_forecast()
                
                if df_enso is not None and not df_enso.empty:
                    # Extraemos la primera fila (trimestre actual) usando tus nombres de columnas
                    fila_actual = df_enso.iloc[0]
                    prob_nina = float(fila_actual.get('La Niña', 0))
                    prob_nino = float(fila_actual.get('El Niño', 0))
                    prob_neutro = float(fila_actual.get('Neutral', 0))
                    trimestre_actual = str(fila_actual.get('Trimestre', 'Actual'))
                    
                    # Lógica de fase para el ícono
                    if prob_nina > 50: estado = "Niña 🌧️"
                    elif prob_nino > 50: estado = "Niño ☀️"
                    else: estado = "Neutro ⚖️"
                    
                    st.session_state['enso_fase'] = estado
                    st.session_state['aleph_iri_nino'] = int(prob_nino)
                    st.session_state['aleph_iri_neutro'] = int(prob_neutro)
                    st.session_state['aleph_iri_nina'] = int(prob_nina)
                    st.session_state['aleph_iri_trimestre'] = trimestre_actual
                    st.session_state['aleph_iri_tendencia'] = "Sincronizado con Columbia University 📡"
                else:
                    raise ValueError("El archivo enso_iri_prob.json no entregó datos válidos.")
                    
            except Exception as e:
                # Registro del error real para monitoreo
                st.session_state['enso_fase'] = "Desconocido ⚠️"
                st.session_state['aleph_iri_nino'] = 0
                st.session_state['aleph_iri_neutro'] = 0
                st.session_state['aleph_iri_nina'] = 0
                st.session_state['aleph_iri_trimestre'] = "Falla"
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
        if st.button("🧹 Purgar Memoria", use_container_width=True):
            st.session_state.clear()
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
# 🌍 SELECTOR ESPACIAL MAESTRO (SQL ESTRICTO - SIN FANTASMAS)
# ====================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def cargar_mapa_cuencas():
    engine = db_manager.get_engine()
    return gpd.read_postgis("SELECT * FROM cuencas", engine, geom_col="geometry")

@st.cache_data(ttl=3600, show_spinner=False)
def cargar_mapa_municipios():
    engine = db_manager.get_engine()
    return gpd.read_postgis("SELECT * FROM municipios", engine, geom_col="geometry")

def render_selector_espacial():
    ids_estaciones = []
    nombre_zona = "Antioquia"
    altitud_ref = 1500.0
    gdf_zona = None
    nivel_jerarquico = "DEPARTAMENTO" # 🔥 FIX 1: Nueva variable para la Llave Universal
    
    try: engine = db_manager.get_engine()
    except Exception as e:
        st.error(f"Error conectando a BD: {e}")
        return ids_estaciones, nombre_zona, altitud_ref, gdf_zona
    
    with st.sidebar.expander("📍 Filtros Geográficos Principales", expanded=True):
        modo = st.radio("Nivel de Agregación:", ["Por Cuenca", "Por Municipio", "Por Región", "Departamento"], index=0)
        
        # --- A. POR CUENCA (DISEÑO INTUITIVO) ---
        if modo == "Por Cuenca":
            gdf_c = cargar_mapa_cuencas()
            ruta = st.selectbox("Ruta de Búsqueda:", ["Hidrología", "CAR"], index=0)
            
            if ruta == "Hidrología":
                # 1. Primero define QUÉ nivel quiere evaluar
                nivel = st.selectbox("1. Nivel a Evaluar:", ["AH", "ZH", "SZH", "NSS1", "NSS2", "NSS3"], index=5)
                col_obj = {"AH": "nomah", "ZH": "nomzh", "SZH": "nom_szh", "NSS1": "nom_nss1", "NSS2": "nom_nss2", "NSS3": "nom_nss3"}[nivel]
                
                st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
                st.markdown("<span style='font-size:0.85em; color:gray;'>Filtros Opcionales de Búsqueda:</span>", unsafe_allow_html=True)
                
                df_f = gdf_c
                # Solo mostramos los filtros superiores al nivel elegido
                if nivel in ["ZH", "SZH", "NSS1", "NSS2", "NSS3"]:
                    ah = st.selectbox("Filtro AH:", ["-- TODAS --"] + sorted(df_f['nomah'].dropna().unique()))
                    if ah != "-- TODAS --": df_f = df_f[df_f['nomah']==ah]
                    
                if nivel in ["SZH", "NSS1", "NSS2", "NSS3"]:
                    zh = st.selectbox("Filtro ZH:", ["-- TODAS --"] + sorted(df_f['nomzh'].dropna().unique()))
                    if zh != "-- TODAS --": df_f = df_f[df_f['nomzh']==zh]
                    
                if nivel in ["NSS1", "NSS2", "NSS3"]:
                    szh = st.selectbox("Filtro SZH:", ["-- TODAS --"] + sorted(df_f['nom_szh'].dropna().unique()))
                    if szh != "-- TODAS --": df_f = df_f[df_f['nom_szh']==szh]
                    
                st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
                
                # 2. Selección FINAL obligatoria
                sel_fin = st.selectbox(f"🎯 2. Territorio Exacto ({nivel}):", ["-- Seleccione --"] + sorted(df_f[col_obj].dropna().unique()))
                
                if sel_fin != "-- Seleccione --":
                    nombre_zona = sel_fin
                    gdf_zona = df_f[df_f[col_obj]==sel_fin]
                    nivel_jerarquico = nivel 
                else:
                    nombre_zona, gdf_zona = "-- Seleccione --", None
                    nivel_jerarquico = "NINGUNO"
            
            elif ruta == "CAR":
                # 🔥 FIX AMVA: Leemos las CARs directamente de la Matriz de Toma de Decisiones, 
                # así garantizamos que "AMVA" siempre aparezca en el menú.
                try:
                    q_cars = text("SELECT DISTINCT territorio FROM matriz_maestra_hidrologia WHERE UPPER(nivel) = 'CAR' ORDER BY territorio")
                    df_cars = pd.read_sql(q_cars, db_manager.get_engine())
                    opciones_car = df_cars['territorio'].tolist() if not df_cars.empty else ["AMVA", "CORANTIOQUIA", "CORNARE", "CORPOURABA"]
                except:
                    opciones_car = ["AMVA", "CORANTIOQUIA", "CORNARE", "CORPOURABA"]

                car_sel = st.selectbox("Autoridad Ambiental (CAR):", ["-- Seleccione --"] + sorted(opciones_car))
                
                if car_sel != "-- Seleccione --":
                    # Mapeamos las geometrías de las CAR (Asegurando que el AMVA tenga geometría)
                    if car_sel == "AMVA":
                        mask_car = gdf_c['corpoamb'].str.contains('AMVA|ABURR|METROPOLITANA', case=False, na=False)
                    else:
                        mask_car = gdf_c['corpoamb'].str.contains(car_sel[:4], case=False, na=False)
                    
                    df_f = gdf_c[mask_car]
                    
                    # Salvavidas de diseño: Si la CAR no está dibujada en el mapa físico, devolvemos el mapa completo para no romper la app
                    if df_f.empty: 
                        df_f = gdf_c
                    
                    # Permite evaluar toda la jurisdicción de la CAR o bajar a sus microcuencas
                    nivel = st.selectbox("1. Resolución a Evaluar:", ["CAR", "NSS1", "NSS2", "NSS3"], index=0)
                    
                    if nivel == "CAR":
                        nombre_zona = car_sel
                        gdf_zona = df_f if not df_f.empty else None
                        nivel_jerarquico = "CAR"
                    else:
                        col_obj = {"NSS1": "nom_nss1", "NSS2": "nom_nss2", "NSS3": "nom_nss3"}[nivel]
                        sel_fin = st.selectbox(f"🎯 2. Territorio Exacto ({nivel}):", ["-- Seleccione --"] + sorted(df_f[col_obj].dropna().unique()))
                        
                        if sel_fin != "-- Seleccione --":
                            nombre_zona = sel_fin
                            gdf_zona = df_f[df_f[col_obj]==sel_fin]
                            nivel_jerarquico = nivel 
                        else:
                            nombre_zona, gdf_zona = "-- Seleccione --", None
                            nivel_jerarquico = "NINGUNO"
                else:
                    nombre_zona, gdf_zona = "-- Seleccione --", None
                    nivel_jerarquico = "NINGUNO"
                    
        # --- B. POR REGIÓN ---
        elif modo == "Por Región":
            try:
                # 🔥 FIX: Forzamos la lectura con openpyxl y usamos la columna correcta 'subregion'
                df_m = pd.read_excel("https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/territorio_maestro.xlsx", engine='openpyxl')
                df_m.columns = [c.lower() for c in df_m.columns]
                
                col_reg = 'subregion' if 'subregion' in df_m.columns else 'region'
                lista_reg = sorted([str(r).title() for r in df_m[col_reg].dropna().unique()])
                
                sel_reg = st.selectbox("📍 Región:", ["-- Seleccione --"] + lista_reg)
                if sel_reg != "-- Seleccione --":
                    nombre_zona = sel_reg # Enviamos el nombre limpio sin prefijo para que SQL haga match
                    nivel_jerarquico = "Regional" # Coincide exacto con la Matriz Demográfica
                    
                    cods = df_m[df_m[col_reg].str.lower()==sel_reg.lower()]['dp_mp'].astype(str).str.replace(".0", "", regex=False).str.zfill(5).tolist()
                    gdf_mun = cargar_mapa_municipios()
                    
                    col_mpio = 'mpio_ccdgo' if 'mpio_ccdgo' in gdf_mun.columns else 'MPIO_CCDGO'
                    gdf_zona = gdf_mun[gdf_mun[col_mpio].astype(str).str.replace(".0", "", regex=False).str.zfill(5).isin(cods)]
                    
                    # Escudo anti-pantalla blanca: si el filtro geográfico falla, evita que muera la app
                    if gdf_zona.empty: gdf_zona = gdf_mun.head(1) 
                else:
                    nombre_zona, gdf_zona = "-- Seleccione --", None
                    nivel_jerarquico = "NINGUNO"
            except Exception as e: 
                st.error(f"Error cargando regiones: {e}")
                nombre_zona, gdf_zona = "-- Seleccione --", None
                nivel_jerarquico = "NINGUNO"

        # --- C. POR MUNICIPIO ---
        elif modo == "Por Municipio":
            gdf_mun = cargar_mapa_municipios()
            # 🔥 FIX: Usamos la función decodificar_tildes que ya existe arriba en este mismo archivo
            try:
                col_nombre = 'mpio_cnmbr' if 'mpio_cnmbr' in gdf_mun.columns else 'MPIO_CNMBR'
                gdf_mun['display'] = gdf_mun[col_nombre].apply(decodificar_tildes).str.title()
            except:
                gdf_mun['display'] = gdf_mun[col_nombre].str.title()
            
            sel_mpio = st.selectbox("🏢 Municipio:", ["-- Seleccione --"] + sorted(gdf_mun['display'].unique()))
            if sel_mpio != "-- Seleccione --":
                nombre_zona, gdf_zona = sel_mpio, gdf_mun[gdf_mun['display']==sel_mpio]
                nivel_jerarquico = "Municipal" 
            else:
                nombre_zona, gdf_zona = "-- Seleccione --", None
                nivel_jerarquico = "NINGUNO"

        # --- D. DEPARTAMENTO ---
        else:
            gdf_mun = cargar_mapa_municipios()
            nombre_zona, gdf_zona = "Antioquia", gpd.GeoDataFrame({'nombre':['Antioquia']}, geometry=[gdf_mun.unary_union], crs=gdf_mun.crs)
            nivel_jerarquico = "Departamental" # 🔥 FIX 5

        # --- FILTRO DE ESTACIONES ---
        if gdf_zona is not None and not gdf_zona.empty:
            buff = st.slider("Buffer (km):", 0.0, 50.0, 25.0)
            minx, miny, maxx, maxy = gdf_zona.to_crs(4326).total_bounds
            q = text(f"SELECT id_estacion, altitud FROM estaciones WHERE longitud BETWEEN {minx-0.2} AND {maxx+0.2} AND latitud BETWEEN {miny-0.2} AND {maxy+0.2}")
            df_est = pd.read_sql(q, engine)
            ids_estaciones = df_est['id_estacion'].astype(str).tolist()
            altitud_ref = df_est['altitud'].mean()

    # ====================================================================
    # 🧠 ORQUESTADOR SILENCIOSO (Equipado con Llave Universal)
    # ====================================================================
    # 🔥 FIX 1: Eliminamos "Antioquia" de la lista de ignorados
    zonas_ignoradas = ["Bloque Regional", "-- TODAS --", "-- Seleccione --", "", "NINGUNO"]
    
    zona_activa = st.session_state.get('zona_activa_global')
    
    # 🔥 FIX 2: Permitimos que el nivel se actualice para Antioquia y Regiones
    if nombre_zona not in zonas_ignoradas and nombre_zona != zona_activa:
        st.session_state['zona_activa_global'] = nombre_zona
        st.session_state['nivel_activo_global'] = nivel_jerarquico 
        
        # Limpieza de estados para nueva carga
        claves_a_borrar = [
            'pob_hum_calc_met', 'ica_bovinos_calc_met', 'ica_porcinos_calc_met', 'ica_aves_calc_met', 
            'demanda_total_m3s', 'carga_dbo_total_ton',
            'aleph_oferta_m3s', 'aleph_lluvia_mm', 'aleph_area_km2', 'aleph_recarga_mm'
        ]
        for k in claves_a_borrar: st.session_state.pop(k, None)
            
        try:
            from modules.utils import obtener_metabolismo_exacto
            datos_precargados = obtener_metabolismo_exacto(nombre_zona, 2025)
            if datos_precargados:
                st.session_state['aleph_pob_total'] = datos_precargados.get('pob_total', 0)
                st.session_state['ica_bovinos_calc_met'] = datos_precargados.get('bovinos', 0)
                st.session_state['ica_porcinos_calc_met'] = datos_precargados.get('porcinos', 0)
                st.session_state['ica_aves_calc_met'] = datos_precargados.get('aves', 0)
                st.session_state['aleph_lugar'] = nombre_zona
        except: pass 
            
        try:
            # 🔥 FIX 7: BÚSQUEDA EXACTA CON LLAVE UNIVERSAL
            q_hidro = text("SELECT * FROM matriz_hidrologica_maestra WHERE \"Jerarquia\" = :nivel AND \"Territorio\" = :zona LIMIT 1")
            df_hidro = pd.read_sql(q_hidro, engine, params={"nivel": nivel_jerarquico, "zona": nombre_zona})
            
            if not df_hidro.empty:
                row_h = df_hidro.iloc[0]
                st.session_state['aleph_oferta_m3s'] = float(row_h['Caudal_Medio_m3s'])
                st.session_state['aleph_lluvia_mm'] = float(row_h['Lluvia_mm'])
                st.session_state['aleph_area_km2'] = float(row_h['Area_km2'])
                st.session_state['aleph_recarga_mm'] = float(row_h['Recarga_mm'])
                st.session_state['aleph_altitud_m'] = float(row_h['Altitud_m'])
        except Exception as e:
            pass

    renderizar_gestor_escenarios(nombre_zona)
    return ids_estaciones, nombre_zona, altitud_ref, gdf_zona


