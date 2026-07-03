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

def renderizar_telemetria_aleph():
    """Panel de control universal que monitorea las variables de estado en tiempo real."""
    import streamlit as st
    
    st.sidebar.markdown("### 🧠 Telemetría del Aleph")
    
    # --- CAJA DESPLEGABLE 1: MÉTRICAS TERRITORIALES Y CARGAS ---
    with st.sidebar.expander("📊 Conexiones Multimodelo", expanded=True):
        
        # Sensor Demográfico
        pob_viva = st.session_state.get('aleph_pob_total', st.session_state.get('poblacion_servida', 0))
        if pob_viva > 0:
            st.markdown(f"👥 **Población:** <span style='color:#2ecc71'>Sincronizada ({pob_viva:,.0f} hab)</span>", unsafe_allow_html=True)
        else:
            st.markdown("👥 **Población:** <span style='color:#e74c3c'>Vacía (Default)</span>", unsafe_allow_html=True)
            
        # Sensor Hidrológico
        oferta_viva = st.session_state.get('aleph_oferta_m3s', 0.0)
        if oferta_viva > 0:
            st.markdown(f"🌧️ **Hidrología:** <span style='color:#3498db'>Enlazada ({oferta_viva:,.3f} m³/s)</span>", unsafe_allow_html=True)
        else:
            st.markdown("🌧️ **Hidrología:** <span style='color:#95a5a6'>Inactiva</span>", unsafe_allow_html=True)

        # Sensor RURH (Concesiones y Extracciones)
        rurh_viva = st.session_state.get('aleph_concesiones_m3s', 0.0)
        if rurh_viva > 0:
            st.markdown(f"🏭 **RURH (Extracción):** <span style='color:#e67e22'>Activa ({rurh_viva:,.3f} m³/s)</span>", unsafe_allow_html=True)
        else:
            st.markdown("🏭 **RURH (Extracción):** <span style='color:#95a5a6'>Inactiva</span>", unsafe_allow_html=True)
            
        # Sensor Calidad (Tormenta Eliminada)
        dbo_viva = st.session_state.get('carga_dbo_total_ton', 0.0)
        if dbo_viva > 0:
            st.markdown(f"☣️ **Carga DBO:** <span style='color:#8e44ad'>Registrada ({dbo_viva:,.0f} Ton)</span>", unsafe_allow_html=True)
        else:
            st.markdown("☣️ **Carga DBO:** <span style='color:#95a5a6'>Inactiva</span>", unsafe_allow_html=True)
            
    # --- CAJA DESPLEGABLE 2: PULSO CLIMÁTICO GLOBAL (ENSO) ---
    with st.sidebar.expander("🌦️ Pulso Climático Global", expanded=True):
        if 'aleph_iri_nino' not in st.session_state:
            try:
                from modules.climate_api import get_iri_enso_forecast
                df_enso, _ = get_iri_enso_forecast()
                if df_enso is not None and not df_enso.empty:
                    # 🚀 ACTUALIZADO a MJJ
                    df_target = df_enso[df_enso['Trimestre'].str.contains('MJJ', na=False)]
                    fila_actual = df_target.iloc[0] if not df_target.empty else df_enso.iloc[0]
                    
                    p_nina = float(fila_actual.get('La Niña', 0))
                    p_nino = float(fila_actual.get('El Niño', 0))
                    p_neutro = float(fila_actual.get('Neutral', 0))
                    trim_txt = str(fila_actual.get('Trimestre', 'Actual'))
                    
                    if p_nina > 50: estado = "Niña 🌧️"
                    elif p_nino > 50: estado = "Niño ☀️"
                    else: estado = "Neutro ⚖️"
                    
                    st.session_state['enso_fase'] = estado
                    st.session_state['aleph_iri_nino'] = int(p_nino)
                    st.session_state['aleph_iri_neutro'] = int(p_neutro)
                    st.session_state['aleph_iri_nina'] = int(p_nina)
                    st.session_state['aleph_iri_trimestre'] = trim_txt
                    # 🚀 ACTUALIZADO a MJJ
                    st.session_state['aleph_iri_tendencia'] = "Sincronización MJJ Exitosa 📡"
            except Exception as e:
                st.session_state['aleph_iri_tendencia'] = f"Error: {str(e)}"
                
        enso_global = st.session_state.get('enso_fase', 'Desconocido ⚠️')
        color_enso = "#3498db" if "Niña" in enso_global else "#e74c3c" if "Niño" in enso_global else "#f39c12" if "Desconocido" in enso_global else "#2ecc71"
        st.markdown(f"🌍 **Clima ENSO:** <span style='color:{color_enso}'><b>{enso_global}</b></span>", unsafe_allow_html=True)
        st.caption("📡 **Pronóstico IRI (Memoria Activa)**")
        
        p_nino = st.session_state.get('aleph_iri_nino', 0)
        p_neutro = st.session_state.get('aleph_iri_neutro', 0)
        p_nina = st.session_state.get('aleph_iri_nina', 0)
        trim = st.session_state.get('aleph_iri_trimestre', 'Sin datos')
        tendencia = st.session_state.get('aleph_iri_tendencia', '')
        
        if (p_nino + p_neutro + p_nina) > 0:
            st.caption(f"Probabilidad ({trim}):")
            st.progress(p_nino, text=f"☀️ El Niño ({p_nino}%)")
            st.progress(p_neutro, text=f"⚖️ Neutro ({p_neutro}%)")
            st.progress(p_nina, text=f"🌧️ La Niña ({p_nina}%)")
        
        if tendencia: st.caption(f"📈 **Estado:** {tendencia}")

    # --- BOTÓN DE PURGA (Fuera de los expanders) ---
    st.sidebar.markdown("<br>", unsafe_allow_html=True) # Espaciador suave
    if st.sidebar.button("🧹 Purgar Memoria y Caché", use_container_width=True, key="btn_purga_telemetria_aleph"):
        st.session_state.clear()
        st.cache_data.clear()
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
        st.page_link("pages/09_🧠_Toma_de_Decisiones.py", label="Toma de Decisiones", icon="🧠") 
        st.page_link("pages/10_👑_Panel_Administracion.py", label="Panel Administración", icon="⚙️")
        st.page_link("pages/11_⚙️_Generador.py", label="Generador", icon="✨")
        st.page_link("pages/12_📚_Ayuda_y_Docs.py", label="Ayuda y Docs", icon="📚")
        st.page_link("pages/13_🕵️_Detective.py", label="Detective", icon="🕵️")
        st.page_link("pages/14_🛰️_Satelite_En_Vivo.py", label="Satélite en Vivo", icon="🛰️")
        st.page_link("pages/15_⚖️_Escenarios_WEAP.py", label="Escenarios WEAP", icon="⚖️")
        st.page_link("pages/16_🏭_Inyeccion_RURH.py", label="Inyección RURH", icon="🏭")
    
    renderizar_telemetria_aleph()

# ====================================================================
# ☁️ CONEXIÓN A SUPABASE
# ====================================================================
@st.cache_resource
def get_supabase_client():
    try:
        from supabase import create_client
        url_sb = st.secrets.get("SUPABASE_URL") or st.secrets.get("supabase", {}).get("SUPABASE_URL") or st.secrets.get("supabase", {}).get("url")
        key_sb = st.secrets.get("SUPABASE_KEY") or st.secrets.get("supabase", {}).get("SUPABASE_KEY") or st.secrets.get("supabase", {}).get("key")
        if url_sb and key_sb: return create_client(url_sb, key_sb)
        else: return "NO_SECRETS"
    except ImportError: return "NO_LIBRARY"
    except Exception as e: return str(e)

def renderizar_gestor_escenarios(lugar):
    """
    Gestor de escenarios persistente conectado a PostgreSQL (Supabase).
    Guarda la configuración actual del usuario en la BD para restaurarla en cualquier momento.
    """
    st.sidebar.markdown("---")
    with st.sidebar.expander("📸 Gestor de Escenarios", expanded=False):
        st.markdown(f"**Territorio:** `{lugar}`")
        
        # 1. CONEXIÓN Y VERIFICACIÓN DE LA TABLA EN BASE DE DATOS
        engine = db_manager.get_engine()
        try:
            with engine.begin() as conn:
                # El sistema crea automáticamente la tabla la primera vez que se ejecuta
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS escenarios_sihcli (
                        id SERIAL PRIMARY KEY,
                        territorio TEXT NOT NULL,
                        nombre_escenario TEXT NOT NULL,
                        fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        datos JSONB,
                        UNIQUE (territorio, nombre_escenario)
                    )
                """))
        except Exception as e:
            st.error(f"⚠️ Error verificando BD de escenarios: {e}")
            return

        # 2. SECCIÓN DE GUARDADO
        nombre_nuevo = st.text_input("Nombre del Nuevo Escenario:", placeholder="Ej: Plan Resiliencia 2027")
        
        if st.button("💾 Guardar Escenario Actual", use_container_width=True):
            if nombre_nuevo.strip():
                # 🚀 FILTRO PREVENTIVO: Empaquetamos todo el estado excluyendo botones y envíos
                estado_actual = {
                    k: v for k, v in st.session_state.items() 
                    if isinstance(v, (int, float, str, bool, list, dict)) 
                    and not k.startswith("FormSubmitter")
                    and not k.startswith("btn_")
                }
                
                try:
                    # Usamos UPSERT: Si el nombre ya existe para este territorio, lo sobrescribe
                    query_insert = text("""
                        INSERT INTO escenarios_sihcli (territorio, nombre_escenario, datos, fecha) 
                        VALUES (:terr, :nom, :datos, CURRENT_TIMESTAMP)
                        ON CONFLICT (territorio, nombre_escenario) 
                        DO UPDATE SET datos = EXCLUDED.datos, fecha = CURRENT_TIMESTAMP
                    """)
                    with engine.begin() as conn:
                        conn.execute(query_insert, {
                            "terr": lugar, 
                            "nom": nombre_nuevo.strip(), 
                            "datos": json.dumps(estado_actual)
                        })
                    st.success(f"✅ Escenario '{nombre_nuevo}' blindado en Supabase.")
                except Exception as e:
                    st.error(f"❌ Error al guardar en la nube: {e}")
            else:
                st.warning("⚠️ Debes asignar un nombre al escenario.")

        st.markdown("---")
        
        # 3. SECCIÓN DE RESTAURACIÓN
        try:
            # Consultamos los escenarios guardados para el territorio actual, del más reciente al más antiguo
            query_load = text("SELECT nombre_escenario FROM escenarios_sihcli WHERE territorio = :terr ORDER BY fecha DESC")
            df_escenarios = pd.read_sql(query_load, engine, params={"terr": lugar})
            opciones_guardadas = df_escenarios['nombre_escenario'].tolist() if not df_escenarios.empty else []
        except Exception:
            opciones_guardadas = []

        if opciones_guardadas:
            esc_sel = st.selectbox("📂 Escenarios en la Nube:", opciones_guardadas)
            
            # Botones de acción alineados
            col1, col2 = st.columns([2, 1])
            with col1:
                if st.button("🚀 Cargar", use_container_width=True):
                    try:
                        q_fetch = text("SELECT datos FROM escenarios_sihcli WHERE territorio = :terr AND nombre_escenario = :nom")
                        with engine.connect() as conn:
                            resultado = conn.execute(q_fetch, {"terr": lugar, "nom": esc_sel}).fetchone()
                            if resultado and resultado[0]:
                                datos_recuperados = resultado[0] if isinstance(resultado[0], dict) else json.loads(resultado[0])
                                
                                # 🚀 FILTRO CURATIVO: Inyectamos la memoria guardada saltando variables protegidas
                                for k, v in datos_recuperados.items():
                                    if not k.startswith("btn_") and not k.startswith("FormSubmitter"):
                                        try:
                                            st.session_state[k] = v
                                        except Exception:
                                            pass
                                
                                st.success(f"Restaurando '{esc_sel}'...")
                                time.sleep(0.5)
                                st.rerun() # Forzamos la recarga para que toda la plataforma reaccione
                    except Exception as e:
                        st.error(f"Error restaurando: {e}")
                        
            with col2:
                if st.button("🗑️", help="Eliminar este escenario"):
                    try:
                        q_del = text("DELETE FROM escenarios_sihcli WHERE territorio = :terr AND nombre_escenario = :nom")
                        with engine.begin() as conn:
                            conn.execute(q_del, {"terr": lugar, "nom": esc_sel})
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.info("Sin escenarios guardados.")

# ====================================================================
# 🛡️ DESCARGA MAESTRA DE CSVs DE SUPABASE
# ====================================================================
@st.cache_data(show_spinner=False, ttl=86400)
def obtener_matriz_maestra_csv(url):
    try:
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip().str.upper()
        return df
    except: return pd.DataFrame()

# ====================================================================
# 🟩 FUNCIÓN DE INTERFAZ: CAJA VERDE DE SÍNTESIS (BODY)
# ====================================================================
def render_cabezote_sintesis_body(nombre_zona):
    """Renderiza la síntesis activa en el centro de la página."""
    pob = st.session_state.get('aleph_pob_total', 0)
    bov = st.session_state.get('ica_bovinos_calc_met', 0)
    por = st.session_state.get('ica_porcinos_calc_met', 0)
    ave = st.session_state.get('ica_aves_calc_met', 0)
    
    if nombre_zona and nombre_zona not in ["Sin Selección", "-- Seleccione --", "NINGUNO", ""]:
        if pob > 0 or bov > 0 or por > 0 or ave > 0:
            st.success(f"📌 **SÍNTESIS ACTIVA** | 📍 Territorio: {nombre_zona} \n\n 👥 Humanos: {pob:,} | 🐄 Bov: {bov:,} | 🐖 Porc: {por:,} | 🐔 Aves: {ave:,}")
        # 🔥 Extirpamos el 'else' con el st.warning. 
        # Si las variables inician en 0 milisegundos antes de calcular, simplemente espera en silencio.

# ====================================================================
# 🌍 SELECTOR ESPACIAL MAESTRO (CARGA DIFERIDA - LAZY LOADING)
# ====================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def cargar_atributos_cuencas():
    """Descarga SOLO los textos para armar los filtros. 0% Geometría. 0% Timeouts."""
    engine = db_manager.get_engine()
    # Usamos pandas puro y evitamos la pesada columna 'geometry'
    query = "SELECT ah, nomah, zh, nomzh, szh, nom_szh, nss1, nom_nss1, nss2, nom_nss2, nss3, nom_nss3, corpoamb FROM cuencas"
    return pd.read_sql(query, engine)

@st.cache_data(ttl=3600, show_spinner=False)
def cargar_atributos_municipios():
    """Descarga los atributos básicos de municipios."""
    engine = db_manager.get_engine()
    try:
        return pd.read_sql("SELECT mpio_ccdgo, mpio_cdpmp, mpio_cnmbr, dane FROM municipios", engine)
    except:
        return pd.read_sql("SELECT * FROM municipios", engine)

def render_selector_espacial():
    ids_estaciones = []
    nombre_zona = "Antioquia"
    altitud_ref = 1500.0
    gdf_zona = None
    nivel_jerarquico = "DEPARTAMENTO" 
    
    try: engine = db_manager.get_engine()
    except Exception as e:
        st.error(f"Error conectando a BD: {e}")
        return ids_estaciones, nombre_zona, altitud_ref, gdf_zona
    
    with st.sidebar.expander("📍 Filtros Geográficos Principales", expanded=True):
        modo = st.radio("Nivel de Agregación:", ["Por Cuenca", "Por Municipio", "Por Región", "Departamento"], index=0)
        
        # --- A. POR CUENCA ---
        if modo == "Por Cuenca":
            df_c = cargar_atributos_cuencas() # 🚀 Usamos el DataFrame ligero
            ruta = st.selectbox("Ruta de Búsqueda:", ["Hidrología", "CAR"], index=0)
            
            if ruta == "Hidrología":
                nombres_niveles = {
                    "AH": "🌊 AH - Área Hidrográfica (Macrorregión Nacional)",
                    "ZH": "💧 ZH - Zona Hidrográfica (Cuenca Mayor)",
                    "SZH": "🌿 SZH - Subzona Hidr. (Río Principal ej. Porce)",
                    "NSS1": "🍃 NSS1 - Cuenca Tributaria de Ríos importantes a escala regional)",
                    "NSS2": "🌱 NSS2 - Microcuenca Local, Ríos y quebradas de importancia municipal",
                    "NSS3": "💧 NSS3 - Cuencas de quebradas y arroyos de menor orden y de importancia veredal"
                }
                nivel_display = st.selectbox("1. Escala a Evaluar:", list(nombres_niveles.values()), index=5)
                nivel = next(key for key, value in nombres_niveles.items() if value == nivel_display)
            
                col_obj_esperada = {"AH": "nomah", "ZH": "nomzh", "SZH": "nom_szh", "NSS1": "nom_nss1", "NSS2": "nom_nss2", "NSS3": "nom_nss3"}[nivel]
                col_cod_esperada = {"AH": "ah", "ZH": "zh", "SZH": "szh", "NSS1": "nss1", "NSS2": "nss2", "NSS3": "nss3"}[nivel]
                
                col_obj = next((c for c in df_c.columns if c.lower() == col_obj_esperada.lower()), col_obj_esperada)
                col_cod = next((c for c in df_c.columns if c.lower() == col_cod_esperada.lower()), col_cod_esperada)
                
                st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
                st.markdown("<span style='font-size:0.85em; color:gray;'>Filtros Opcionales de Búsqueda:</span>", unsafe_allow_html=True)
                
                df_f = df_c
                col_f_ah = next((c for c in df_f.columns if c.lower() == 'nomah'), 'nomah')
                col_f_zh = next((c for c in df_f.columns if c.lower() == 'nomzh'), 'nomzh')
                col_f_szh = next((c for c in df_f.columns if c.lower() == 'nom_szh'), 'nom_szh')
                
                if nivel in ["ZH", "SZH", "NSS1", "NSS2", "NSS3"]:
                    if col_f_ah in df_f.columns:
                        ah = st.selectbox("Filtro AH:", ["-- TODAS --"] + sorted(df_f[col_f_ah].dropna().unique()))
                        if ah != "-- TODAS --": df_f = df_f[df_f[col_f_ah]==ah]
                    
                if nivel in ["SZH", "NSS1", "NSS2", "NSS3"]:
                    if col_f_zh in df_f.columns:
                        zh = st.selectbox("Filtro ZH:", ["-- TODAS --"] + sorted(df_f[col_f_zh].dropna().unique()))
                        if zh != "-- TODAS --": df_f = df_f[df_f[col_f_zh]==zh]
                    
                if nivel in ["NSS1", "NSS2", "NSS3"]:
                    if col_f_szh in df_f.columns:
                        szh = st.selectbox("Filtro SZH:", ["-- TODAS --"] + sorted(df_f[col_f_szh].dropna().unique()))
                        if szh != "-- TODAS --": df_f = df_f[df_f[col_f_szh]==szh]
                    
                st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
                
                df_f = df_f.copy()
                if col_cod in df_f.columns and col_obj in df_f.columns:
                    df_f['codigo_limpio'] = df_f[col_cod].fillna('Sin Código')
                    df_f['Llave_Visual'] = df_f[col_obj].astype(str) + " - (" + df_f['codigo_limpio'].astype(str) + ")"
                elif col_obj in df_f.columns:
                    df_f['Llave_Visual'] = df_f[col_obj].astype(str)
                else:
                    df_f['Llave_Visual'] = "Desconocido"
                
                sel_fin = st.selectbox(f"🎯 2. Territorio Exacto ({nivel}):", ["-- Seleccione --"] + sorted(df_f['Llave_Visual'].dropna().unique()))
                
                if sel_fin != "-- Seleccione --":
                    nombre_zona = sel_fin
                    nivel_jerarquico = nivel
                    
                    # 🚀 CARGA DIFERIDA: Descargamos la geometría ÚNICAMENTE de la cuenca seleccionada
                    cod_val = df_f[df_f['Llave_Visual']==sel_fin][col_cod].iloc[0]
                    from sqlalchemy import text
                    q_geom = text(f"SELECT *, geometry FROM cuencas WHERE {col_cod} = :cod LIMIT 1")
                    try:
                        gdf_zona = gpd.read_postgis(q_geom, engine, params={"cod": str(cod_val)}, geom_col="geometry")
                    except: gdf_zona = None
                else:
                    nombre_zona, gdf_zona = "-- Seleccione --", None
                    nivel_jerarquico = "NINGUNO"
            
            elif ruta == "CAR":
                try:
                    q_cars = text("SELECT DISTINCT territorio FROM matriz_maestra_hidrologia WHERE UPPER(nivel) = 'CAR' ORDER BY territorio")
                    df_cars = pd.read_sql(q_cars, db_manager.get_engine())
                    opciones_car = df_cars['territorio'].tolist() if not df_cars.empty else ["AMVA", "CORANTIOQUIA", "CORNARE", "CORPOURABA"]
                except: opciones_car = ["AMVA", "CORANTIOQUIA", "CORNARE", "CORPOURABA"]

                car_sel = st.selectbox("Autoridad Ambiental (CAR):", ["-- Seleccione --"] + sorted(opciones_car))
                
                if car_sel != "-- Seleccione --":
                    col_car = next((c for c in df_c.columns if c.lower() == 'corpoamb'), 'corpoamb')
                    if col_car in df_c.columns:
                        if car_sel == "AMVA": mask_car = df_c[col_car].str.contains('AMVA|ABURR|METROPOLITANA', case=False, na=False)
                        else: mask_car = df_c[col_car].str.contains(car_sel[:4], case=False, na=False)
                        df_f = df_c[mask_car]
                    else:
                        df_f = df_c
                        
                    if df_f.empty: df_f = df_c
                    
                    nivel = st.selectbox("1. Resolución a Evaluar:", ["CAR", "NSS1", "NSS2", "NSS3"], index=0)
                    
                    if nivel == "CAR":
                        nombre_zona = car_sel
                        nivel_jerarquico = "CAR"
                        # 🚀 CARGA DIFERIDA: Geometrías de toda la CAR
                        q_geom = text(f"SELECT *, geometry FROM cuencas WHERE {col_car} ILIKE :car_n")
                        gdf_zona = gpd.read_postgis(q_geom, engine, params={"car_n": f"%{car_sel[:4]}%"}, geom_col="geometry")
                    else:
                        col_obj_esperada = {"NSS1": "nom_nss1", "NSS2": "nom_nss2", "NSS3": "nom_nss3"}[nivel]
                        col_cod_esperada = {"NSS1": "nss1", "NSS2": "nss2", "NSS3": "nss3"}[nivel]
                        
                        col_obj = next((c for c in df_f.columns if c.lower() == col_obj_esperada.lower()), col_obj_esperada)
                        col_cod = next((c for c in df_f.columns if c.lower() == col_cod_esperada.lower()), col_cod_esperada)
                        
                        df_f = df_f.copy()
                        if col_cod in df_f.columns and col_obj in df_f.columns:
                            df_f['codigo_limpio'] = df_f[col_cod].fillna('Sin Código')
                            df_f['Llave_Visual'] = df_f[col_obj].astype(str) + " - (" + df_f['codigo_limpio'].astype(str) + ")"
                        elif col_obj in df_f.columns:
                            df_f['Llave_Visual'] = df_f[col_obj].astype(str)
                        else:
                            df_f['Llave_Visual'] = "Desconocido"
                            
                        sel_fin = st.selectbox(f"🎯 2. Territorio Exacto ({nivel}):", ["-- Seleccione --"] + sorted(df_f['Llave_Visual'].dropna().unique()))
                        
                        if sel_fin != "-- Seleccione --":
                            nombre_zona = sel_fin
                            nivel_jerarquico = nivel 
                            # 🚀 CARGA DIFERIDA
                            cod_val = df_f[df_f['Llave_Visual']==sel_fin][col_cod].iloc[0]
                            q_geom = text(f"SELECT *, geometry FROM cuencas WHERE {col_cod} = :cod LIMIT 1")
                            try: gdf_zona = gpd.read_postgis(q_geom, engine, params={"cod": str(cod_val)}, geom_col="geometry")
                            except: gdf_zona = None
                        else:
                            nombre_zona, gdf_zona = "-- Seleccione --", None
                            nivel_jerarquico = "NINGUNO"
                else:
                    nombre_zona, gdf_zona = "-- Seleccione --", None
                    nivel_jerarquico = "NINGUNO"
                    
        # --- B. POR REGIÓN (ROBUSTO Y FUSIONADO) ---
        elif modo == "Por Región":
            try:
                df_m = pd.read_excel("https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/territorio_maestro.xlsx", engine='openpyxl')
                df_m.columns = [str(c).lower().strip() for c in df_m.columns]
                
                col_reg = 'subregion' if 'subregion' in df_m.columns else next((c for c in df_m.columns if c in ['region', 'provincia']), None)
                col_dane_ex = next((c for c in df_m.columns if c in ['dp_mp', 'cod_dane', 'dane', 'codigo_mpio']), None)
                
                if col_reg and col_dane_ex:
                    lista_reg = sorted([str(r).strip().title() for r in df_m[col_reg].dropna().unique() if str(r).strip() != ''])
                    sel_reg = st.selectbox("📍 Región:", ["-- Seleccione --"] + lista_reg)
                    
                    if sel_reg != "-- Seleccione --":
                        nombre_zona = sel_reg 
                        nivel_jerarquico = "Regional" 
                        
                        cods_crudos = df_m[df_m[col_reg].astype(str).str.strip().str.lower() == sel_reg.lower()][col_dane_ex]
                        cods = pd.to_numeric(cods_crudos, errors='coerce').dropna().astype(int).astype(str).str.zfill(5).tolist()
                        
                        # 🚀 CARGA DIFERIDA: Obtenemos solo los polígonos de los municipios de la región
                        cods_tuple = tuple(cods)
                        from sqlalchemy import text
                        q_reg = text("SELECT *, geometry FROM municipios WHERE CAST(mpio_cdpmp AS TEXT) IN :cods OR CAST(dane AS TEXT) IN :cods OR CAST(mpio_ccdgo AS TEXT) IN :cods")
                        gdf_zona_filtrada = gpd.read_postgis(q_reg, engine, params={"cods": cods_tuple}, geom_col="geometry")
                            
                        if not gdf_zona_filtrada.empty:
                            poly_region = gdf_zona_filtrada.unary_union
                            gdf_zona = gpd.GeoDataFrame({'nombre': [nombre_zona]}, geometry=[poly_region], crs=gdf_zona_filtrada.crs)
                        else:
                            st.warning(f"⚠️ No se encontraron cruces. Excel buscó: {cods[:5]}...")
                            nombre_zona, gdf_zona = "-- Seleccione --", None
                    else:
                        nombre_zona, gdf_zona = "-- Seleccione --", None
                        nivel_jerarquico = "NINGUNO"
                else:
                    st.error("⚠️ El archivo Excel no tiene las columnas 'subregion' o 'dp_mp' necesarias.")
                    nombre_zona, gdf_zona = "-- Seleccione --", None
                    nivel_jerarquico = "NINGUNO"
            except Exception as e: 
                st.error(f"🚨 Error conectando con el Maestro de Regiones: {e}")
                nombre_zona, gdf_zona = "-- Seleccione --", None
                nivel_jerarquico = "NINGUNO"

        # --- C. POR MUNICIPIO ---
        elif modo == "Por Municipio":
            df_mun = cargar_atributos_municipios()
            try:
                col_nombre = 'mpio_cnmbr' if 'mpio_cnmbr' in df_mun.columns else 'MPIO_CNMBR'
                df_mun['display'] = df_mun[col_nombre].apply(decodificar_tildes).str.title()
            except: df_mun['display'] = df_mun[col_nombre].str.title()
            
            sel_mpio = st.selectbox("🏢 Municipio:", ["-- Seleccione --"] + sorted(df_mun['display'].dropna().unique()))
            if sel_mpio != "-- Seleccione --":
                nombre_zona = sel_mpio
                nivel_jerarquico = "Municipal" 
                # 🚀 CARGA DIFERIDA
                q_mun = text(f"SELECT *, geometry FROM municipios WHERE {col_nombre} ILIKE :mpio LIMIT 1")
                try: gdf_zona = gpd.read_postgis(q_mun, engine, params={"mpio": sel_mpio}, geom_col="geometry")
                except: gdf_zona = None
            else:
                nombre_zona, gdf_zona = "-- Seleccione --", None
                nivel_jerarquico = "NINGUNO"

        # --- D. DEPARTAMENTO ---
        else:
            nombre_zona = "Antioquia"
            nivel_jerarquico = "Departamental" 
            # 🚀 CARGA DIFERIDA DEPARTAMENTAL
            q_dep = text("SELECT *, geometry FROM municipios")
            gdf_muns = gpd.read_postgis(q_dep, engine, geom_col="geometry")
            gdf_zona = gpd.GeoDataFrame({'nombre':['Antioquia']}, geometry=[gdf_muns.unary_union], crs=gdf_muns.crs)

    # ====================================================================
    # 🧠 ORQUESTADOR SILENCIOSO (FUSIONADO CON CSV SUPABASE)
    # ====================================================================
    zonas_ignoradas = ["Bloque Regional", "-- TODAS --", "-- Seleccione --", "", "NINGUNO", "Sin Selección"]
    zona_activa = st.session_state.get('zona_activa_global')
    
    if nombre_zona not in zonas_ignoradas and nombre_zona != zona_activa:
        st.session_state['zona_activa_global'] = nombre_zona
        st.session_state['nivel_activo_global'] = nivel_jerarquico 
        st.session_state['aleph_lugar'] = nombre_zona
        
        # Limpieza de estados para nueva carga
        for k in ['aleph_pob_total', 'aleph_pob_urbana', 'aleph_pob_rural', 'ica_bovinos_calc_met', 'ica_porcinos_calc_met', 'ica_aves_calc_met', 'aleph_oferta_m3s', 'aleph_lluvia_mm', 'aleph_area_km2', 'aleph_recarga_mm']: 
            st.session_state.pop(k, None)
            
        # 1. 🔍 BÚSQUEDA ROBUSTA EN CSV SUPABASE (Evita errores de BD faltantes)
        url_demo = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/Matriz_Maestra_Demografica.csv"
        url_pecu = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/Matriz_Maestra_Pecuaria.csv"
        
        # Normalizamos la búsqueda (quitando tildes pero conservando guiones)
        term_search = unicodedata.normalize('NFKD', str(nombre_zona)).encode('ASCII', 'ignore').decode('utf-8').strip().upper()
        # Cortamos el "- NSS" para la búsqueda difusa porque a veces los CSV de Supabase no lo traen
        term_search_short = term_search.split(" - ")[0].strip() if " - " in term_search else term_search

        # Carga Demográfica
        df_demo = obtener_matriz_maestra_csv(url_demo)
        if not df_demo.empty:
            # Buscamos en todo el dataframe convertido a texto
            mask = df_demo.astype(str).apply(lambda col: col.str.contains(term_search_short, regex=False)).any(axis=1)
            if mask.any():
                fila = df_demo[mask].iloc[0]
                st.session_state['aleph_pob_total'] = int(fila.get('POB_TOTAL', fila.get('POBLACION', 0)))
                st.session_state['aleph_pob_urbana'] = int(fila.get('POB_URBANA', fila.get('URBANA', 0)))
                st.session_state['aleph_pob_rural'] = int(fila.get('POB_RURAL', fila.get('RURAL', 0)))

        # Carga Pecuaria
        df_pecu = obtener_matriz_maestra_csv(url_pecu)
        if not df_pecu.empty:
            mask_p = df_pecu.astype(str).apply(lambda col: col.str.contains(term_search_short, regex=False)).any(axis=1)
            if mask_p.any():
                filtro_p = df_pecu[mask_p]
                if 'ESPECIE' in filtro_p.columns:
                    for _, row in filtro_p.iterrows():
                        esp = str(row['ESPECIE']).upper()
                        cab = int(row.get('CABEZAS_ACTUALES', row.get('CANTIDAD', row.get('TOTAL', 0))))
                        if 'BOVINO' in esp: st.session_state['ica_bovinos_calc_met'] = cab
                        elif 'PORCINO' in esp: st.session_state['ica_porcinos_calc_met'] = cab
                        elif 'AVE' in esp: st.session_state['ica_aves_calc_met'] = cab
                else:
                    fila_p = filtro_p.iloc[0]
                    st.session_state['ica_bovinos_calc_met'] = int(fila_p.get('BOVINOS', 0))
                    st.session_state['ica_porcinos_calc_met'] = int(fila_p.get('PORCINOS', 0))
                    st.session_state['ica_aves_calc_met'] = int(fila_p.get('AVES', 0))
            
        # 2. 🔍 HIDROLOGÍA (Se mantiene tu SQL original)
        try:
            q_hidro = text("SELECT * FROM matriz_hidrologica_maestra WHERE \"Jerarquia\" = :nivel AND \"Territorio\" = :zona LIMIT 1")
            df_hidro = pd.read_sql(q_hidro, engine, params={"nivel": nivel_jerarquico, "zona": nombre_zona})
            if not df_hidro.empty:
                row_h = df_hidro.iloc[0]
                st.session_state['aleph_oferta_m3s'] = float(row_h['Caudal_Medio_m3s'])
                st.session_state['aleph_lluvia_mm'] = float(row_h['Lluvia_mm'])
                st.session_state['aleph_area_km2'] = float(row_h['Area_km2'])
                st.session_state['aleph_recarga_mm'] = float(row_h['Recarga_mm'])
                st.session_state['aleph_altitud_m'] = float(row_h['Altitud_m'])
        except Exception: pass

    renderizar_gestor_escenarios(nombre_zona)
    return ids_estaciones, nombre_zona, altitud_ref, gdf_zona
