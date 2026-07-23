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

        # 🚀 FIX: Sensor RURH Multi-Variable (Caudal y Puntos)
        rurh_m3s = st.session_state.get('aleph_concesiones_m3s', 0.0)
        rurh_lps = st.session_state.get('aleph_rurh_caudal_lps', 0.0)
        rurh_puntos = st.session_state.get('aleph_rurh_puntos', 0)
        
        # Unificamos el caudal a m³/s para la visualización
        caudal_total_m3s = rurh_m3s if rurh_m3s > 0 else (rurh_lps / 1000.0)
        
        if caudal_total_m3s > 0 or rurh_puntos > 0:
            puntos_str = f" | {rurh_puntos} pts" if rurh_puntos > 0 else ""
            st.markdown(f"🏭 **RURH (Extracción):** <span style='color:#e67e22'>Activa ({caudal_total_m3s:,.3f} m³/s{puntos_str})</span>", unsafe_allow_html=True)
        else:
            st.markdown("🏭 **RURH (Extracción):** <span style='color:#95a5a6'>Inactiva</span>", unsafe_allow_html=True)
            
        # Sensor Calidad (Tormenta Eliminada)
        dbo_viva = st.session_state.get('carga_dbo_total_ton', 0.0)
        if dbo_viva > 0:
            st.markdown(f"☣️ **Carga DBO:** <span style='color:#8e44ad'>Registrada ({dbo_viva:,.0f} Ton)</span>", unsafe_allow_html=True)
        else:
            st.markdown("☣️ **Carga DBO:** <span style='color:#95a5a6'>Inactiva</span>", unsafe_allow_html=True)
            
    # --- CAJA DESPLEGABLE 2: PULSO CLIMÁTICO GLOBAL (ENSO E ÍNDICES) ---
    with st.sidebar.expander("🌦️ Pulso Climático Global", expanded=True):
        if 'aleph_iri_nino' not in st.session_state:
            try:
                from modules.climate_api import get_iri_enso_forecast, get_live_oni_data, get_live_soi_data, get_live_iod_data
                
                # 1. Pronóstico IRI ENSO
                df_enso, _ = get_iri_enso_forecast()
                if df_enso is not None and not df_enso.empty:
                    # 🚀 FIX: Toma dinámicamente el primer trimestre vigente (ej. JJA)
                    fila_actual = df_enso.iloc[0]
                    
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
                    st.session_state['aleph_iri_tendencia'] = f"Sincronización {trim_txt} Exitosa 📡"
                
                # 2. Telemetría de Índices Climáticos Adicionales
                df_oni = get_live_oni_data()
                df_soi = get_live_soi_data()
                df_iod = get_live_iod_data()
                
                if df_oni is not None and not df_oni.empty:
                    st.session_state['aleph_oni_val'] = df_oni['anomalia_oni'].iloc[-1]
                if df_soi is not None and not df_soi.empty:
                    st.session_state['aleph_soi_val'] = df_soi['soi'].iloc[-1]
                if df_iod is not None and not df_iod.empty:
                    st.session_state['aleph_iod_val'] = df_iod['iod'].iloc[-1]
                    
            except Exception as e:
                st.session_state['aleph_iri_tendencia'] = f"Error: {str(e)}"
                
        # --- RENDERIZADO VISUAL DEL ESTADO GLOBAL ---
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
            
        # --- RENDERIZADO DE ÍNDICES ADICIONALES ---
        st.markdown("---")
        st.markdown("📊 **Índices Climáticos Actuales**")
        col_oni, col_soi, col_iod = st.columns(3)
        
        oni_val = st.session_state.get('aleph_oni_val', 'N/A')
        soi_val = st.session_state.get('aleph_soi_val', 'N/A')
        iod_val = st.session_state.get('aleph_iod_val', 'N/A')
        
        col_oni.metric(
            "ONI", 
            f"{oni_val:.2f}" if isinstance(oni_val, float) else oni_val,
            help="**Oceanic Niño Index (ONI):** Mide la anomalía de la temperatura superficial del mar en la región Niño 3.4 del Pacífico.\n\n🔴 **≥ +0.5°C:** Fase cálida (El Niño).\n🔵 **≤ -0.5°C:** Fase fría (La Niña).\n⚪ **Entre -0.5 y 0.5:** Neutro."
        )
        col_soi.metric(
            "SOI", 
            f"{soi_val:.2f}" if isinstance(soi_val, float) else soi_val,
            help="**Southern Oscillation Index (SOI):** Mide la diferencia de presión atmosférica entre Tahití y Darwin.\n\n🔴 **Valores negativos sostenidos:** Indican El Niño (aguas más cálidas).\n🔵 **Valores positivos sostenidos:** Indican La Niña."
        )
        col_iod.metric(
            "IOD", 
            f"{iod_val:.2f}" if isinstance(iod_val, float) else iod_val,
            help="**Indian Ocean Dipole (IOD):** Diferencia de temperatura superficial entre el este y el oeste del Océano Índico.\n\n🔴 **Fase Positiva:** Aguas más cálidas en el oeste.\n🔵 **Fase Negativa:** Aguas más frías en el oeste.\n*Afecta la humedad global y modula los efectos del ENSO en Suramérica.*"
        )
        
        if tendencia: st.caption(f"📈 **Estado:** {tendencia}")

    # --- BOTÓN DE PURGA (Fuera de los expanders) ---
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
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
        st.page_link("pages/14_🌍_Satelite_Terrestre.py", label="Satélite Terrestre", icon="🌍")
        st.page_link("pages/17_🛰️_Radar_Meteorologico.py", label="Radar Meteorológico", icon="🛰️")
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
# 🌍 SELECTOR ESPACIAL MAESTRO (CARGA DIFERIDA BLINDADA)
# ====================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def cargar_atributos_cuencas():
    """Descarga SOLO los textos. 0% Geometría. 100% Velocidad."""
    import pandas as pd
    from sqlalchemy import text # 🚀 EL ESCUDO FALTANTE
    engine = db_manager.get_engine()
    query = text("SELECT ah, nomah, zh, nomzh, szh, nom_szh, nss1, nom_nss1, nss2, nom_nss2, nss3, nom_nss3, corpoamb FROM cuencas")
    try:
        with engine.connect() as conn:
            return pd.read_sql(query, conn)
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def cargar_atributos_municipios():
    import pandas as pd
    from sqlalchemy import text # 🚀 EL ESCUDO FALTANTE
    engine = db_manager.get_engine()
    try:
        with engine.connect() as conn:
            return pd.read_sql(text("SELECT mpio_ccdgo, mpio_cdpmp, mpio_cnmbr, dane FROM municipios"), conn)
    except:
        try:
            with engine.connect() as conn:
                return pd.read_sql(text("SELECT * FROM municipios"), conn)
        except:
            return pd.DataFrame()

def render_selector_espacial(modo_firma="clasica"):
    from sqlalchemy import text 
    # 🚀 Asegúrate de que estas 3 líneas estén aquí arriba:
    import geopandas as gpd
    import pandas as pd
    from modules.utils import cargar_capa_espacial_cache
    
    ids_estaciones = []
    nombre_zona = "Antioquia"
    altitud_ref = 1500.0
    gdf_zona = None
    nivel_jerarquico = "Departamento" 
    
    engine = db_manager.get_engine()
    if engine is None:
        st.error("Error crítico: No hay conexión a la base de datos.")
        return ids_estaciones, nombre_zona, altitud_ref, gdf_zona
    
    with st.sidebar.expander("📍 Filtros Geográficos Principales", expanded=True):
        modo = st.radio("Nivel de Agregación:", ["Por Cuenca", "Por Municipio", "Por Región", "Departamento"], index=0)
        
        # --- A. POR CUENCA ---
        if modo == "Por Cuenca":
            df_c = cargar_atributos_cuencas()
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
                    # 🚀 FIX: Eliminamos los ".0" de los códigos asegurando formato entero/texto puro
                    def limpiar_codigo(x):
                        if pd.isnull(x): return 'Sin Código'
                        try: return str(int(float(x)))
                        except: return str(x).strip()
                        
                    df_f['codigo_limpio'] = df_f[col_cod].apply(limpiar_codigo)
                    df_f['Llave_Visual'] = df_f[col_obj].astype(str).str.strip() + " - (" + df_f['codigo_limpio'] + ")"
                elif col_obj in df_f.columns:
                    df_f['Llave_Visual'] = df_f[col_obj].astype(str).str.strip()
                else:
                    df_f['Llave_Visual'] = "Desconocido"
                
                sel_fin = st.selectbox(f"🎯 2. Territorio Exacto ({nivel}):", ["-- Seleccione --"] + sorted(df_f['Llave_Visual'].dropna().unique()))
                
                if sel_fin != "-- Seleccione --":
                    nombre_zona = sel_fin
                    nivel_jerarquico = nivel
                    
                    df_filtrado = df_f[df_f['Llave_Visual'] == sel_fin]
                    if not df_filtrado.empty:
                        cod_val = str(df_filtrado[col_cod].iloc[0])
                        # 🚀 FIX: Sin LIMIT 1 y usando parámetros
                        q_geom = text(f"SELECT * FROM cuencas WHERE {col_cod} = :cod")
                        with engine.connect() as conn:
                            gdf_zona = gpd.read_postgis(q_geom, conn, params={"cod": cod_val}, geom_col="geometry")
                        
                        # 🚀 FIX: Disolver geometrías hijas si existen múltiples
                        if not gdf_zona.empty and len(gdf_zona) > 1:
                            gdf_zona = gdf_zona.dissolve(by=col_cod).reset_index()
                    else:
                        nombre_zona, gdf_zona = "-- Seleccione --", None
                        nivel_jerarquico = "NINGUNO"
                else:
                    nombre_zona, gdf_zona = "-- Seleccione --", None
                    nivel_jerarquico = "NINGUNO"
            
            elif ruta == "CAR":
                try:
                    q_cars = text("SELECT DISTINCT territorio FROM matriz_maestra_hidrologia WHERE UPPER(nivel) = 'CAR' ORDER BY territorio")
                    with engine.connect() as conn:
                        df_cars = pd.read_sql(q_cars, conn)
                    opciones_car = df_cars['territorio'].tolist() if not df_cars.empty else ["AMVA", "CORANTIOQUIA", "CORNARE", "CORPOURABA"]
                except Exception as e:
                    st.warning("No se pudo conectar a la base de datos de hidrología. Usando opciones CAR por defecto.")
                    opciones_car = ["AMVA", "CORANTIOQUIA", "CORNARE", "CORPOURABA"]

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
                        # 🚀 FIX: Parámetros seguros
                        q_geom = text(f"SELECT * FROM cuencas WHERE {col_car} ILIKE :car_val")
                        with engine.connect() as conn:
                            gdf_zona = gpd.read_postgis(q_geom, conn, params={"car_val": f"%{car_sel[:4]}%"}, geom_col="geometry")
                    else:
                        col_obj_esperada = {"NSS1": "nom_nss1", "NSS2": "nom_nss2", "NSS3": "nom_nss3"}[nivel]
                        col_cod_esperada = {"NSS1": "nss1", "NSS2": "nss2", "NSS3": "nss3"}[nivel]
                        
                        col_obj = next((c for c in df_f.columns if c.lower() == col_obj_esperada.lower()), col_obj_esperada)
                        col_cod = next((c for c in df_f.columns if c.lower() == col_cod_esperada.lower()), col_cod_esperada)
                        
                        df_f = df_f.copy()
                        if col_cod in df_f.columns and col_obj in df_f.columns:
                            def limpiar_codigo(x):
                                if pd.isnull(x): return 'Sin Código'
                                try: return str(int(float(x)))
                                except: return str(x).strip()
                            
                            df_f['codigo_limpio'] = df_f[col_cod].apply(limpiar_codigo)
                            df_f['Llave_Visual'] = df_f[col_obj].astype(str).str.strip() + " - (" + df_f['codigo_limpio'] + ")"
                        elif col_obj in df_f.columns:
                            df_f['Llave_Visual'] = df_f[col_obj].astype(str).str.strip()
                        else:
                            df_f['Llave_Visual'] = "Desconocido"
                            
                        sel_fin = st.selectbox(f"🎯 2. Territorio Exacto ({nivel}):", ["-- Seleccione --"] + sorted(df_f['Llave_Visual'].dropna().unique()))
                        
                        if sel_fin != "-- Seleccione --":
                            nombre_zona = sel_fin
                            nivel_jerarquico = nivel 
                            
                            df_filtrado = df_f[df_f['Llave_Visual'] == sel_fin]
                            if not df_filtrado.empty:
                                cod_val = str(df_filtrado[col_cod].iloc[0])
                                # 🚀 FIX: Sin LIMIT 1 y usando parámetros
                                q_geom = text(f"SELECT * FROM cuencas WHERE {col_cod} = :cod")
                                with engine.connect() as conn:
                                    gdf_zona = gpd.read_postgis(q_geom, conn, params={"cod": cod_val}, geom_col="geometry")
                                
                                # 🚀 FIX: Disolver geometrías
                                if not gdf_zona.empty and len(gdf_zona) > 1:
                                    gdf_zona = gdf_zona.dissolve(by=col_cod).reset_index()
                            else:
                                nombre_zona, gdf_zona = "-- Seleccione --", None
                                nivel_jerarquico = "NINGUNO"
                        else:
                            nombre_zona, gdf_zona = "-- Seleccione --", None
                            nivel_jerarquico = "NINGUNO"
                else:
                    nombre_zona, gdf_zona = "-- Seleccione --", None
                    nivel_jerarquico = "NINGUNO"
                    
        # --- B. POR REGIÓN ---
        elif modo == "Por Región":
            try:
                # 🚀 FIX 1: Usamos la matriz curada de memoria, no el Excel crudo de internet
                from modules.data_processor import cargar_territorio_maestro
                df_m = cargar_territorio_maestro()
                
                if not df_m.empty and 'subregion_norm' in df_m.columns:
                    # Filtramos y organizamos las subregiones normalizadas
                    lista_reg = sorted([str(r).strip().title() for r in df_m['subregion_norm'].dropna().unique() if str(r).strip() != ''])
                    sel_reg = st.selectbox("📍 Región:", ["-- Seleccione --"] + lista_reg)
                    
                    if sel_reg != "-- Seleccione --":
                        nombre_zona = sel_reg 
                        nivel_jerarquico = "Región"
                        
                        # 🚀 FIX 2: Buscamos usando el nombre normalizado (ignora mayúsculas/tildes)
                        from modules.utils import normalizar_texto_maestro
                        reg_norm = normalizar_texto_maestro(sel_reg)
                        
                        # Obtenemos los códigos DANE curados (de 5 dígitos)
                        cods = df_m[df_m['subregion_norm'] == reg_norm]['dp_mp'].tolist()
                        
                        if cods:
                            df_mun_attr = cargar_atributos_municipios()
                            cols_mun = [c.lower() for c in df_mun_attr.columns]
                            
                            cods_str = ", ".join([f"'{c}'" for c in cods])
                            condiciones = []
                            if 'mpio_cdpmp' in cols_mun: condiciones.append(f"CAST(mpio_cdpmp AS TEXT) IN ({cods_str})")
                            if 'dane' in cols_mun: condiciones.append(f"CAST(dane AS TEXT) IN ({cods_str})")
                            if 'mpio_ccdgo' in cols_mun: condiciones.append(f"CAST(mpio_ccdgo AS TEXT) IN ({cods_str})")
                            
                            if condiciones:
                                where_clause = " OR ".join(condiciones)
                                q_reg = f"SELECT * FROM municipios WHERE {where_clause}"
                                
                                with engine.connect() as conn:
                                    gdf_zona_filtrada = gpd.read_postgis(text(q_reg), conn, geom_col="geometry")
                                
                                if not gdf_zona_filtrada.empty:
                                    poly_region = gdf_zona_filtrada.unary_union
                                    gdf_zona = gpd.GeoDataFrame({'nombre': [nombre_zona]}, geometry=[poly_region], crs=gdf_zona_filtrada.crs)
                                else:
                                    st.warning(f"⚠️ No se encontraron cruces espaciales en la BD.")
                                    nombre_zona, gdf_zona = "-- Seleccione --", None
                            else:
                                st.error("⚠️ No se encontró llave de cruce en BD espacial.")
                                nombre_zona, gdf_zona = "-- Seleccione --", None
                        else:
                            st.warning("⚠️ La región no tiene municipios en el maestro.")
                            nombre_zona, gdf_zona = "-- Seleccione --", None
                    else:
                        nombre_zona, gdf_zona = "-- Seleccione --", None
                        nivel_jerarquico = "NINGUNO"
                else:
                    st.error("⚠️ Base maestra sin columna 'subregion_norm'. Ejecuta la forja.")
                    nombre_zona, gdf_zona = "-- Seleccione --", None
                    nivel_jerarquico = "NINGUNO"
            except Exception as e: 
                st.error(f"🚨 Error de Regiones: {e}")
                nombre_zona, gdf_zona = "-- Seleccione --", None
                nivel_jerarquico = "NINGUNO"

        # --- C. POR MUNICIPIO ---
        elif modo == "Por Municipio":
            df_mun = cargar_atributos_municipios()
            col_nombre = 'mpio_cnmbr' if 'mpio_cnmbr' in df_mun.columns else 'MPIO_CNMBR'
            
            # 🚀 FIX: Mapeo Inverso. Mostramos el nombre bonito en la UI, pero enviamos el texto ORIGINAL (con mayúsculas/sin tildes) al backend
            mapeo_mun = {str(orig).strip(): str(orig).title() for orig in df_mun[col_nombre].dropna().unique()}
            opciones_pretty = sorted(list(set(mapeo_mun.values())))
            
            sel_mpio_pretty = st.selectbox("🏢 Municipio:", ["-- Seleccione --"] + opciones_pretty)
            if sel_mpio_pretty != "-- Seleccione --":
                # Rescatamos la cadena de texto exacta de la base de datos
                nombre_zona_orig = next((k for k, v in mapeo_mun.items() if v == sel_mpio_pretty), sel_mpio_pretty)
                
                nombre_zona = nombre_zona_orig
                nivel_jerarquico = "Municipio" # 🚀 FIX: Restaurado de "Municipal" a "Municipio"
                
                mpio_limpio = nombre_zona_orig.replace("'", "''") 
                q_mun = f"SELECT * FROM municipios WHERE {col_nombre} = '{mpio_limpio}' LIMIT 1"
                
                with engine.connect() as conn:
                    gdf_zona = gpd.read_postgis(text(q_mun), conn, geom_col="geometry")
            else:
                nombre_zona, gdf_zona = "-- Seleccione --", None
                nivel_jerarquico = "NINGUNO"

        # --- D. DEPARTAMENTO ---
        elif modo == "Departamento":
            nombre_zona = "Antioquia"
            nivel_jerarquico = "Departamento"
            
            try:
                # 🚀 FIX QUIRÚRGICO 1: Leemos los 125 municipios desde la Memoria RAM (Instantáneo)
                gdf_muns = cargar_capa_espacial_cache("SELECT * FROM municipios")
                
                if gdf_muns is not None and not gdf_muns.empty:
                    # 🚀 FIX QUIRÚRGICO 2: Fusión (unary_union) segura y asignación de CRS
                    geometria_unida = gdf_muns.unary_union
                    gdf_zona = gpd.GeoDataFrame({'nombre': ['Antioquia']}, geometry=[geometria_unida], crs=gdf_muns.crs)
                else:
                    gdf_zona = None
                    
            except Exception as e:
                import logging
                logging.error(f"Error procesando mapa departamental: {e}")
                gdf_zona = None

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

    # ==========================================================
    # 🚀 SOLUCIÓN ESTRUCTURAL: FIRMA EXPLÍCITA
    # ==========================================================
    if modo_firma == "weap":
        return nombre_zona, gdf_zona, nivel_jerarquico, False
    else:
        # 🚀 FIX: Añadimos 'nivel_jerarquico' a la firma clásica
        return ids_estaciones, nombre_zona, altitud_ref, gdf_zona, nivel_jerarquico
