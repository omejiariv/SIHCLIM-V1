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
    
    st.sidebar.markdown("### 🧠 Telemetría del Aleph")
    
    # --- CAJA DESPLEGABLE 1: MÉTRICAS TERRITORIALES Y CARGAS ---
    with st.sidebar.expander("📊 Métricas Demográficas y Cargas", expanded=True):
        pob_viva = st.session_state.get('aleph_pob_total', st.session_state.get('poblacion_servida', 0))
        if pob_viva > 0:
            st.markdown(f"👥 **Población:** <span style='color:#2ecc71'>Sincronizada ({pob_viva:,.0f} hab)</span>", unsafe_allow_html=True)
        else:
            st.markdown("👥 **Población:** <span style='color:#e74c3c'>Vacía (Default)</span>", unsafe_allow_html=True)
            
        lodo_vivo = st.session_state.get('eco_lodo_total_m3', 0.0)
        if lodo_vivo > 0:
            st.markdown(f"⛈️ **Tormenta:** <span style='color:#e67e22'>Activa ({lodo_vivo:,.0f} m³ Lodo)</span>", unsafe_allow_html=True)
        else:
            st.markdown("⛈️ **Tormenta:** <span style='color:#95a5a6'>Inactiva</span>", unsafe_allow_html=True)
            
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
    if st.sidebar.button("🧹 Purgar Memoria y Caché", use_container_width=True):
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
                # Empaquetamos todo el estado actual excluyendo objetos no serializables de Streamlit
                estado_actual = {
                    k: v for k, v in st.session_state.items() 
                    if isinstance(v, (int, float, str, bool, list, dict)) and not k.startswith("FormSubmitter")
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
                                
                                # Inyectamos la memoria guardada de vuelta a la sesión actual
                                for k, v in datos_recuperados.items():
                                    st.session_state[k] = v
                                    
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
        else:
            st.warning(f"⚠️ **SIN DATOS MAESTROS** para: {nombre_zona}. Verifica las matrices de población y pecuaria.")

# ====================================================================
# 🌍 SELECTOR ESPACIAL MAESTRO (SQL ESTRICTO - TU CÓDIGO ORIGINAL)
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
    nivel_jerarquico = "DEPARTAMENTO" 
    
    try: engine = db_manager.get_engine()
    except Exception as e:
        st.error(f"Error conectando a BD: {e}")
        return ids_estaciones, nombre_zona, altitud_ref, gdf_zona
    
    with st.sidebar.expander("📍 Filtros Geográficos Principales", expanded=True):
        modo = st.radio("Nivel de Agregación:", ["Por Cuenca", "Por Municipio", "Por Región", "Departamento"], index=0)
        
        # --- A. POR CUENCA ---
        if modo == "Por Cuenca":
            gdf_c = cargar_mapa_cuencas()
            ruta = st.selectbox("Ruta de Búsqueda:", ["Hidrología", "CAR"], index=0)
            
            if ruta == "Hidrología":
                nivel = st.selectbox("1. Nivel a Evaluar:", ["AH", "ZH", "SZH", "NSS1", "NSS2", "NSS3"], index=5)
                
                # 🚀 BUSCADOR INTELIGENTE DE COLUMNAS (Ignora mayúsculas/minúsculas)
                col_obj_esperada = {"AH": "nomah", "ZH": "nomzh", "SZH": "nom_szh", "NSS1": "nom_nss1", "NSS2": "nom_nss2", "NSS3": "nom_nss3"}[nivel]
                col_cod_esperada = {"AH": "ah", "ZH": "zh", "SZH": "szh", "NSS1": "nss1", "NSS2": "nss2", "NSS3": "nss3"}[nivel]
                
                # Encuentra el nombre real exacto en tu base de datos
                col_obj = next((c for c in gdf_c.columns if c.lower() == col_obj_esperada.lower()), col_obj_esperada)
                col_cod = next((c for c in gdf_c.columns if c.lower() == col_cod_esperada.lower()), col_cod_esperada)
                
                st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
                st.markdown("<span style='font-size:0.85em; color:gray;'>Filtros Opcionales de Búsqueda:</span>", unsafe_allow_html=True)
                
                df_f = gdf_c
                # Extraemos los nombres de las columnas para los filtros opcionales también
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
                
                # 🚀 CREACIÓN AL VUELO BLINDADA DE LA LLAVE VISUAL
                df_f = df_f.copy()
                if col_cod in df_f.columns and col_obj in df_f.columns:
                    # Ignoramos nulos en el código para que no diga "None"
                    df_f['codigo_limpio'] = df_f[col_cod].fillna('Sin Código')
                    df_f['Llave_Visual'] = df_f[col_obj].astype(str) + " - (" + df_f['codigo_limpio'].astype(str) + ")"
                elif col_obj in df_f.columns:
                    st.warning(f"⚠️ No se encontró la columna de código '{col_cod_esperada}'. Usando solo el nombre.")
                    df_f['Llave_Visual'] = df_f[col_obj].astype(str)
                else:
                    st.error("🚨 Error crítico: No se encontró la columna de nombre.")
                    df_f['Llave_Visual'] = "Desconocido"
                
                sel_fin = st.selectbox(f"🎯 2. Territorio Exacto ({nivel}):", ["-- Seleccione --"] + sorted(df_f['Llave_Visual'].dropna().unique()))
                
                if sel_fin != "-- Seleccione --":
                    nombre_zona = sel_fin
                    gdf_zona = df_f[df_f['Llave_Visual']==sel_fin]
                    nivel_jerarquico = nivel 
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
                    # Busca columna corpoamb o CorpoAmb
                    col_car = next((c for c in gdf_c.columns if c.lower() == 'corpoamb'), 'corpoamb')
                    if col_car in gdf_c.columns:
                        if car_sel == "AMVA": mask_car = gdf_c[col_car].str.contains('AMVA|ABURR|METROPOLITANA', case=False, na=False)
                        else: mask_car = gdf_c[col_car].str.contains(car_sel[:4], case=False, na=False)
                        df_f = gdf_c[mask_car]
                    else:
                        df_f = gdf_c
                        
                    if df_f.empty: df_f = gdf_c
                    
                    nivel = st.selectbox("1. Resolución a Evaluar:", ["CAR", "NSS1", "NSS2", "NSS3"], index=0)
                    
                    if nivel == "CAR":
                        nombre_zona = car_sel
                        gdf_zona = df_f if not df_f.empty else None
                        nivel_jerarquico = "CAR"
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
                            gdf_zona = df_f[df_f['Llave_Visual']==sel_fin]
                            nivel_jerarquico = nivel 
                        else:
                            nombre_zona, gdf_zona = "-- Seleccione --", None
                            nivel_jerarquico = "NINGUNO"
                else:
                    nombre_zona, gdf_zona = "-- Seleccione --", None
                    nivel_jerarquico = "NINGUNO"
                    
        # --- B. POR REGIÓN (TU CÓDIGO MAESTRO RESTAURADO) ---
        elif modo == "Por Región":
            try:
                df_m = pd.read_excel("https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/territorio_maestro.xlsx", engine='openpyxl')
                df_m.columns = [c.lower() for c in df_m.columns]
                
                col_reg = 'subregion' if 'subregion' in df_m.columns else 'region'
                lista_reg = sorted([str(r).title() for r in df_m[col_reg].dropna().unique()])
                
                sel_reg = st.selectbox("📍 Región:", ["-- Seleccione --"] + lista_reg)
                if sel_reg != "-- Seleccione --":
                    nombre_zona = sel_reg 
                    nivel_jerarquico = "Regional" 
                    
                    cods = df_m[df_m[col_reg].str.lower()==sel_reg.lower()]['dp_mp'].astype(str).str.replace(".0", "", regex=False).str.zfill(5).tolist()
                    gdf_mun = cargar_mapa_municipios()
                    
                    col_mpio = 'mpio_ccdgo' if 'mpio_ccdgo' in gdf_mun.columns else 'MPIO_CCDGO'
                    gdf_zona = gdf_mun[gdf_mun[col_mpio].astype(str).str.replace(".0", "", regex=False).str.zfill(5).isin(cods)]
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
            try:
                col_nombre = 'mpio_cnmbr' if 'mpio_cnmbr' in gdf_mun.columns else 'MPIO_CNMBR'
                gdf_mun['display'] = gdf_mun[col_nombre].apply(decodificar_tildes).str.title()
            except: gdf_mun['display'] = gdf_mun[col_nombre].str.title()
            
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
            nivel_jerarquico = "Departamental" 

        # --- FILTRO DE ESTACIONES (CON CONEXIÓN DINÁMICA AL BUFFER) ---
        if gdf_zona is not None and not gdf_zona.empty:
            buff = st.slider("Buffer (km):", 0.0, 50.0, 25.0)
            
            # 1. Extraemos los límites geográficos del territorio seleccionado
            minx, miny, maxx, maxy = gdf_zona.to_crs(4326).total_bounds
            
            # 2. Convertimos el valor del slider (km) a su equivalente aproximado en grados decimales
            tolerancia_grados = buff * 0.009
            
            # 3. Inyectamos la tolerancia dinámica en la consulta SQL
            q = text(f"""
                SELECT id_estacion, altitud 
                FROM estaciones 
                WHERE longitud BETWEEN {minx - tolerancia_grados} AND {maxx + tolerancia_grados} 
                  AND latitud BETWEEN {miny - tolerancia_grados} AND {maxy + tolerancia_grados}
            """)
            
            try:
                df_est = pd.read_sql(q, engine)
                ids_estaciones = df_est['id_estacion'].astype(str).tolist()
                altitud_ref = df_est['altitud'].mean()
            except Exception as e:
                st.sidebar.error(f"Error consulta buffer SQL: {e}")
                ids_estaciones = []

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
