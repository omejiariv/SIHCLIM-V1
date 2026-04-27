# pages/08_🔗_Sistemas_Hidricos_Territoriales.py

import os
import sys

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import geopandas as gpd

from modules.db_manager import get_engine  # 🚀 Conexión al motor SQL
from modules.utils import normalizar_texto  

import streamlit as st

# =========================================================================
# 1. CONFIGURACIÓN Y DICCIONARIO BASE (Debe ir primero)
# =========================================================================
st.set_page_config(page_title="Metabolismo Complejo", page_icon="🔗", layout="wide")

# --- 📂 IMPORTACIÓN ROBUSTA DE MÓDULOS ---
try:
    from modules import selectors
    from modules.demografia_tools import render_motor_demografico
    from modules.utils import encender_gemelo_digital, obtener_metabolismo_exacto
except ImportError:
    # Fallback de rutas por si hay problemas de lectura entre carpetas
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules import selectors
    from modules.demografia_tools import render_motor_demografico
    from modules.utils import encender_gemelo_digital, obtener_metabolismo_exacto

# --- 🚀 FUNCIÓN DE CONEXIÓN A MATRICES MAESTRAS ---
@st.cache_data(ttl=3600)
def consultar_matriz_sql_sistemas(tabla, territorio, nivel, col_nivel="Nivel"):
    try:
        from sqlalchemy import text
        engine_sql = get_engine()
        q = text(f'''
            SELECT * FROM {tabla} 
            WHERE TRIM(UPPER("Territorio")) = UPPER(:t) 
            AND TRIM(UPPER("{col_nivel}")) = UPPER(:n)
        ''')
        df_res = pd.read_sql(q, engine_sql, params={'t': str(territorio).strip(), 'n': str(nivel).strip()})
        
        # Rescate por aproximación si falla la exacta
        if df_res.empty:
            q_fall = text(f'''
                SELECT * FROM {tabla} 
                WHERE UPPER("Territorio") LIKE UPPER(:t) 
                AND UPPER("{col_nivel}") LIKE UPPER(:n)
            ''')
            df_res = pd.read_sql(q_fall, engine_sql, params={'t': f"%{str(territorio).strip()}%", 'n': f"%{str(nivel)[:3]}%"})
            
        return df_res
    except Exception:
        return pd.DataFrame()

# ==========================================
# 📂 MENÚ DE NAVEGACIÓN
# ==========================================
selectors.renderizar_menu_navegacion("Sistemas Hídricos")

# Encendido automático del Gemelo Digital (Lectura de matrices maestras)
encender_gemelo_digital()

# Datos paramétricos (Estructura Ampliada)
sistemas_embalses = {
    "La Fe": {
        "capacidad_util_Mm3": 11.5, 
        "afluentes_naturales": {"Quebrada Espíritu Santo": 1.2},
        "demanda_acueducto_m3s": 5.0, 
        "generacion_energia_m3s": 0.0,
        "evaporacion_m3s": 0.1,
        "caudal_ecologico_m3s": 0.3,
        "factor_energia_kwh_m3": 0.0, 
        "costo_bombeo_kwh_m3": 0.85,
        "ha_conservadas_base": 3600.0
    },
    "Río Grande II": {
        "capacidad_util_Mm3": 220.0, 
        "afluentes_naturales": {"Río Grande": 10.0, "Río Chico": 3.0, "Quebrada Las Ánimas": 2.0},
        "demanda_acueducto_m3s": 6.5, 
        "generacion_energia_m3s": 12.0, 
        "evaporacion_m3s": 0.5,
        "caudal_ecologico_m3s": 1.0,
        "factor_energia_kwh_m3": 0.65,
        "costo_bombeo_kwh_m3": 0.0,
        "ha_conservadas_base": 4500.0
    },
    "El Peñol (Guatapé)": {
        "capacidad_util_Mm3": 1070.0, 
        "afluentes_naturales": {"Río Nare": 35.0},
        "demanda_acueducto_m3s": 0.0, 
        "generacion_energia_m3s": 30.0, 
        "evaporacion_m3s": 1.5,
        "caudal_ecologico_m3s": 2.0,
        "factor_energia_kwh_m3": 1.2,
        "costo_bombeo_kwh_m3": 0.0,
        "ha_conservadas_base": 0.0
    },
    "Punchiná (San Carlos)": {
        "capacidad_util_Mm3": 68.0, 
        "afluentes_naturales": {"Río Guatapé": 40.0, "Río San Carlos": 15.0},
        "demanda_acueducto_m3s": 0.0, 
        "generacion_energia_m3s": 45.0, 
        "evaporacion_m3s": 0.3,
        "caudal_ecologico_m3s": 4.0,
        "factor_energia_kwh_m3": 2.5,
        "costo_bombeo_kwh_m3": 0.0,
        "ha_conservadas_base": 0.0
    },
    "Hidroituango": {
        "capacidad_util_Mm3": 2720.0, 
        "afluentes_naturales": {"Río Cauca": 1100.0},
        "demanda_acueducto_m3s": 0.0, 
        "generacion_energia_m3s": 900.0, 
        "evaporacion_m3s": 5.0,
        "caudal_ecologico_m3s": 200.0, 
        "factor_energia_kwh_m3": 0.9,
        "costo_bombeo_kwh_m3": 0.0,
        "ha_conservadas_base": 0.0
    }
}

# ==============================================================================
# 2. 🧠 EL ALEPH HÍDRICO (Sincronización Demográfica)
# ==============================================================================
import unicodedata

conectado_aleph = False
pob_amva_aleph = None
pob_local_aleph = None

if 'aleph_lugar' in st.session_state:
    aleph_lugar = st.session_state['aleph_lugar']
    aleph_anio = st.session_state.get('aleph_anio', 2025)
    
    # Extraemos población real desde la memoria
    datos_metabolismo = obtener_metabolismo_exacto(aleph_lugar, aleph_anio)
    aleph_pob = datos_metabolismo['pob_total']
    
    if aleph_pob > 0:
        conectado_aleph = True
        lugar_limpio = unicodedata.normalize('NFKD', str(aleph_lugar).lower()).encode('ascii', 'ignore').decode('utf-8')
        
        claves_rg2 = ["belmira", "donmatias", "san pedro", "entrerrios", "santa rosa", "chico", "grande", "animas"]
        claves_lafe = ["retiro", "ceja", "rionegro", "negro", "espiritu santo", "pantanillo", "buey", "piedras", "arma"]
        claves_amva = ["medellin", "bello", "itagui", "envigado", "sabaneta", "copacabana", "estrella", "girardota", "caldas", "barbosa", "aburra", "amva", "total"]
        
        if any(x in lugar_limpio for x in claves_amva): st.session_state['nodo_sugerido'] = "La Fe" 
        elif any(x in lugar_limpio for x in claves_rg2): st.session_state['nodo_sugerido'] = "Río Grande II"
        elif any(x in lugar_limpio for x in claves_lafe): st.session_state['nodo_sugerido'] = "La Fe"

# =========================================================================
# 3. 🎛️ SIDEBAR Y MOTOR DE CONCESIONES (Cornare / Corantioquia)
# =========================================================================
st.sidebar.markdown("### 🎛️ Centro de Operaciones")

nodos_lista = list(sistemas_embalses.keys())
idx_defecto = 0
if 'nodo_sugerido' in st.session_state and st.session_state['nodo_sugerido'] in nodos_lista:
    idx_defecto = nodos_lista.index(st.session_state['nodo_sugerido'])

nodo_seleccionado = st.sidebar.selectbox("Seleccione el Nodo Principal:", nodos_lista, index=idx_defecto)
datos_nodo = sistemas_embalses[nodo_seleccionado]

# 📜 Lógica Legal de Captación y Trasvase
st.sidebar.markdown("---")
st.sidebar.markdown("#### ⚖️ Marco Legal (Trasvases)")

concesiones_maestras = {
    "La Fe": {
        "Río Pantanillo (Cornare)": 1.5, 
        "Río Buey (Cornare)": 3.0, 
        "Río Piedras (Corantioquia)": 1.2
    },
    "Río Grande II": {
        "Río Grande (Corantioquia)": 10.0, 
        "Río Chico (Corantioquia)": 2.5
    },
    "Piedras Blancas": {
        "Qb. Piedras Blancas (Cornare)": 1.0
    },
    "El Peñol (Guatapé)": {
        "Río Nare (Cornare)": 40.0, 
        "Río Guatapé (Cornare)": 15.0
    }
}

caudal_total_trasvase = 0.0
fuentes_activas = []

if nodo_seleccionado in concesiones_maestras:
    concesiones_sistema = concesiones_maestras[nodo_seleccionado]
    for fuente, q_max in concesiones_sistema.items():
        # Deslizador para simular porcentaje de uso de la concesión otorgada
        pct_uso = st.sidebar.slider(f"{fuente} (Max: {q_max})", 0.0, float(q_max), float(q_max), step=0.1)
        caudal_total_trasvase += pct_uso
        fuentes_activas.append(fuente.split(" ")[1]) # Para los labels del Sankey
    st.sidebar.success(f"💧 **Caudal Transferido:** {caudal_total_trasvase:.2f} m³/s")
else:
    caudal_total_trasvase = st.sidebar.number_input("Caudal de Trasvase (m³/s):", 0.0, 100.0, 0.0)

# =========================================================================
# 4. 🏷️ TÍTULOS Y UI PRINCIPAL
# =========================================================================
st.title(f"🔗 Metabolismo Territorial Complejo: Nodos y Trasvases ({nodo_seleccionado})")
st.markdown("""
Modelo de topología de redes para el **Sistema de Abastecimiento de Agua del Valle de Aburrá y Generación Eléctrica**. 
Evalúa cómo los embalses integran las cuencas propias con los trasvases legales (Cornare/Corantioquia) requeridos para sostener la demanda.
""")

contenedor_sankey = st.empty()

st.markdown("""
<style>
div[data-testid="stExpander"] details summary p { font-family: 'Georgia', serif !important; font-size: 1.15em !important; color: #2c3e50 !important; font-weight: 600 !important; }
div[data-testid="stExpander"] { border: 1px solid #d3c0a3 !important; border-radius: 6px !important; box-shadow: 2px 2px 8px rgba(0,0,0,0.04) !important; background-color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# =========================================================================
# 🖼️ LA OBRA MAESTRA: EL ALEPH DEL AGUA
# =========================================================================
with st.expander("📜 Revelar el Aleph del Agua (Manuscrito Original)", expanded=False):
    st.markdown("""
    <style>
    .museo-wrapper { display: flex; flex-direction: row; align-items: stretch; justify-content: center; gap: 2rem; margin: 1rem 0; flex-wrap: wrap; }
    .aleph-frame { flex: 0 0 65%; border: 8px solid #3e2723; border-radius: 4px; box-shadow: 0px 15px 30px rgba(0,0,0,0.8); overflow: hidden; background-color: #000; }
    .aleph-frame img { display: block; width: 100%; height: auto; transition: transform 1.5s ease, filter 1.5s ease; filter: brightness(0.35) sepia(0.7) blur(1px); }
    .museo-wrapper:hover .aleph-frame img { transform: scale(1.02); filter: brightness(1.0) sepia(0) blur(0px); }
    .aleph-pergamino-side { flex: 1; background-color: #fdfaf2; color: #2c3e50; text-align: justify; border: 2px solid #d3c0a3; border-radius: 8px; padding: 25px; box-shadow: 0px 10px 20px rgba(0,0,0,0.5); opacity: 0; transform: translateX(-30px); transition: all 1.0s cubic-bezier(0.175, 0.885, 0.32, 1.275); font-family: 'Georgia', serif; overflow-y: auto; max-height: 800px; }
    .museo-wrapper:hover .aleph-pergamino-side { opacity: 1; transform: translateX(0); }
    .pergamino-titulo { font-size: 1.4em; font-weight: bold; color: #5d4037; text-align: center; border-bottom: 2px solid #d3c0a3; padding-bottom: 10px; margin-bottom: 15px; }
    .pergamino-seccion { font-size: 1.05em; font-weight: bold; color: #2980b9; margin-top: 15px; margin-bottom: 5px; }
    .pergamino-texto { font-size: 0.9em; line-height: 1.6; }
    </style>
    """, unsafe_allow_html=True)

    url_imagen_aleph = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/imagenes/El%20Aleph%20del%20Agua.png"

    st.markdown(f"""
<div class="museo-wrapper">
    <div class="aleph-frame"><img src="{url_imagen_aleph}" alt="El Aleph del Agua"></div>
    <div class="aleph-pergamino-side">
        <div class="pergamino-titulo">El Aleph del Agua: Síntesis Visual Mística-Científica</div>
        <div class="pergamino-texto">Esta obra está plasmada sobre un pergamino envejecido y texturizado, diseñado para emular un manuscrito de Leonardo da Vinci o las láminas de Humboldt, con una complejidad que trasciende las épocas.</div>
        <div class="pergamino-seccion">1. El Punto Focal: El Aleph</div>
        <div class="pergamino-texto">En el centro geométrico brilla una pequeña esfera con "fulgor casi intolerable tornasolado". Contiene la totalidad de Antioquia y del cosmos.</div>
        <div class="pergamino-seccion">2. La Estructura Fractal</div>
        <div class="pergamino-texto">Un "Engranaje Anatómico-Hídrico" inspirado en Da Vinci. Los flujos irradian transformándose en ríos fractales, filtrándose por cortes geológicos (acuíferos). El agua bautiza la biología (bosques, páramos) y en el borde externo, se vuelve infraestructura y sociedad.</div>
        <div class="pergamino-seccion">3. Texto y Mito</div>
        <div class="pergamino-texto">Fragmentos de "El Aleph" y Pessoa rodean la esfera. En los rincones, deidades como Bochica, Poseidón y Tláloc observan el ciclo, superpuestos con vocabulario científico (Precipitación, Evapotranspiración).</div>
    </div>
</div>
<p style="text-align: center; color: #7f8c8d; font-style: italic; font-size: 0.9em; margin-top: 15px;">Acércate a la obra para encender el Aleph y revelar sus secretos.</p>
    """, unsafe_allow_html=True)

if conectado_aleph:
    with st.expander("🧠 Conexión Activa con el Modelo Demográfico (El Aleph)", expanded=False):
        st.success(f"Recibiendo proyección para **{aleph_lugar}** (Año **{aleph_anio}**): **{aleph_pob:,.0f} habitantes**.")

# =========================================================================
# 5. CARGA DE CARTOGRAFÍA (Desde Supabase en la Nube)
# =========================================================================
url_supabase = None
if "SUPABASE_URL" in st.secrets: url_supabase = st.secrets["SUPABASE_URL"]
elif "supabase" in st.secrets: url_supabase = st.secrets["supabase"].get("url") or st.secrets["supabase"].get("SUPABASE_URL")
elif "connections" in st.secrets and "supabase" in st.secrets["connections"]: url_supabase = st.secrets["connections"]["supabase"]["SUPABASE_URL"]

gdf_embalses = None 
if url_supabase:
    nombre_bucket = "sihcli_maestros"
    nombre_archivo = "embalses_CV_9377.geojson"
    ruta_embalses_nube = f"{url_supabase}/storage/v1/object/public/{nombre_bucket}/Puntos_de_interes/{nombre_archivo}"
    
    try:
        gdf_embalses = gpd.read_file(ruta_embalses_nube)
    except Exception as e:
        pass # Fallback silencioso para no interrumpir UI

if gdf_embalses is not None and not gdf_embalses.empty:
    col_nombre = next((c for c in gdf_embalses.columns if 'nom' in c.lower() or 'proyect' in c.lower() or 'embalse' in c.lower()), None)
    col_vol = next((c for c in gdf_embalses.columns if 'vol' in c.lower() or 'cap' in c.lower()), None)
    
    if col_nombre and col_vol:
        def inyectar_capacidad_real(nombre_nodo, texto_busqueda):
            try:
                match = gdf_embalses[gdf_embalses[col_nombre].astype(str).str.contains(texto_busqueda, case=False, na=False)]
                if not match.empty:
                    vol_real = match.iloc[0][col_vol]
                    if pd.notnull(vol_real) and float(vol_real) > 0:
                        if float(vol_real) > 10000: vol_real = float(vol_real) / 1000000
                        sistemas_embalses[nombre_nodo]["capacidad_util_Mm3"] = round(float(vol_real), 2)
            except: pass

        inyectar_capacidad_real("La Fe", "fe")
        inyectar_capacidad_real("Río Grande II", "grande")
        inyectar_capacidad_real("El Peñol (Guatapé)", "peñol|guatap")
        inyectar_capacidad_real("Punchiná (San Carlos)", "punchin|carlos")
        inyectar_capacidad_real("Hidroituango", "ituango")

        # -----------------------------------------------------------------
        # C. INYECCIÓN DE PREDIOS CONSERVADOS (SbN) DESDE SUPABASE
        # -----------------------------------------------------------------
        try:
            sistemas_embalses["La Fe"]["ha_conservadas_base"] = 3600.0
            
            ruta_predios = f"{url_supabase}/storage/v1/object/public/{nombre_bucket}/Puntos_de_interes/PrediosEjecutados.geojson"
            import requests, tempfile
            
            res_predios = requests.get(ruta_predios)
            if res_predios.status_code == 200:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".geojson") as tmp_p:
                    tmp_p.write(res_predios.content)
                    tmp_path_p = tmp_p.name
                
                gdf_predios = gpd.read_file(tmp_path_p)
                
                if 'EMBALSE' in gdf_predios.columns and 'AREA_HA' in gdf_predios.columns:
                    resumen_predios = gdf_predios.groupby('EMBALSE')['AREA_HA'].sum().to_dict()
                    for embalse_mapa, ha_total in resumen_predios.items():
                        nombre_mapa_limpio = str(embalse_mapa).lower().strip()
                        for nombre_nodo in sistemas_embalses.keys():
                            if nombre_mapa_limpio in nombre_nodo.lower() or (nombre_mapa_limpio == "riogrande ii" and "grande" in nombre_nodo.lower()):
                                sistemas_embalses[nombre_nodo]["ha_conservadas_base"] += float(ha_total)
                                
                    st.sidebar.success("🌲 Áreas de conservación inyectadas correctamente.")
        except Exception as e:
            pass

# --- ☁️ MOTOR CLIMÁTICO LOCAL (ENSO) ---
st.sidebar.markdown("### 🌍 Motor Climático (ENSO)")
escenario_enso = st.sidebar.select_slider(
    "Fase actual del Pacífico:",
    options=["Niño Severo", "Niño Moderado", "Neutro", "Niña Moderada", "Niña Fuerte"],
    value="Neutro"
)

# Acoplamiento Termodinámico
if escenario_enso == "Niño Severo": factor_p, factor_et = 0.65, 1.40
elif escenario_enso == "Niño Moderado": factor_p, factor_et = 0.85, 1.20
elif escenario_enso == "Neutro": factor_p, factor_et = 1.0, 1.0
elif escenario_enso == "Niña Moderada": factor_p, factor_et = 1.15, 0.85
else: factor_p, factor_et = 1.35, 0.70  # Niña Fuerte

# =========================================================================
# 💧 MÓDULO 1: BALANCE DE MASA EN TIEMPO REAL
# =========================================================================
with st.expander(f"💧 Balance de Masa en Tiempo Real: {nodo_seleccionado}", expanded=False):
    if factor_p < 1.0:
        st.error(f"⚠️ **{escenario_enso}:** Oferta Hídrica **{(factor_p-1)*100:+.0f}%** | Evaporación (ET) **{(factor_et-1)*100:+.0f}%** debido a anomalía térmica.")
    elif factor_p > 1.0:
        st.info(f"🌧️ **{escenario_enso}:** Oferta Hídrica **{(factor_p-1)*100:+.0f}%** | Evaporación (ET) reducida al **{(factor_et)*100:.0f}%**.")
    else:
        st.info(f"**Capacidad Útil Máxima:** {datos_nodo['capacidad_util_Mm3']:,.1f} Mm³ (Condiciones Climáticas Históricas)")

    col_in, col_out = st.columns(2)

    # --- ENTRADAS DINÁMICAS ---
    afluentes_inputs = {}
    trasvases_inputs = {}

    with col_in:
        st.markdown("#### 📥 ENTRADAS (Inflows)")
        st.caption("Aportes Naturales de la Cuenca (Afectados por ENSO):")
        
        caudal_real_aleph = st.session_state.get('aleph_q_rio_m3s', 0.0)
        
        for i, (nombre, caudal) in enumerate(datos_nodo["afluentes_naturales"].items()):
            if i == 0 and caudal_real_aleph > 0:
                caudal_base = caudal_real_aleph
                nombre_mostrar = f"💧 {nombre} (Física Real)"
            else:
                caudal_base = caudal
                nombre_mostrar = nombre
                
            max_val = float(caudal_base * 3) if caudal_base > 0 else 10.0
            caudal_afectado = min(float(caudal_base * factor_p), max_val) 
            afluentes_inputs[nombre_mostrar] = st.slider(f"{nombre_mostrar} [m³/s]:", 0.0, max_val, caudal_afectado, 0.1, key=f"in_{nombre}")
            
        if caudal_real_aleph > 0:
            st.success("🧠 Oferta Hídrica sincronizada con el modelo distribuido.")
        
        st.caption("Bombas y Túneles (Trasvases Externos):")
        st.info(f"⚖️ Conectado a marco legal: **{caudal_total_trasvase:.2f} m³/s** inyectados desde cuencas cedentes.")

    # --- SALIDAS DINÁMICAS ---
    with col_out:
        st.markdown("#### 📤 SALIDAS (Outflows)")
        
        evaporacion_dinamica = datos_nodo["evaporacion_m3s"] * factor_et
        st.metric("Evaporación Directa (Afectada por T°)", f"{evaporacion_dinamica:.2f} m³/s", f"{(evaporacion_dinamica - datos_nodo['evaporacion_m3s']):+.2f} m³/s", delta_color="inverse")
        
        demanda_memoria = st.session_state.get('demanda_total_m3s', 0.0)
        
        if demanda_memoria > 0 and demanda_memoria >= (datos_nodo["demanda_acueducto_m3s"] * 0.5):
            st.success(f"🧠 Demanda Metabólica Sincronizada: {demanda_memoria:.3f} m³/s")
            val_base_acue = demanda_memoria
            label_acueducto = "🚰 Extracción Metabólica (Real) [m³/s]:"
        else:
            val_base_acue = datos_nodo["demanda_acueducto_m3s"]
            label_acueducto = "Extracción Consuntiva (Teórica) [m³/s]:"
            if demanda_memoria > 0:
                st.info("⚠️ Se ignoró la demanda en memoria por ser de otra escala territorial.")
                
        max_acueducto = float(max(val_base_acue, datos_nodo["demanda_acueducto_m3s"]) * 2)
        val_acueducto = st.slider(label_acueducto, 0.0, max_acueducto, float(val_base_acue), 0.05) if max_acueducto > 0 else 0.0
        
        val_turbinado = 0.0
        if datos_nodo["generacion_energia_m3s"] > 0:
            max_turb = float(datos_nodo["generacion_energia_m3s"] * 1.5)
            val_def_turb = min(float(datos_nodo["generacion_energia_m3s"]), max_turb)
            val_turbinado = st.slider("Caudal Turbinado (Energía) [m³/s]:", 0.0, max_turb, val_def_turb, 1.0)
            
        val_ecologico = st.number_input("Caudal Ecológico / Vertimiento [m³/s]:", min_value=0.0, value=float(datos_nodo["caudal_ecologico_m3s"]), step=1.0)

    # CÁLCULO DE BALANCE Y ENERGÍA
    sum_entradas = sum(afluentes_inputs.values()) + caudal_total_trasvase
    sum_salidas = val_acueducto + val_turbinado + val_ecologico + evaporacion_dinamica # Sensible al ENSO
    balance = sum_entradas - sum_salidas

    # --- HUELLA ENERGÉTICA ---
    m3_hora_turbinados = val_turbinado * 3600
    m3_hora_bombeados = caudal_total_trasvase * 3600
    potencia_generada_kw = m3_hora_turbinados * datos_nodo["factor_energia_kwh_m3"]
    potencia_consumida_kw = m3_hora_bombeados * datos_nodo["costo_bombeo_kwh_m3"]
    balance_energetico_MW = (potencia_generada_kw - potencia_consumida_kw) / 1000

    ingreso_hora_cop = potencia_generada_kw * 350
    costo_hora_cop = potencia_consumida_kw * 350

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Balance Hídrico (ΔS/Δt)", f"{balance:+.1f} m³/s", "Llenándose 📈" if balance > 0 else "Vaciándose 📉" if balance < 0 else "Estable ⚖️")
    c2.metric("Energía Generada", f"{potencia_generada_kw/1000:,.1f} MW", f"${ingreso_hora_cop/1e6:,.1f} M/hora", delta_color="normal")
    c3.metric("Energía Consumida", f"{potencia_consumida_kw/1000:,.1f} MW", f"-${costo_hora_cop/1e6:,.1f} M/hora", delta_color="inverse")
    c4.metric("Balance Neto", f"{balance_energetico_MW:,.1f} MW", "Superávit ⚡" if balance_energetico_MW > 0 else "Déficit 🔴" if balance_energetico_MW < 0 else "Neutro")
    
# =========================================================================
# 4. TABLERO WRI: NEUTRALIDAD, RESILIENCIA Y CALIDAD
# =========================================================================
with st.expander(f"🌐 Inteligencia Territorial WRI: {nodo_seleccionado}", expanded=False):

    anio_analisis = st.slider("Seleccione el Año de Evaluación (Actual o Futuro):", min_value=2024, max_value=2050, value=2025, step=1)

    delta_anios = anio_analisis - 2025
    factor_demanda = (1 + 0.015) ** delta_anios
    factor_clima = (1 - 0.005) ** delta_anios

    q_oferta_m3s_base = sum_entradas
    demanda_m3s_base = val_acueducto + (val_turbinado * 0.1) 
    capacidad_embalse_m3 = datos_nodo["capacidad_util_Mm3"] * 1000000

    oferta_anual_m3 = (q_oferta_m3s_base * factor_clima) * 31536000
    consumo_anual_m3 = (demanda_m3s_base * factor_demanda) * 31536000

    # 🌊 ALERTA DINÁMICA ENSO (Sincronización con el Slider Lateral)
    if factor_p < 1.0:
        # Calculamos cuántos meses de reserva le quedan al sistema si el consumo supera la oferta mermada
        if consumo_anual_m3 > oferta_anual_m3:
            meses_reserva = (capacidad_embalse_m3) / ((consumo_anual_m3 - oferta_anual_m3) / 12)
        else:
            meses_reserva = 99.0
            
        if meses_reserva < 12:
            st.markdown(f"""
            <div style='background-color: #fdf2e9; border-left: 5px solid #e67e22; padding: 15px; border-radius: 5px; margin-bottom: 15px;'>
                <h5 style='color: #d35400; margin-top: 0;'>⚠️ Impacto '{escenario_enso}' en Seguridad Hídrica</h5>
                La reducción de lluvias está forzando al sistema a consumir sus reservas estructurales. Si esta anomalía térmica se mantiene, Medellín y el Valle de Aburrá entrarían en <b>riesgo de racionamiento en {meses_reserva:.1f} meses</b>.
            </div>
            """, unsafe_allow_html=True)

    # --- 2. INTEGRACIÓN CARTOGRÁFICA Y SIMULACIÓN SbN DINÁMICA ---
    st.markdown("---")
    st.markdown(f"#### 🌲 Beneficios Volumétricos (SbN) en el Sistema: **{nodo_seleccionado}**")

    ha_reales_sig = float(datos_nodo.get("ha_conservadas_base", 0.0))
    st.markdown("##### ⚙️ Escenario Base vs. Proyectado")
    activar_sig = st.toggle("✅ Incluir Área Restaurada/Conservada del SIG en el cálculo WRI", value=True, 
                            help="Apaga este interruptor para simular el escenario contrafactual.")

    # 🚨 MEJORA ESTRUCTURAL: Definimos el área activa que realmente opera como filtro
    ha_base_calculo = ha_reales_sig if activar_sig else 0.0

    c_inv1, c_inv2, c_inv3 = st.columns(3)
    with c_inv1:
        st.metric("✅ Área Restaurada/Conservada (SIG)", f"{ha_reales_sig:,.1f} ha", "Línea base actual")
        ha_simuladas = st.number_input("➕ Adicionar Hectáreas (Simulación):", min_value=0.0, value=0.0, step=50.0, key="sim_ha_new")
        
        # El área total es la suma de la base activa más lo que el usuario quiera simular
        ha_operativas_totales = ha_base_calculo + ha_simuladas
        beneficio_restauracion_m3 = ha_operativas_totales * 2500 
        
    with c_inv2:
        val_base_stam = 120
        sist_saneamiento = st.number_input("Sistemas Tratamiento (STAM):", min_value=0, value=val_base_stam if activar_sig else 0, step=5)
        beneficio_calidad_m3 = (sist_saneamiento * 1200) 
        
    with c_inv3:
        volumen_repuesto_m3 = beneficio_restauracion_m3 + beneficio_calidad_m3
        st.metric("💧 Agua 'Devuelta' (VWBA)", f"{volumen_repuesto_m3/1e6:,.2f} Mm³/año", "Total compensado")

    # =========================================================================
    # ☣️ MOTOR DE CARGAS CON FILTRACIÓN DINÁMICA (Base + Simulación)
    # =========================================================================

    # 🌪️ CAPTURA DE DATOS (Blindaje de Variables)
    eco_fosforo_kg = st.session_state.get('eco_fosforo_kg', 0.0)
    activar_tormenta = st.session_state.get('activar_tormenta_sankey', False)

    # 1. Configuración de Población Local (Conectado a Módulo 6)
    pob_hum_local = st.session_state.get(f'pob_asig_{nodo_seleccionado}_met', 15000)
    pob_bov_local = st.session_state.get('ica_bovinos_calc_met', 5000)
    pob_por_local = st.session_state.get('ica_porcinos_calc_met', 2000)
    
    # 2. FACTOR DE FILTRACIÓN DINÁMICO (Vínculo Estructural con Simulación)
    # A más hectáreas (reales o simuladas), mayor es el factor de filtración.
    # 3600 ha equivalen al 60% de eficiencia de retención.
    eficiencia_filtro_dinamico = min(0.90, (ha_operativas_totales / 6000.0)) if ha_operativas_totales > 0 else 0.0
    
    dr_difuso = 0.15 * (1.0 - eficiencia_filtro_dinamico)
    dr_puntual = 0.80 

    # 3. Cálculo de Carga Neta (Ahora sensible a la simulación manual)
    carga_tormenta_ton = (eco_fosforo_kg * 10) / 1000.0 if activar_tormenta else 0.0
    carga_neta_ton = (((pob_bov_local * 0.18) + (pob_por_local * 0.11)) * dr_difuso) + ((pob_hum_local * 0.018) * dr_puntual) + carga_tormenta_ton
    
    # 4. Balance de Remoción (Saneamiento Base + Proyectado)
    sist_saneamiento_base = 40 
    sist_saneamiento_total = sist_saneamiento_base + sist_saneamiento
    carga_removida_ton = sist_saneamiento_total * 2.2
    
    carga_final_rio_ton = max(0.0, carga_neta_ton - carga_removida_ton)
    
    # 🚨 INTERVENCIÓN 3: Prevenir sobreescritura de la carga total del Módulo 6
    st.session_state['carga_dbo_mitigada_ton'] = carga_final_rio_ton

    # =========================================================================
    # 🌊 MOTOR DE CALIDAD E INDICADORES: BALANCE DE MASA ESTRUCTURAL
    # =========================================================================
    
    # 1. Seguridad Hídrica (WEI+)
    consumo_real_validado = max(consumo_anual_m3, (val_acueducto * 31536000))
    estres_decimal = consumo_real_validado / (oferta_anual_m3 + 1.0)
    ind_estres = max(0.0, min(100.0, 100.0 - (estres_decimal * 120))) 

    # 2. Neutralidad Hídrica (VWBA)
    ind_neutralidad = min(100.0, (volumen_repuesto_m3 / consumo_real_validado) * 100) if consumo_real_validado > 0 else 0.0

    # 3. Calidad de Agua (WQI) con Efecto de Zona Crítica
    # Recuperamos la carga orgánica local
    carga_total_ton_local = carga_final_rio_ton 
    factor_heterogeneidad = 3.5 # Las colas del embalse sufren mayor concentración
    
    if activar_tormenta:
        carga_total_ton_local *= 2.5 # Arrastre masivo superficial
    
    carga_mg_s_final = (carga_total_ton_local * factor_heterogeneidad * 1_000_000_000) / 31536000 
    
    # Caudal de dilución (Natural + Trasvases para dilución real en embalse)
    q_natural_local = sum(datos_nodo["afluentes_naturales"].values())
    q_trasvases_local = caudal_total_trasvase
    caudal_L_s_final = ((q_natural_local + q_trasvases_local) if (q_natural_local + q_trasvases_local) > 0 else 0.1) * 1000
    
    concentracion_dbo_final = carga_mg_s_final / caudal_L_s_final
    # Divisor 8.0 para que >4mg/L de DBO empiece a desplomar el índice (Realismo en embalses)
    ind_calidad = max(0.0, min(100.0, 100.0 - (concentracion_dbo_final / 8.0 * 100)))

    # 4. Resiliencia Estructural (Buffer - Lodo de Fondo)
    capacidad_util_ajustada = capacidad_embalse_m3
    if activar_tormenta:
        lodo_fondo_inyectado = st.session_state.get('eco_lodo_fondo_m3', 0.0)
        # Si no hay bosque (activar_sig OFF), el lodo de fondo es un 40% más severo por falta de retención
        penalidad_sbn = 1.0 if activar_sig else 1.4
        capacidad_util_ajustada -= (lodo_fondo_inyectado * 1.5 * penalidad_sbn)
        capacidad_util_ajustada = max(0, capacidad_util_ajustada)

    buffer_ratio = (capacidad_util_ajustada + oferta_anual_m3) / consumo_real_validado if consumo_real_validado > 0 else 5.0
    ind_resiliencia = min(100.0, (buffer_ratio / 2.5) * 100)

    # =========================================================================
    # 🎨 FUNCIONES DE EVALUACIÓN Y RENDERIZADO (INTEGRIDAD TOTAL)
    # =========================================================================
    def evaluar_indice(valor, umbral_rojo, umbral_verde, invertido=False):
        if not invertido: 
            return ("🔴 CRÍTICO", "#c0392b") if valor < umbral_rojo else ("🟡 VULNERABLE", "#f39c12") if valor < umbral_verde else ("🟢 ÓPTIMO", "#27ae60")
        else: 
            return ("🟢 HOLGADO", "#27ae60") if valor < umbral_verde else ("🟡 MODERADO", "#f39c12") if valor < umbral_rojo else ("🔴 CRÍTICO", "#c0392b")

    def generar_leyenda(u_r, u_v, inv):
        if not inv: 
            return f"🔴 < {u_r}% | 🟡 {u_r}-{u_v}% | 🟢 > {u_v}%"
        else: 
            return f"🟢 < {u_v}% | 🟡 {u_v}-{u_r}% | 🔴 > {u_r}%"

    # =========================================================================
    def crear_velocimetro(valor, titulo, color_bar, umbral_rojo, umbral_verde, invertido=False):
    
        # 🌪️ AJUSTE DE TÍTULO ANTE TORMENTA
        if "Resiliencia" in titulo and st.session_state.get('activar_tormenta_sankey', False):
            titulo = f"🌪️ {titulo} (Colmatación)"

        # 🚨 CORRECCIÓN DE VISIBILIDAD: Forzamos la aparición del número central
        fig = go.Figure(go.Indicator(
            mode = "gauge+number", 
            value = float(valor), # Garantizamos que ingrese como escalar numérico
            number = {
                'suffix': "%", 
                # Reducimos de 28 a 24 para que valores como "100.0%" no desborden la caja
                'font': {'size': 24, 'color': '#2c3e50', 'family': 'Arial'}, 
                'valueformat': ".1f"
            },
            # Reducimos sutilmente el título para evitar colisiones superiores
            title = {'text': titulo, 'font': {'size': 14, 'family': 'Georgia', 'color': '#34495e'}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': color_bar},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, umbral_rojo], 'color': "#ffcccb" if not invertido else "#e8f8f5"},
                    {'range': [umbral_rojo, umbral_verde], 'color': "#fff2cc"},
                    {'range': [umbral_verde, 100], 'color': "#e8f8f5" if not invertido else "#ffcccb"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4}, 
                    'thickness': 0.75, 
                    'value': valor
                }
            }
        ))
        
        # 🛠️ AMPLIACIÓN DE MÁRGENES: Elevamos 'height' a 240 y el margen inferior 'b' a 30
        fig.update_layout(height=240, margin=dict(l=20, r=20, t=50, b=30))
        return fig
        
    # =========================================================================
    # 🎛️ PANEL DE CALIBRACIÓN AVANZADA (WQI): SENSIBILIDAD AMBIENTAL
    # =========================================================================
    with st.expander("⚙️ Calibración del Modelo de Calidad (WQI)", expanded=True):
        st.caption("Ajuste los parámetros físicos para evitar el colapso matemático por saturación de carga.")
        c_cal1, c_cal2 = st.columns(2)
        with c_cal1:
            umbral_dbo_colapso = st.slider(
                "Umbral de Colapso DBO (mg/L):", 
                5.0, 50.0, 25.0, 1.0, # Subimos a 25.0 para salir del cero
                help="Concentración donde la calidad cae a 0%. Sugerido La Fe: 25-30 mg/L."
            )
        with c_cal2:
            factor_zona_critica = st.slider(
                "Factor de Zona Crítica:", 
                1.0, 10.0, 2.5, 0.5,
                help="Multiplicador de concentración en colas del embalse. 2.5 es estándar para embalses alargados."
            )

    # =========================================================================
    # 🌪️ RECÁLCULO DINÁMICO DE ALTA SENSIBILIDAD
    # =========================================================================
    # 1. SEGURIDAD HÍDRICA (WEI+)
    val_seguridad = max(0.0, min(100.0, 100.0 - (consumo_real_validado / (oferta_anual_m3 + 1.0) * 120)))

    # 2. RESILIENCIA ESTRUCTURAL
    cap_dinamica = capacidad_embalse_m3
    if st.session_state.get('activar_tormenta_sankey', False):
        cap_dinamica -= (st.session_state.get('eco_lodo_fondo_m3', 0.0) * 1.5)
        cap_dinamica = max(0, cap_dinamica)
    val_resiliencia = min(100.0, ((cap_dinamica + oferta_anual_m3) / consumo_real_validado / 2.5) * 100) if consumo_real_validado > 0 else 100.0

    # 3. CALIDAD (WQI): Aplicación de Calibración
    q_nat_loc = sum(datos_nodo["afluentes_naturales"].values())
    q_tras_loc = sum(trasvases_inputs.values())
    caudal_L_s_calc = ((q_nat_loc + q_tras_loc) if (q_nat_loc + q_tras_loc) > 0 else 0.1) * 1000
    carga_mg_s_calc = (carga_final_rio_ton * 1e9) / 31536000
    
    # El factor se incrementa bajo tormenta para simular el 'tapón' de carga inicial
    f_ajustado = factor_zona_critica if not st.session_state.get('activar_tormenta_sankey', False) else (factor_zona_critica * 1.5)
    conc_dbo_inst = (carga_mg_s_calc * f_ajustado) / caudal_L_s_calc
    val_calidad = max(0.0, min(100.0, 100.0 - (conc_dbo_inst / umbral_dbo_colapso * 100)))

    # --- 📦 CONFIGURACIÓN DE VISUALIZACIÓN (SIN LATENCIA) ---
    col_g = st.columns(4)
    data_viz = [
        {"val": ind_neutralidad, "tit": "Neutralidad", "col": "#2ecc71", "u_r": 40, "u_v": 70, "inv": False},
        {"val": val_resiliencia, "tit": "Resiliencia", "col": "#3498db", "u_r": 30, "u_v": 60, "inv": False},
        {"val": val_seguridad, "tit": "Seguridad (WEI+)", "col": "#e74c3c", "u_r": 40, "u_v": 75, "inv": False},
        {"val": val_calidad, "tit": "Calidad (WQI)", "col": "#9b59b6", "u_r": 50, "u_v": 80, "inv": False}
    ]

    for i, item in enumerate(data_viz):
        with col_g[i]:
            # Evaluamos estado antes de pintar
            est, color_txt = evaluar_indice(item["val"], item["u_r"], item["u_v"], item["inv"])
            
            # Renderizamos el velocímetro optimizado (ahora con números visibles)
            st.plotly_chart(crear_velocimetro(item["val"], item["tit"], item["col"], item["u_r"], item["u_v"], item["inv"]), use_container_width=True)
            
            # Subtítulos de Estado y Leyenda con contraste mejorado
            st.markdown(f"<p style='text-align: center; color: {color_txt}; font-weight: bold; margin-top:-35px; font-size: 16px;'>{est}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; font-size: 11px; color: #7f8c8d; margin-top: -10px;'>{generar_leyenda(item['u_r'], item['u_v'], item['inv'])}</p>", unsafe_allow_html=True)
            
            # Aclaración específica para La Fe
            if item["tit"] == "Neutralidad" and nodo_seleccionado == "La Fe":
                st.caption("ℹ️ Aporte marginal Q. Esp. Santo vs Valle de Aburrá")

    # 🪄 MAGIA: Convertimos el expander anidado en un toggle elegante
    st.markdown("<br>", unsafe_allow_html=True)
    if st.toggle("📚 Mostrar Conceptos, Metodología y Fuentes (VWBA - WRI)"):
        st.info("""
        ### 📖 Glosario de Indicadores
        * **Neutralidad Hídrica (Volumetric Water Benefit VWBA):** Mide si el volumen de agua restituido a la cuenca mediante Soluciones Basadas en la Naturaleza (SbN) compensa la Huella Hídrica del consumo humano/industrial.
        * **Resiliencia Territorial:** Capacidad del ecosistema para soportar eventos de sequía (El Niño) sin colapsar el suministro.
        * **Estrés Hídrico (Indicador Falkenmark / ODS 6.4.2):** Porcentaje de la oferta total anual que está siendo extraída por los diversos sectores económicos.
        * **Calidad de Agua (WQI):** Índice modificado basado en la capacidad de dilución natural y mitigación sanitaria.
          
        ### 🌐 Fuentes y Estándares de Referencia
        * **WRI (World Resources Institute):** Volumetric Water Benefit Accounting (VWBA).
        * **CEO Water Mandate:** Resiliencia hídrica corporativa.
        """)
        
# --- OPTIMIZADOR DE INVERSIONES Y METAS (PORTAFOLIO INTEGRAL) ---
st.subheader("💼 Portafolios de Inversión Multi-Objetivo")

# -------------------------------------------------------------------------
# PORTAFOLIO 1: CANTIDAD (NEUTRALIDAD HÍDRICA)
# -------------------------------------------------------------------------
with st.expander("🎯 1. Optimización de Brechas: Oferta y Demanda (Neutralidad)", expanded=False):
    st.markdown("Combina estrategias (SbN, Saneamiento, Eficiencia) y estima el presupuesto necesario para compensar la Huella Hídrica.")
    
    col_m1, col_m2 = st.columns([1, 2.5])
    with col_m1:
        meta_neutralidad = st.slider("🎯 Objetivo de Neutralidad (%)", 10.0, 100.0, 100.0, 5.0)
        st.markdown("**Costos Unitarios (Millones COP):**")
        costo_ha = st.number_input("Restauración (1 ha):", value=8.5, step=0.5)
        costo_stam_n = st.number_input("Saneamiento (1 STAM):", value=15.0, step=1.0)
        costo_lps = st.number_input("Eficiencia (1 L/s ahorrado):", value=120.0, step=10.0)
    
    with col_m2:
        vol_requerido_m3 = (meta_neutralidad / 100.0) * consumo_anual_m3
        brecha_m3 = vol_requerido_m3 - volumen_repuesto_m3
        
        if brecha_m3 <= 0:
            st.success(f"✅ ¡El sistema cumple o supera la meta del {meta_neutralidad}% con las acciones actuales!")
        else:
            st.warning(f"⚠️ **Déficit para la meta:** Faltan compensar **{brecha_m3/1e6:,.2f} Mm³/año**.")
            
            st.markdown("🎚️ **Diseña tu Mix de Intervención (% de aporte):**")
            c_mix1, c_mix2, c_mix3 = st.columns(3)
            pct_a = c_mix1.number_input("% Restauración", 0, 100, 40)
            pct_b = c_mix2.number_input("% Saneamiento", 0, 100, 40, key="pct_b_neut")
            pct_c = c_mix3.number_input("% Eficiencia", 0, 100, 20)
            
            total_pct = pct_a + pct_b + pct_c
            if total_pct != 100:
                st.error(f"La suma de las estrategias debe ser 100%. Actual: {total_pct}%")
            else:
                vol_a = brecha_m3 * (pct_a / 100.0)
                vol_b = brecha_m3 * (pct_b / 100.0)
                vol_c = brecha_m3 * (pct_c / 100.0)
                
                ha_req = vol_a / 2500.0
                stam_req = vol_b / 1200.0
                lps_req = (vol_c * 1000) / 31536000 
                
                inv_a = ha_req * costo_ha
                inv_b = stam_req * costo_stam_n
                inv_c = lps_req * costo_lps
                inv_total = inv_a + inv_b + inv_c
                
                st.markdown("📊 **Requerimientos Físicos y Presupuesto:**")
                c_op1, c_op2, c_op3, c_op4 = st.columns(4)
                
                costo_L_a = (inv_a * 1000000) / (vol_a * 1000) if vol_a > 0 else 0
                costo_L_b = (inv_b * 1000000) / (vol_b * 1000) if vol_b > 0 else 0
                costo_L_c = (inv_c * 1000000) / (vol_c * 1000) if vol_c > 0 else 0
                
                c_op1.metric("🌲 Restaurar (ha)", f"{ha_req:,.1f}", f"${inv_a:,.0f} Millones")
                c_op1.caption(f"**Costo Eficiencia:** ${costo_L_a:,.1f} COP por Litro")
                c_op2.metric("🚽 Saneamiento (STAM)", f"{stam_req:,.0f}", f"${inv_b:,.0f} Millones")
                c_op2.caption(f"**Costo Eficiencia:** ${costo_L_b:,.1f} COP por Litro")
                c_op3.metric("🚰 Eficiencia (L/s)", f"{lps_req:,.1f}", f"${inv_c:,.0f} Millones")
                c_op3.caption(f"**Costo Eficiencia:** ${costo_L_c:,.1f} COP por Litro")
                c_op4.metric("💰 INVERSIÓN (CANTIDAD)", f"${inv_total:,.0f} M", "Millones COP", delta_color="off")

# -------------------------------------------------------------------------
# PORTAFOLIO 2: CALIDAD (SANEAMIENTO Y REMOCIÓN DBO)
# -------------------------------------------------------------------------
with st.expander("🎯 2. Optimización de Cargas Contaminantes (Saneamiento DBO5)", expanded=False):
    st.markdown("Combina infraestructura gris y verde para estimar el presupuesto necesario para limpiar la cuenca.")
    
    # Lee la carga del metabolismo (Sección 6) a través de la memoria global
    carga_total_ton = st.session_state.get('carga_dbo_total_ton', 1000.0)
    carga_removida_actual = 0.0 
    
    col_c1, col_c2 = st.columns([1, 2.5])
    with col_c1:
        meta_remocion = st.slider("🎯 Meta de Remoción de DBO (%)", 10.0, 100.0, 85.0, 5.0)
        st.markdown("**Costos Unitarios (Millones COP por Ton/año):**")
        costo_ptar = st.number_input("PTAR Urbana (1 Ton DBO/a):", value=150.0, step=10.0)
        costo_stam_c = st.number_input("STAM Rural (1 Ton DBO/a):", value=45.0, step=5.0)
        costo_sbn = st.number_input("SbN/Filtros (1 Ton DBO/a):", value=12.0, step=2.0)
    
    with col_c2:
        carga_objetivo_remover = (meta_remocion / 100.0) * carga_total_ton
        brecha_ton = carga_objetivo_remover - carga_removida_actual
        
        if brecha_ton <= 0:
            st.success(f"✅ ¡El sistema cumple o supera la meta del {meta_remocion}% de remoción!")
        else:
            st.warning(f"⚠️ **Déficit de Saneamiento:** Faltan remover **{brecha_ton:,.1f} Toneladas/año** de DBO5. *(Base: {carga_total_ton:,.0f} Ton)*")
            
            st.markdown("🎚️ **Diseña tu Mix de Mitigación (% de aporte):**")
            c_mix_c1, c_mix_c2, c_mix_c3 = st.columns(3)
            pct_ptar = c_mix_c1.number_input("% PTAR Urbana", 0, 100, 50)
            pct_stam = c_mix_c2.number_input("% STAM Rural", 0, 100, 30, key="pct_stam_cal")
            pct_sbn = c_mix_c3.number_input("% SbN / Humedales", 0, 100, 20)
            
            total_pct_c = pct_ptar + pct_stam + pct_sbn
            if total_pct_c != 100:
                st.error(f"La suma debe ser 100%. Actual: {total_pct_c}%")
            else:
                ton_ptar = brecha_ton * (pct_ptar / 100.0)
                ton_stam = brecha_ton * (pct_stam / 100.0)
                ton_sbn = brecha_ton * (pct_sbn / 100.0)
                
                inv_ptar = ton_ptar * costo_ptar
                inv_stam = ton_stam * costo_stam_c
                inv_sbn = ton_sbn * costo_sbn
                inv_total_c = inv_ptar + inv_stam + inv_sbn
                
                st.markdown("📊 **Requerimientos Físicos y Presupuesto:**")
                c_op_c1, c_op_c2, c_op_c3, c_op_c4 = st.columns(4)
                
                costo_kg_ptar = (inv_ptar * 1000000) / (ton_ptar * 1000) if ton_ptar > 0 else 0
                costo_kg_stam = (inv_stam * 1000000) / (ton_stam * 1000) if ton_stam > 0 else 0
                costo_kg_sbn = (inv_sbn * 1000000) / (ton_sbn * 1000) if ton_sbn > 0 else 0
                
                c_op_c1.metric("🏙️ PTAR Urbana", f"{ton_ptar:,.0f} Ton", f"${inv_ptar:,.0f} Millones")
                c_op_c1.caption(f"**Eficiencia:** ${costo_kg_ptar:,.0f} COP/Kg DBO")
                c_op_c2.metric("🏡 STAM Rural", f"{ton_stam:,.0f} Ton", f"${inv_stam:,.0f} Millones")
                c_op_c2.caption(f"**Eficiencia:** ${costo_kg_stam:,.0f} COP/Kg DBO")
                c_op_c3.metric("🌿 SbN / Humedales", f"{ton_sbn:,.0f} Ton", f"${inv_sbn:,.0f} Millones")
                c_op_c3.caption(f"**Eficiencia:** ${costo_kg_sbn:,.0f} COP/Kg DBO")
                c_op_c4.metric("💰 INVERSIÓN (CALIDAD)", f"${inv_total_c:,.0f} M", "Millones COP", delta_color="off")

# -------------------------------------------------------------------------
# PORTAFOLIO 3: EL ROI DE LA NATURALEZA (AHORRO INTEGRAL Y VALOR EXISTENCIAL)
# -------------------------------------------------------------------------
with st.expander("🎯 3. El ROI de la Naturaleza (Costo-Beneficio de la Infraestructura Verde)", expanded=False):
    st.markdown("Convierte la ecología en finanzas estratégicas. Evalúa el retorno de inversión (ROI) sumando ahorros en químicos, protección de vida útil (evitación de dragado) y valor de venta del agua garantizada.")
    
    # Leemos el impacto base sincronizado desde la Pág 04
    lodo_tormenta_m3 = st.session_state.get('eco_lodo_total_m3', 41256.0) 
    lodo_fondo_m3 = st.session_state.get('eco_lodo_fondo_m3', 18565.0) # Solo el de fondo afecta vida útil
    sobrecosto_tormenta_usd = st.session_state.get('eco_sobrecosto_usd', 15141.0)
    
    c_roi1, c_roi2 = st.columns([1, 2.5])
    
    with c_roi1:
        st.markdown("**Parámetros del Proyecto:**")
        ha_proteger = st.slider("Hectáreas a Proteger/Restaurar:", 100.0, 5000.0, 500.0, 50.0)
        costo_ha_usd = st.number_input("Costo de Restauración (USD/ha):", value=2500.0, step=100.0)
        eventos_ano = st.slider("Avenidas Torrenciales al año:", 1, 10, 3)
        horizonte_roi = st.slider("Años de evaluación del Proyecto:", 5, 30, 10)
        
    with c_roi2:
        inversion_total = ha_proteger * costo_ha_usd
        
        # --- MOTOR DE MITIGACIÓN DINÁMICA (Curva de Saturación) ---
        mitigacion_pct = (1 - np.exp(-ha_proteger / 1200.0)) * 0.95
        
        # 💰 1. BENEFICIO QUÍMICO (OPEX EVITADO)
        ahorro_anual_quimicos = (sobrecosto_tormenta_usd * mitigacion_pct) * eventos_ano
        total_quimicos = ahorro_anual_quimicos * horizonte_roi
        
        # 💰 2. BENEFICIO DE INFRAESTRUCTURA (EVITACIÓN DE DRAGADO)
        costo_dragado_m3 = 12.0 
        lodo_evitado_m3_total = (lodo_tormenta_m3 * mitigacion_pct) * eventos_ano * horizonte_roi
        total_dragado_evitado = lodo_evitado_m3_total * costo_dragado_m3
        
        # 💰 3. VALOR DEL AGUA PROTEGIDA (VENTA DE SERVICIO)
        vol_agua_protegida_m3 = (oferta_anual_m3 * 0.03) * horizonte_roi
        total_venta_agua = vol_agua_protegida_m3 * 0.45 
        
        # 💰 4. BENEFICIO ENERGÉTICO
        total_energia = (vol_agua_protegida_m3 * datos_nodo.get("factor_energia_kwh_m3", 0.65) * 0.08) if val_turbinado > 0 else 0.0

    with c_roi2:
        # ... (Cálculos de mitigación_pct, total_quimicos, etc. se mantienen) ...
        
        # 💰 5. MANTENIMIENTO MECÁNICO (ÁLABES)
        lodo_susp_m3 = st.session_state.get('eco_lodo_abrasivo_m3', 0.0)
        costo_alabes = 2.8 # USD/m3 de lodo que viaja por el túnel
        total_ahorro_turbinas = (lodo_susp_m3 * mitigacion_pct * costo_alabes) * eventos_ano * horizonte_roi
        
        # 💰 6. MANTENIMIENTO GEOTÉCNICO (TÚNELES)
        # El sedimento aumenta la rugosidad y el desgaste del revestimiento de los túneles
        costo_geotecnico_anual = 45000.0 # USD/año en inspecciones extra por sedimentos
        total_ahorro_tuneles = (costo_geotecnico_anual * mitigacion_pct) * horizonte_roi

        # =====================================================================
        # 💰 MOTOR FINANCIERO INTEGRAL Y CASO DE NEGOCIO (SbN)
        # =====================================================================
        # --- SUMATORIA DE BENEFICIOS ---
        beneficio_total = (total_quimicos + total_dragado_evitado + total_venta_agua + 
                           total_energia + total_ahorro_turbinas + total_ahorro_tuneles)
        
        roi_real = ((beneficio_total - inversion_total) / inversion_total) * 100 if inversion_total > 0 else 0
        
        # --- 📍 RECUADRO DE DIMENSIONAMIENTO TERRITORIAL UNIFICADO ---
        area_unificada = 17300.0
        pct_esf = (ha_proteger / area_unificada) * 100
        st.markdown(f"""
        <div style="background-color: #f1f8e9; border: 1px solid #c5e1a5; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
            <p style="margin: 0; color: #33691e; font-size: 0.95em;">
                📍 <b>Dimensionamiento Territorial:</b> Estás interviniendo el <b>{pct_esf:.1f}%</b> del área total de la cuenca ({area_unificada:,.0f} ha).<br>
                🚀 <b>Punto Óptimo:</b> Se requieren <b>3,594 ha</b> restauradas para alcanzar el 95% de la protección hidrológica máxima.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # --- 📊 RENDERIZADO DE RESULTADOS PRINCIPALES ---
        st.markdown("##### 📊 Caso de Negocio Integral (Inversión Basada en la Naturaleza)")
        c_r1, c_r2, c_r3 = st.columns(3)
        c_r1.metric("Inversión (CAPEX)", f"${inversion_total/1e6:,.2f} M USD")
        c_r2.metric("Beneficio Proyectado", f"${beneficio_total/1e6:,.2f} M USD", "Impacto Multinivel")
        c_r3.metric("R.O.I. Estratégico", f"{roi_real:,.0f}%", delta_color="normal" if roi_real > 0 else "inverse")
        
        # --- 🔍 DESGLOSE TÉCNICO DE BENEFICIOS ACUMULADOS ---
        with st.expander("🔍 Ver Desglose de Beneficios Acumulados", expanded=True):
            st.markdown(f"Distribución del valor generado en **{horizonte_roi} años**:")
            d_c1, d_c2, d_c3 = st.columns(3)
            
            with d_c1:
                st.write(f"• **Tratamiento:** ${total_quimicos/1e6:,.2f} M USD")
                st.write(f"• **Dragado Evitado:** ${total_dragado_evitado/1e6:,.2f} M USD")
            with d_c2:
                st.write(f"• **Agua y Energía:** ${(total_venta_agua + total_energia)/1e6:,.2f} M USD")
                st.write(f"• **Abrasión Álabes:** ${total_ahorro_turbinas/1e6:,.2f} M USD")
            with d_c3:
                st.write(f"• **Mantenimiento Túneles:** ${total_ahorro_tuneles/1e6:,.2f} M USD")
                st.write(f"**⚡ TOTAL:** :green[${beneficio_total/1e6:,.2f} M USD]")

        st.markdown("---")
        
        # --- 🛡️ RESILIENCIA DE INFRAESTRUCTURA ---
        st.markdown("##### 🛡️ Resiliencia de Activos Grises en los años de evaluacion del proyecto")
        c_r4, c_r5 = st.columns(2)
        lodo_fondo_evitado = (lodo_fondo_m3 * mitigacion_pct) * eventos_ano * horizonte_roi
        anos_salvados = lodo_fondo_evitado / 400000.0 
        
        c_r4.metric("Lodo Evitado (Total)", f"{lodo_evitado_m3_total:,.0f} m³", "Protección estructural")
        c_r5.metric("Vida Útil Salvada", f"+{anos_salvados:,.1f} Años", "Atraso del colapso funcional")
        
        # --- 📝 MENSAJE FINAL DE VIABILIDAD ---
        if roi_real > 0:
            ratio_retorno = beneficio_total / inversion_total
            ahorro_dragado_m = total_dragado_evitado / 1e6
            ahorro_agua_m = total_venta_agua / 1e6
            
            # Construcción de string pura para evitar errores de renderizado
            msg_exito = (
                f"✅ VIABILIDAD ESTRATÉGICA: La naturaleza devuelve {ratio_retorno:.1f} USD por cada dólar invertido. "
                f"El mayor peso financiero recae en la evitación de dragado ({ahorro_dragado_m:.1f} M USD) "
                f"y el agua protegida ({ahorro_agua_m:.1f} M USD)."
            )
            st.success(msg_exito)
        else:
            st.warning("⚠️ ANÁLISIS DE LARGO PLAZO: El retorno directo es bajo, pero el valor existencial del recurso trasciende la contabilidad.")

# =========================================================================
# 5. TRAYECTORIA CLIMÁTICA Y DEMOGRÁFICA (EXPLORADOR DE ESCENARIOS)
# =========================================================================
with st.expander(f"📈 Proyección Dinámica de Seguridad Hídrica {nodo_seleccionado} (2024 - 2050)", expanded=False):
    st.caption("Simulación a largo plazo que integra crecimiento poblacional, pérdida base por Cambio Climático y la variabilidad cíclica/extrema del fenómeno ENSO.")

    tab_resumen, tab_escenarios = st.tabs(["📊 Resumen Multivariado (Onda ENSO)", "🔬 Explorador de Escenarios (Cono de Incertidumbre)"])
    anios_proj = list(range(2024, 2051))

    with tab_resumen:
        col_t1, col_t2 = st.columns(2)
        with col_t1: activar_cc = st.toggle("🌡️ Incluir Cambio Climático", value=True, key="t1_cc")
        with col_t2: activar_enso = st.toggle("🌊 Incluir Variabilidad ENSO", value=True, key="t1_enso")

        # --- BUCLE DE PROYECCIÓN 2050 CON COHERENCIA ECOSISTÉMICA ---
        datos_proj = []
        
        # 1. Recuperamos el estado de salud actual del bosque (SIG)
        # Determina si el futuro es "verde" (filtración activa) o "gris" (carga cruda)
        eficiencia_futura = 0.60 if activar_sig else 0.0
        dr_difuso_futuro = 0.15 * (1.0 - eficiencia_futura)

        for a in anios_proj:
            delta_a = a - 2024
            f_dem = (1 + 0.015) ** delta_a
            f_cc_base = (1 - 0.005) ** delta_a if activar_cc else 1.0
            
            f_enso = 0.0
            estado_enso = "Neutro ⚖️"
            if activar_enso:
                f_enso = 0.25 * np.sin((2 * np.pi * delta_a) / 4.5) 
                estado_enso = "Niña 🌧️" if f_enso > 0.1 else "Niño ☀️" if f_enso < -0.1 else "Neutro ⚖️"
            
            f_cli_total = max(0.1, f_cc_base + f_enso) 
            
            # Oferta y Demanda Futura
            o_m3 = (q_oferta_m3s_base * f_cli_total) * 31536000
            c_m3 = (val_acueducto * f_dem) * 31536000
            
            # Neutralidad y Resiliencia Futura
            n = min(100.0, (volumen_repuesto_m3 / c_m3) * 100) if c_m3 > 0 else 100.0
            buff_sim = (capacidad_embalse_m3 + o_m3) / c_m3 if c_m3 > 0 else 5.0
            r = min(100.0, (buff_sim / 2.0) * 100)
            
            # Seguridad Hídrica (Estrés WEI+)
            wei_sim = c_m3 / o_m3 if o_m3 > 0 else 1.0
            e = max(0.0, min(100.0, 100.0 - (wei_sim / 0.40) * 60))
            
            # --- 🧪 CALIDAD PROYECTADA SENSIBLE AL BOSQUE ---
            # Carga proyectada: La población crece, y la filtración depende del SIG
            carga_pecuaria_f = ((pob_bov_local * 0.18) + (pob_por_local * 0.11)) * dr_difuso_futuro
            carga_humana_f = (pob_hum_local * f_dem * 0.018) * 0.80
            
            # Eventos cíclicos de tormenta (cada 3 años) para ver "dientes de sierra"
            carga_tormenta_f = (st.session_state.get('eco_fosforo_kg', 0.0) * 10 / 1000) if (activar_tormenta and a % 3 == 0) else 0.0
            
            # --- 🧪 CALIDAD PROYECTADA CON CALIBRACIÓN DINÁMICA ---
            carga_neta_f = carga_pecuaria_f + carga_humana_f + carga_tormenta_f
            remocion_f = (40 + (sist_saneamiento if activar_sig else 0)) * 2.2 
            carga_final_ton_f = max(0.5, carga_neta_f - remocion_f)
            
            # Dilución proyectada (Conectado a Concesiones Legales)
            q_mezcla_f = (sum(datos_nodo["afluentes_naturales"].values()) * f_cli_total) + (caudal_total_trasvase * 0.4)
            caudal_L_s_f = (q_mezcla_f if q_mezcla_f > 0 else 1.0) * 1000
            
            # 🚨 SINCRONIZACIÓN: Usamos umbral_dbo_colapso y factor_zona_critica de los sliders
            concentracion_f = (carga_final_ton_f * factor_zona_critica * 1e9) / (caudal_L_s_f * 31536000)
            cal = max(0.0, min(100.0, 100.0 - (concentracion_f / umbral_dbo_colapso * 100)))
            
            datos_proj.extend([
                {"Año": a, "Indicador": "Neutralidad", "Valor (%)": n, "Fase ENSO": estado_enso},
                {"Año": a, "Indicador": "Resiliencia", "Valor (%)": r, "Fase ENSO": estado_enso},
                {"Año": a, "Indicador": "Seguridad Hídrica", "Valor (%)": e, "Fase ENSO": estado_enso},
                {"Año": a, "Indicador": "Calidad", "Valor (%)": cal, "Fase ENSO": estado_enso}
            ])
            
        fig_line1 = px.line(pd.DataFrame(datos_proj), x="Año", y="Valor (%)", color="Indicador", hover_data=["Fase ENSO"],
                           color_discrete_map={"Neutralidad": "#2ecc71", "Resiliencia": "#3498db", "Seguridad Hídrica": "#e74c3c", "Calidad": "#9b59b6"})
        fig_line1.add_hrect(y0=0, y1=40, fillcolor="red", opacity=0.05, layer="below", annotation_text="  Zona Crítica (<40%)")
        fig_line1.add_hrect(y0=40, y1=70, fillcolor="orange", opacity=0.05, layer="below", annotation_text="  Zona Vulnerable (40-70%)")
        fig_line1.update_layout(height=400, hovermode="x unified", yaxis_range=[0, 105])
        st.plotly_chart(fig_line1, use_container_width=True)

    with tab_escenarios:
        st.markdown("Analiza la dispersión de futuros posibles para un solo indicador bajo intensidades climáticas sostenidas.")
        
        col_e1, col_e2 = st.columns([1, 2])
        with col_e1:
            ind_sel = st.selectbox("🎯 Seleccione el Indicador a Evaluar:", ["Estrés Hídrico", "Resiliencia", "Neutralidad", "Calidad"])
            activar_cc_esc = st.toggle("🌡️ Sumar Efecto Cambio Climático", value=True, key="t2_cc")
        
        with col_e2:
            diccionario_escenarios = {
                "Onda Dinámica (Ciclo de 4.5 años)": "onda",
                "Condición Neutra (Línea Base)": 0.0,
                "🟡 El Niño Moderado Constante (-15%)": -0.15,
                "🔴 El Niño Severo Constante (-35%)": -0.35,
                "🟢 La Niña Moderada Constante (+15%)": 0.15,
                "🔵 La Niña Fuerte Constante (+35%)": 0.35
            }
            
            curvas_sel = st.multiselect(
                "🌊 Curvas Climáticas a Superponer:", 
                list(diccionario_escenarios.keys()), 
                default=["Onda Dinámica (Ciclo de 4.5 años)", "Condición Neutra (Línea Base)", "🔴 El Niño Severo Constante (-35%)"]
            )

        datos_esc = []
        for a in anios_proj:
            delta_a = a - 2024
            f_dem = (1 + 0.015) ** delta_a
            f_cc_base = (1 - 0.005) ** delta_a if activar_cc_esc else 1.0
            
            for nombre_esc in curvas_sel:
                val_esc = diccionario_escenarios[nombre_esc]
                f_enso = 0.25 * np.sin((2 * np.pi * delta_a) / 4.5) if val_esc == "onda" else val_esc
                f_cli_total = max(0.1, f_cc_base + f_enso)
                
                o_m3 = (q_oferta_m3s_base * f_cli_total) * 31536000
                c_m3 = (val_acueducto * f_dem) * 31536000
                
                if ind_sel == "Neutralidad": val = min(100.0, (volumen_repuesto_m3 / c_m3) * 100) if c_m3 > 0 else 100.0
                elif ind_sel == "Resiliencia": 
                    buff_sim = (capacidad_embalse_m3 + o_m3) / c_m3 if c_m3 > 0 else 5.0
                    val = min(100.0, (buff_sim / 2.0) * 100)
                elif ind_sel == "Estrés Hídrico": 
                    wei_sim = c_m3 / o_m3 if o_m3 > 0 else 1.0
                    val = max(0.0, min(100.0, 100.0 - (wei_sim / 0.40) * 60))
                else: 
                    # Calidad Proyectada Escenario (Conectado a Concesiones Legales)
                    q_n_esc = sum(datos_nodo["afluentes_naturales"].values()) * f_cli_total
                    q_m_esc = q_n_esc + (caudal_total_trasvase * 0.4)
                    caudal_L_s_esc = (q_m_esc if q_m_esc > 0 else 1.0) * 1000
                    carga_f_esc = carga_mg_s_final * f_dem
                    val = max(0.0, min(100.0, 100.0 - ((carga_f_esc / caudal_L_s_esc) / 15.0 * 100)))
                    
                datos_esc.append({"Año": a, "Escenario Climático": nombre_esc, "Valor (%)": val})
                
        if datos_esc:
            df_esc = pd.DataFrame(datos_esc)
            color_map = {
                "Onda Dinámica (Ciclo de 4.5 años)": "#9b59b6",  
                "Condición Neutra (Línea Base)": "#34495e",        
                "🟡 El Niño Moderado Constante (-15%)": "#f1c40f",
                "🔴 El Niño Severo Constante (-35%)": "#e74c3c",
                "🟢 La Niña Moderada Constante (+15%)": "#2ecc71",
                "🔵 La Niña Fuerte Constante (+35%)": "#3498db"
            }
            fig_esc = px.line(df_esc, x="Año", y="Valor (%)", color="Escenario Climático", color_discrete_map=color_map)
            fig_esc.update_traces(line=dict(width=3)) 
            fig_esc.add_hrect(y0=0, y1=40, fillcolor="red", opacity=0.05, layer="below", annotation_text="Colapso / Crítico (<40%)")
            fig_esc.add_hrect(y0=40, y1=70, fillcolor="orange", opacity=0.05, layer="below", annotation_text="Vulnerable (40-70%)")
            fig_esc.update_layout(height=450, hovermode="x unified", yaxis_range=[0, 105], legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))
            st.plotly_chart(fig_esc, use_container_width=True)
            
# =========================================================================
# 6. METABOLISMO TERRITORIAL: PRIORIDAD DE CENSO REAL (SINCRONIZADO)
# =========================================================================
with st.expander(f"💧 Metabolismo Hídrico y Material: {nodo_seleccionado}", expanded=False):
    st.info("Cálculo integrado de extracción hídrica y cargas orgánicas. Los datos nacen sincronizados con el modelo de Huella Hídrica.")

    # 🔍 SINCRONIZACIÓN MAESTRA: Prioridad absoluta a los datos de la Nube (Supabase)
    if nodo_seleccionado == "La Fe": 
        f_pob, f_ext, f_bov, f_por, f_ave = 15000, 450000, 5000, 2000, 150000
    elif "Grande" in nodo_seleccionado: 
        f_pob, f_ext, f_bov, f_por, f_ave = 45000, 1200000, 85000, 45000, 800000
    else: 
        f_pob, f_ext, f_bov, f_por, f_ave = 20000, 0, 25000, 10000, 50000

    # 🚀 EXTRACCIÓN MAESTRA (Soldadura de Variables)
    pob_sinc = st.session_state.get(f'pob_asig_{nodo_seleccionado}', f_pob)
    bov_sinc = st.session_state.get('ica_bovinos_calc', f_bov)
    por_sinc = st.session_state.get('ica_porcinos_calc', f_por)
    ave_sinc = st.session_state.get('ica_aves_calc', f_ave)

    st.markdown("### 1. Inventario Poblacional Sincronizado (Censo Real + Proyección)")

    c_p1, c_p2, c_p3, c_p4, c_p5 = st.columns(5)
    
    # 🚨 MEJORA: Keys únicas para evitar colisiones de UI en Streamlit
    pob_residente = c_p1.number_input("🏘️ Pob. Residente:", value=int(pob_sinc), step=1000, key=f"ui_pob_res_{nodo_seleccionado}")
    pob_externa = c_p2.number_input("🏙️ Pob. Externa:", value=int(f_ext), step=50000, key=f"ui_pob_ext_{nodo_seleccionado}")
    cabezas_bovinas = c_p3.number_input("🐄 Bovinos:", value=int(bov_sinc), step=1000, key=f"ui_bov_{nodo_seleccionado}")
    cabezas_porcinas = c_p4.number_input("🐖 Porcinos:", value=int(por_sinc), step=1000, key=f"ui_por_{nodo_seleccionado}")
    cabezas_aves = c_p5.number_input("🐔 Aves:", value=int(ave_sinc), step=5000, key=f"ui_ave_{nodo_seleccionado}")
    
    # 🪄 Módulos de Consumo y Generación
    dot_hum, dot_bov, dot_por, kg_rs_hab, pct_organico = 150, 40, 15, 0.8, 55.0
    if st.toggle("⚙️ Mostrar y Ajustar Módulos de Consumo y Generación", key=f"tgl_modulos_{nodo_seleccionado}"):
        st.markdown("<div style='padding: 10px; background-color: #f8f9fa; border-radius: 5px;'>", unsafe_allow_html=True)
        c_d1, c_d2, c_d3 = st.columns(3)
        dot_hum = c_d1.number_input("Dotación Humana (L/d):", value=150, key=f"dot_hum_{nodo_seleccionado}")
        dot_bov = c_d2.number_input("Dotación Bovina (L/d):", value=40, key=f"dot_bov_{nodo_seleccionado}")
        dot_por = c_d3.number_input("Dotación Porcina (L/d):", value=15, key=f"dot_por_{nodo_seleccionado}")
        
        st.markdown("**Residuos Sólidos (Población Residente):**")
        c_rs1, c_rs2 = st.columns(2)
        kg_rs_hab = c_rs1.number_input("Generación RS (kg/hab/día):", value=0.8, step=0.1, key=f"rs_{nodo_seleccionado}")
        pct_organico = c_rs2.number_input("Fracción Orgánica (%):", value=55.0, step=5.0, key=f"rs_org_{nodo_seleccionado}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### 2. Balance Metabólico Integral")

    # =========================================================================
    # ⚖️ BALANCE METABÓLICO INTEGRAL 
    # =========================================================================
    pob_total_agua = pob_residente + pob_externa
    dem_hum_m3_dia = (pob_total_agua * dot_hum) / 1000
    dem_agro_m3_dia = ((cabezas_bovinas * dot_bov) + (cabezas_porcinas * dot_por) + (cabezas_aves * 0.3)) / 1000
    dem_total_m3_s = (dem_hum_m3_dia + dem_agro_m3_dia) / 86400

    carga_hum_ton = (pob_residente * 18.25) / 1000
    carga_bov_ton = (cabezas_bovinas * 292.0) / 1000
    carga_por_ton = (cabezas_porcinas * 91.25) / 1000
    carga_ave_ton = (cabezas_aves * 5.47) / 1000
    carga_total_ton = carga_hum_ton + carga_bov_ton + carga_por_ton + carga_ave_ton
    
    # 🚨 APRETÓN DE MANOS (HANDSHAKE) CON LA PÁGINA 09
    st.session_state['carga_dbo_total_ton'] = carga_total_ton
    st.session_state['demanda_total_m3s'] = dem_total_m3_s
    st.session_state['poblacion_servida'] = pob_total_agua # Vital para Pág 09

    rs_total_ton_ano = (pob_residente * kg_rs_hab * 365) / 1000
    rs_org_ton_ano = rs_total_ton_ano * (pct_organico / 100.0)
    rs_inorg_ton_ano = rs_total_ton_ano - rs_org_ton_ano

    c_res1, c_res2 = st.columns([1, 1.5])
    with c_res1:
        st.metric("💧 Extracción Continua (Agua)", f"{dem_total_m3_s:,.2f} m³/s")
        st.metric("☣️ Carga Orgánica al Río (DBO5)", f"{carga_total_ton:,.0f} Ton/año")
        st.metric("🗑️ Residuos Sólidos Totales", f"{rs_total_ton_ano:,.0f} Ton/año", f"{rs_org_ton_ano:,.0f} Ton Orgánicas")
        
        if st.button("🚀 Actualizar Dinámica del Embalse", use_container_width=True, key=f"btn_act_{nodo_seleccionado}"):
            st.success("✅ Extracción y metabolismo guardados en el Aleph.")

    with c_res2:
        df_plot = pd.DataFrame({
            "Categoria": ["Agua", "Agua", "Vertimientos", "Vertimientos", "Vertimientos", "Residuos", "Residuos"],
            "Subcategoria": ["Urbana", "Agropecuaria", "Urbana", "Bovinos/Porcinos", "Avicultura", "Orgánicos", "Inorgánicos"],
            "Valor": [
                (dem_hum_m3_dia*365)/1e6, (dem_agro_m3_dia*365)/1e6, 
                carga_hum_ton, (carga_bov_ton + carga_por_ton), carga_ave_ton,
                rs_org_ton_ano, rs_inorg_ton_ano
            ]
        })
        
        fig_sun = px.sunburst(df_plot, path=['Categoria', 'Subcategoria'], values='Valor', 
                              title="Distribución del Metabolismo Material",
                              color='Categoria', color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_sun.update_layout(height=400, margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig_sun, use_container_width=True)
        
# =========================================================================
# 8. HUELLA HÍDRICA TERRITORIAL (CONECTADA AL GEMELO DIGITAL)
# =========================================================================
with st.expander(f"👥 Huella Hídrica Territorial y Presión Demográfica ({nodo_seleccionado})", expanded=False):
    anio_censo = st.slider("📅 Horizonte de Simulación (Demanda y Presión):", min_value=2025, max_value=2050, value=2035, step=1, key=f"slider_anio_{nodo_seleccionado}")
    st.info("Cálculo de la demanda hídrica real integrando la población humana proyectada y la vocación pecuaria de la cuenca.")

    col_h1, col_h2 = st.columns([1, 1.5])

    with col_h1:
        st.subheader("1. Conexión a la Matriz Pecuaria")
        bovinos_tot, porcinos_tot, aves_tot = 0, 0, 0
        origen_datos_pec = "Valores Estáticos (Fallback)"

        mpios_cuenca = []
        if "Grande" in nodo_seleccionado: mpios_cuenca = ["BELMIRA", "DON MATIAS", "SAN PEDRO DE LOS MILAGROS", "ENTRERRIOS", "SANTA ROSA DE OSOS"]
        elif "Fe" in nodo_seleccionado: mpios_cuenca = ["EL RETIRO", "LA CEJA", "RIONEGRO"]

        if mpios_cuenca:
            for mpio in mpios_cuenca:
                datos_mpio = obtener_metabolismo_exacto(mpio, anio_censo)
                bovinos_tot += datos_mpio.get('bovinos', 0.0)
                porcinos_tot += datos_mpio.get('porcinos', 0.0)
                aves_tot += datos_mpio.get('aves', 0.0)
                
            if bovinos_tot > 0: origen_datos_pec = "Matriz Maestra (Proyectada)"
            
        if origen_datos_pec == "Valores Estáticos (Fallback)":
            bovinos_tot = 85000 if "Grande" in nodo_seleccionado else 5000
            porcinos_tot = 45000 if "Grande" in nodo_seleccionado else 2000
            aves_tot = 800000 if "Grande" in nodo_seleccionado else 150000

        if origen_datos_pec == "Matriz Maestra (Proyectada)": 
            st.success(f"🧠 **Sincronización Perfecta:** Metabolismo pecuario proyectado al {anio_censo}.")
        else: 
            st.warning(f"⚠️ **Memoria Vacía:** Entrena el Modelo Pecuario en la pág 06 para ver proyecciones dinámicas.")

        st.info(f"🐄 Bovinos: **{bovinos_tot:,.0f}** | 🐖 Porcinos: **{porcinos_tot:,.0f}** | 🐔 Aves: **{aves_tot:,.0f}**")

        # Guardamos silenciosamente en el estado para que otras páginas lo lean
        st.session_state['ica_bovinos_calc'] = bovinos_tot
        st.session_state['ica_porcinos_calc'] = porcinos_tot
        st.session_state['ica_aves_calc'] = aves_tot
        
    with col_h2:
        st.subheader(f"2. Demanda Total y Estrés Local ({nodo_seleccionado})")
        pob_global_memoria = 4000000 
        mpios_amva = ["MEDELLIN", "BELLO", "ITAGUI", "ENVIGADO", "SABANETA", "COPACABANA", "LA ESTRELLA", "GIRARDOTA", "CALDAS", "BARBOSA"]
        
        pob_calculada = 0
        for mpio in mpios_amva:
            datos_amva = obtener_metabolismo_exacto(mpio, anio_censo)
            if datos_amva['pob_total'] > 1000: pob_calculada += datos_amva['pob_total']
                
        if pob_calculada > 10000: pob_global_memoria = pob_calculada

        st.markdown("##### 🛡️ Seguridad Hídrica: Nivel de Dependencia")
        st.caption("Ajusta qué porcentaje de la población total del Valle de Aburrá es abastecida por este embalse.")
        
        dependencia_defecto = 58.0 if "Fe" in nodo_seleccionado else 33.0 if "Grande" in nodo_seleccionado else 9.0
        
        pct_dependencia = st.slider(f"% de Población dependiente:", 0.0, 100.0, dependencia_defecto, step=1.0, key=f"pct_dep_{nodo_seleccionado}")
        pob_real_dependiente = pob_global_memoria * (pct_dependencia / 100.0)

        # Aquí solo mostramos métricas basadas en lo que calculó la Sección 6 y la proyección
        st.markdown(f"### 📊 Demanda Metabólica Proyectada")
        dem_hum_futura = (pob_real_dependiente * dot_hum) / 1000
        dem_agro_futura = ((bovinos_tot * dot_bov) + (porcinos_tot * dot_por) + (aves_tot * 0.3)) / 1000
        dem_tot_futura_m3s = (dem_hum_futura + dem_agro_futura) / 86400  
        
        st.markdown("---")
        oferta_local_m3s = st.number_input("💧 Oferta Total del Sistema (m³/s) [Natural + Trasvases]:", value=5.7, step=0.1, key=f"oferta_{nodo_seleccionado}")
        estres_local = (dem_tot_futura_m3s / oferta_local_m3s) * 100 if oferta_local_m3s > 0 else 100
        
        estado_estres, alerta_color = ("Colapso", "inverse") if estres_local > 100 else ("Crítico", "inverse") if estres_local > 80 else ("Alto", "off") if estres_local > 40 else ("Estable", "normal")
        
        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        c_m1.metric("Demanda Humana", f"{dem_hum_futura:,.0f} m³/d")
        c_m2.metric("Demanda Agro", f"{dem_agro_futura:,.0f} m³/d")
        c_m3.metric("Extracción Proyectada", f"{dem_tot_futura_m3s:,.3f} m³/s")
        c_m4.metric(f"⚠️ Estrés ({estado_estres})", f"{estres_local:.1f}%", delta_color=alerta_color)

# ==============================================================================
# 🕸️ MAPA CONCEPTUAL: TOPOLOGÍA DEL METABOLISMO HÍDRICO
# ==============================================================================
with contenedor_sankey.container():
    with st.expander("🕸️ Mapa Conceptual: Topología del Metabolismo Hídrico", expanded=False):
        
        # 🔍 1. CAPTURA DE COBERTURAS REALES (Desde el motor de Pág 04 - Supabase)
        area_ref = datos_nodo.get("ha_conservadas_base", 3460.0)
        ha_bosque = st.session_state.get('aleph_ha_bosque', area_ref) 
        ha_agricola = st.session_state.get('aleph_ha_agricola', area_ref * 2.5)
        ha_pastos = st.session_state.get('aleph_ha_pastos', area_ref * 0.75)
        ha_urbana = st.session_state.get('aleph_area_urbana', area_ref * 0.75)
        
        area_total_cuenca = ha_bosque + ha_agricola + ha_pastos + ha_urbana
        
        # 🚨 HANDSHAKE DE ÁREA: Convertimos hectáreas a km2 para que la Pág 09 lo lea perfecto
        st.session_state['area_total_cuenca_val'] = area_total_cuenca
        st.session_state['aleph_area_km2'] = area_total_cuenca / 100.0

        # 🌪️ CAPTURA DEL IMPACTO FÍSICO (Sincronización con Módulo 6 de Pág 04)
        memoria_lodo_total = st.session_state.get('eco_lodo_total_m3', 0.0)
        lodo_colas = st.session_state.get('eco_lodo_colas_m3', 0.0)
        lodo_fondo = st.session_state.get('eco_lodo_fondo_m3', 0.0)
        lodo_suspension = st.session_state.get('eco_lodo_suspension_m3', 
                          st.session_state.get('eco_lodo_abrasivo_m3', 0.0))

        if memoria_lodo_total > 0:
            st.markdown(f"""
            <div style='background-color: rgba(231, 76, 60, 0.05); border-left: 5px solid #e74c3c; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                <span style='color: #c0392b; font-weight: bold;'>🚨 Evento de Ingeniería Detectado:</span> 
                La avenida torrencial de <b>{memoria_lodo_total:,.0f} m³</b> ha sido particionada físicamente según cotas de captación.
            </div>
            """, unsafe_allow_html=True)
            
            # 🚨 MEJORA: KEY ÚNICA en el toggle para evitar colisiones si cambias rápido de embalse
            activar_tormenta = st.toggle(
                "⛈️ Inyectar Trifurcación de Lodos en el Sistema", 
                value=st.session_state.get('activar_tormenta_sankey', False),
                key=f"tgl_lodos_{nodo_seleccionado}" 
            )
            st.session_state['activar_tormenta_sankey'] = activar_tormenta
            
            if activar_tormenta:
                col_i1, col_i2, col_i3 = st.columns(3)
                col_i1.metric("En Colas (Cota Alta)", f"{lodo_colas:,.0f} m³", "Depósito Periférico")
                
                # Cálculo de años robados basado en el lodo de fondo que colmata la torre
                anos_robados_hoy = lodo_fondo / 400000.0 if lodo_fondo > 0 else 0
                col_i2.metric("En Fondo (Torre)", f"{lodo_fondo:,.0f} m³", f"{anos_robados_hoy:+.2f} Años Vida Útil", delta_color="inverse")
                
                col_i3.metric("Suspendido (Abrasión)", f"{lodo_suspension:,.0f} m³", "Riesgo Máquinas", delta_color="inverse")
            st.markdown("---")

        # --- 2. LÓGICA DINÁMICA DEL DIAGRAMA ---
        labels = [f"Embalse {nodo_seleccionado}"] # Nodo 0
        source, target, value, color = [], [], [], []
        idx = 1

        # A. NODOS DE ORIGEN (USOS DEL SUELO)
        usos_nodos = [
            ("🌲 Bosques (Infiltración)", ha_bosque, "rgba(39, 174, 96, 0.6)"),
            ("🚜 Agrícola (Escorrentía)", ha_agricola, "rgba(241, 196, 15, 0.6)"),
            ("🐄 Pastos (Compactación)", ha_pastos, "rgba(230, 126, 34, 0.6)"),
            ("🏙️ Urbano (Impermeable)", ha_urbana, "rgba(149, 165, 166, 0.6)")
        ]
        
        indices_usos = {}
        for nom, area, col in usos_nodos:
            labels.append(nom); indices_usos[nom] = idx; idx += 1

        # B. CONEXIÓN SUELO -> RÍOS -> EMBALSE
        # Asume que afluentes_inputs y area_total_cuenca ya están calculados arriba
        for nombre_rio, q_rio in afluentes_inputs.items():
            labels.append(nombre_rio); current_rio_idx = idx
            for nom_u, area, col_u in usos_nodos:
                flujo_desde_suelo = (q_rio * (area / area_total_cuenca)) if area_total_cuenca > 0 else 0
                if flujo_desde_suelo > 0.01:
                    source.append(indices_usos[nom_u]); target.append(current_rio_idx); value.append(flujo_desde_suelo); color.append(col_u)
            source.append(current_rio_idx); target.append(0); value.append(q_rio); color.append("rgba(52, 152, 219, 0.5)")
            idx += 1

        # 🌪️ 3. TRIFURCACIÓN DE LODO (Si el toggle está activo)
        l_susp_s = 0.0
        if st.session_state.get('activar_tormenta_sankey', False) and memoria_lodo_total > 0:
            l_colas_s = lodo_colas / (12 * 3600)
            labels.append("Depósito en Colas"); source.append(idx); target.append(0); value.append(l_colas_s); color.append("rgba(210, 180, 140, 0.7)"); idx += 1
            l_fondo_s = lodo_fondo / (12 * 3600)
            labels.append("Lodo de Fondo"); source.append(idx); target.append(0); value.append(l_fondo_s); color.append("rgba(101, 67, 33, 0.9)"); idx += 1
            l_susp_s = lodo_suspension / (12 * 3600)
            if l_susp_s > 0:
                labels.append("Sedimento Abrasivo"); source.append(idx); target.append(0); value.append(l_susp_s); color.append("rgba(205, 133, 63, 0.8)"); idx += 1

        # C. TRASVASES LEGALES (Conectado a la Barra Lateral)
        if caudal_total_trasvase > 0:
            labels.append("Trasvases (Cornare/Corantioquia)")
            source.append(idx)
            target.append(0)
            value.append(caudal_total_trasvase)
            color.append("rgba(231, 76, 60, 0.4)")
            idx += 1

        # 4. SALIDAS Y VENA MARRÓN
        destinos = [
            ("Acueducto (Consumo)", val_acueducto, "rgba(52, 152, 219, 0.6)"),
            ("Generación Eléctrica", val_turbinado, "rgba(241, 196, 15, 0.7)"),
            ("Caudal Ecológico", val_ecologico, "rgba(149, 165, 166, 0.5)")
        ]
        salidas_con_infra = [d for d in destinos if d[1] > 0 and ("Acueducto" in d[0] or "Generación" in d[0])]
        reparto_lodo = l_susp_s / len(salidas_con_infra) if salidas_con_infra else 0

        for lab, val, col in destinos:
            if val > 0:
                target_node = idx; labels.append(lab)
                source.append(0); target.append(target_node); value.append(val); color.append(col)
                if reparto_lodo > 0 and ("Acueducto" in lab or "Generación" in lab):
                    source.append(0); target.append(target_node); value.append(reparto_lodo); color.append("rgba(205, 133, 63, 0.5)")
                idx += 1

        # =====================================================================
        # 🧪 INYECCIÓN DE CALIDAD MULTIVARIABLE (DBO, P, N)
        # =====================================================================
        calidad_usos = {
            "🌲 Bosques (Infiltración)": {"dbo": 0.5, "p": 0.01, "n": 0.2, "sed": 0.5},
            "🚜 Agrícola (Escorrentía)": {"dbo": 12.0, "p": 0.85, "n": 4.5, "sed": 450.0},
            "🐄 Pastos (Compactación)": {"dbo": 25.0, "p": 1.20, "n": 6.8, "sed": 180.0},
            "🏙️ Urbano (Impermeable)": {"dbo": 180.0, "p": 4.50, "n": 15.0, "sed": 85.0}
        }

        link_tooltips = []
        for i in range(len(source)):
            # Blindaje adicional: Aseguramos no salirnos del índice
            if source[i] < len(labels):
                nombre_origen = labels[source[i]]
                if nombre_origen in calidad_usos:
                    c = calidad_usos[nombre_origen]
                    txt = (f"<b>Calidad del Aporte:</b><br>"
                           f"• DBO: {c['dbo']} mg/L<br>"
                           f"• Fósforo: {c['p']} mg/L<br>"
                           f"• Nitrógeno: {c['n']} mg/L<br>"
                           f"• Sedimentos: {c['sed']} mg/L")
                    link_tooltips.append(txt)
                elif "Lodo" in nombre_origen or "Sedimento" in nombre_origen:
                    link_tooltips.append("<b>Carga Sólida Extrema</b><br>Evento Torrencial")
                else:
                    link_tooltips.append(f"Flujo de {nombre_origen}")
            else:
                link_tooltips.append("Flujo")

        # 5. RENDER FINAL (Tipografía Limpia)
        fig_sankey = go.Figure(data=[go.Sankey(
            arrangement="snap", # 🚀 Ayuda a organizar los nodos sin solaparse
            valueformat=".2f", valuesuffix=" m³/s",
            node=dict(pad=30, thickness=15, line=dict(color="none"), label=labels, color="#2C3E50"),
            link=dict(
                source=source, target=target, value=value, color=color,
                customdata=link_tooltips,
                hovertemplate='<b>De:</b> %{source.label}<br><b>A:</b> %{target.label}<br><b>Caudal:</b> %{value}%{valuesuffix}<br>%{customdata}<extra></extra>'
            )
        )])
        fig_sankey.update_layout(
            height=600, margin=dict(l=20, r=20, t=20, b=20),
            font=dict(size=12, family="Arial, sans-serif", color="#1f2937"), # 🚀 Letra más legible y moderna
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_sankey, use_container_width=True)
        
# =========================================================================
# 8. MATEMÁTICA Y CIENCIA
# =========================================================================
with st.expander("🔬 Ecuaciones de Dinámica de Sistemas (Embalses)"):
    st.markdown("La variación de almacenamiento en el tiempo se rige por la ecuación de continuidad:")
    st.markdown("$$\\frac{\\Delta S}{\\Delta t} = I_{nat} + \\sum I_{trasvases} - O_{urb} - O_{eco} - O_{energia} - E_{vap}$$")
    st.markdown("Si $\\frac{\\Delta S}{\\Delta t}$ es negativo de forma sostenida (ej. durante un fenómeno de El Niño donde $I_{nat} \\approx 0$), el volumen útil del embalse se agota, generando racionamiento en la metrópolis externa.")

# =========================================================================
# 🔄 ECONOMÍA CIRCULAR Y RUTA METABÓLICA (Valle de Aburrá -> Porce)
# =========================================================================
with contenedor_sankey.container():
    st.markdown("---")
    st.markdown("### ♻️ Metabolismo Regional: Ruta del Agua y Economía Circular")
    st.caption("Ruta física del agua desde la captación, su vertimiento tratado al Río Aburrá/Porce y el retorno de biosólidos como abono a las regiones productivas.")
    
    # 1. Población conectada al Censo Local Sincronizado
    pob_aburra = pob_total_agua if 'pob_total_agua' in locals() else 4200000
    
    # 2. Biosólidos (Ton/día)
    carga_biosolidos_ton_dia = (pob_aburra * 0.050) / 1000 
    
    # Factor visual para que la línea de biosólidos se vea en la escala de m³/s del gráfico
    val_consumo = val_acueducto if 'val_acueducto' in locals() else 5.0
    visual_lodo = max(0.2, val_consumo * 0.08) 
    
    # 3. Etiquetas de los Nodos (Estructuradas para legibilidad)
    nodos_circulares = [
        "1. Cuencas Cedentes", 
        f"2. {nodo_seleccionado}", 
        "3. Consumo (AMVA)", 
        "4. Red Alcantarillado", 
        "5. PTARs (S. Fernando / Aguas Claras)", 
        "6. Río Aburrá", 
        "7. Río Porce", 
        "8. Sist. Porce II, III, IV", 
        "🌱 Suelos Norte (R. Grande/Chico)", 
        "🌱 Suelos Oriente (Nare/Piedras)"
    ]
    
    # 4. Enlaces: (Origen, Destino, Valor Gráfico, Hover Text Real, Color)
    enlaces = [
        (0, 1, caudal_total_trasvase, f"{caudal_total_trasvase:.1f} m³/s", "rgba(52, 152, 219, 0.4)"),
        (1, 2, val_consumo, f"{val_consumo:.1f} m³/s", "rgba(41, 128, 185, 0.4)"),
        (2, 3, val_consumo * 0.80, f"{val_consumo * 0.80:.1f} m³/s (Agua Residual)", "rgba(149, 165, 166, 0.4)"),
        (3, 4, val_consumo * 0.75, f"{val_consumo * 0.75:.1f} m³/s (Afluente PTAR)", "rgba(127, 140, 141, 0.4)"),
        
        # Bifurcación PTAR -> RÍO ABURRÁ -> PORCE -> EMBALSES
        (4, 5, val_consumo * 0.75, f"{val_consumo * 0.75:.1f} m³/s (Agua Tratada)", "rgba(52, 152, 219, 0.5)"),
        (5, 6, val_consumo * 0.75, f"{val_consumo * 0.75:.1f} m³/s (Transición Cuenca)", "rgba(41, 128, 185, 0.6)"),
        (6, 7, val_consumo * 0.75, f"{val_consumo * 0.75:.1f} m³/s (Generación Eléctrica)", "rgba(241, 196, 15, 0.5)"),
        
        # Bifurcación PTAR -> BIOSÓLIDOS -> NORTE Y ORIENTE
        (4, 8, visual_lodo * 0.4, f"{carga_biosolidos_ton_dia * 0.4:.1f} Ton/día (Abono Orgánico)", "rgba(39, 174, 96, 0.6)"),
        (4, 9, visual_lodo * 0.6, f"{carga_biosolidos_ton_dia * 0.6:.1f} Ton/día (Abono Orgánico)", "rgba(39, 174, 96, 0.6)"),
    ]
    
    source = [e[0] for e in enlaces]
    target = [e[1] for e in enlaces]
    value = [e[2] for e in enlaces]
    hover = [e[3] for e in enlaces]
    color = [e[4] for e in enlaces]
    
    colores_nodos = ["#2E86C1", "#1F618D", "#D35400", "#7B7D7D", "#5D6D7E", "#3498DB", "#2980B9", "#F39C12", "#27AE60", "#229954"]

    fig_sankey_circ = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node = dict(pad=30, thickness=15, line=dict(color="none"), label=nodos_circulares, color=colores_nodos),
        link = dict(
            source=source, target=target, value=value, color=color, customdata=hover,
            hovertemplate='<b>%{source.label}</b> ➔ <b>%{target.label}</b><br>Flujo: %{customdata}<extra></extra>'
        )
    )])
    
    fig_sankey_circ.update_layout(
        height=550, margin=dict(l=20, r=20, t=30, b=20),
        font=dict(size=12, family="Arial, sans-serif", color="#1f2937"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
    
    st.plotly_chart(fig_sankey_circ, use_container_width=True)
