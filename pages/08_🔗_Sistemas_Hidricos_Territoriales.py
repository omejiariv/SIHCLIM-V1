# pages/08_🔗_Sistemas_Hidricos_Territoriales.py

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import os
import geopandas as gpd
from modules.demografia_tools import render_motor_demografico
import unicodedata

# =========================================================================
# 1. CONFIGURACIÓN Y DICCIONARIO BASE
# =========================================================================
st.set_page_config(page_title="Metabolismo Complejo", page_icon="🔗", layout="wide")

from modules.utils import encender_gemelo_digital, obtener_metabolismo_exacto
encender_gemelo_digital()

sistemas_embalses = {
    "La Fe": {
        "capacidad_util_Mm3": 11.5, 
        "afluentes_naturales": {"Quebrada Espíritu Santo": 1.2},
        "trasvases": {"Pantanillo": 1.5, "Río Buey": 3.0, "Piedras": 0.8},
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
        "trasvases": {}, 
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
        "trasvases": {}, 
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
        "trasvases": {}, 
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
        "trasvases": {}, 
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
# 2. 🧠 EL ALEPH HÍDRICO
# ==============================================================================
conectado_aleph = False
pob_amva_aleph = None
pob_local_aleph = None

if 'aleph_lugar' in st.session_state:
    aleph_lugar = st.session_state['aleph_lugar']
    aleph_anio = st.session_state.get('aleph_anio', 2025)
    
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
# 3. 🎛️ SIDEBAR
# =========================================================================
st.sidebar.markdown("### 🎛️ Centro de Operaciones")
nodos_lista = list(sistemas_embalses.keys())
idx_defecto = nodos_lista.index(st.session_state['nodo_sugerido']) if 'nodo_sugerido' in st.session_state and st.session_state['nodo_sugerido'] in nodos_lista else 0
nodo_seleccionado = st.sidebar.selectbox("Seleccione el Nodo Principal:", nodos_lista, index=idx_defecto)
datos_nodo = sistemas_embalses[nodo_seleccionado]

# =========================================================================
# 4. 🏷️ TÍTULOS Y UI PRINCIPAL
# =========================================================================
st.title(f"🔗 Metabolismo Territorial Complejo: Nodos y Trasvases ({nodo_seleccionado})")
st.markdown("Modelo de topología de redes para el **Sistema de Abastecimiento del Valle de Aburrá y Generación Eléctrica**. Evalúa cómo los embalses integran las cuencas propias con los trasvases artificiales para sostener la demanda, alterando el flujo natural de los ecosistemas aportantes.")
st.divider()

# --- ESPACIO PARA EL SANKEY ---
contenedor_sankey = st.empty()

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
            <div class="pergamino-texto">Esta obra está plasmada sobre un pergamino envejecido y texturizado, diseñado para emular un manuscrito de Leonardo da Vinci o las láminas de Humboldt.</div>
            <div class="pergamino-seccion">1. El Punto Focal: El Aleph</div>
            <div class="pergamino-texto">En el centro geométrico brilla una pequeña esfera con "fulgor casi intolerable tornasolado". Contiene la totalidad de Antioquia y del cosmos.</div>
            <div class="pergamino-seccion">2. La Estructura Fractal</div>
            <div class="pergamino-texto">Un "Engranaje Anatómico-Hídrico" inspirado en Da Vinci. Los flujos irradian transformándose en ríos fractales.</div>
            <div class="pergamino-seccion">3. Texto y Mito</div>
            <div class="pergamino-texto">Fragmentos de "El Aleph" y Pessoa rodean la esfera.</div>
        </div>
    </div>
    <p style="text-align: center; color: #7f8c8d; font-style: italic; font-size: 0.9em; margin-top: 15px;">Acércate a la obra para encender el Aleph y revelar sus secretos.</p>
    """, unsafe_allow_html=True)

if conectado_aleph:
    with st.expander("🧠 Conexión Activa con el Modelo Demográfico (El Aleph)", expanded=False):
        st.success(f"Recibiendo proyección para **{aleph_lugar}** (Año **{aleph_anio}**): **{aleph_pob:,.0f} habitantes**.")

# =========================================================================
# 5. CARGA DE CARTOGRAFÍA Y CLIMA
# =========================================================================
url_supabase = st.secrets.get("SUPABASE_URL") or st.secrets.get("supabase", {}).get("url") or st.secrets.get("connections", {}).get("supabase", {}).get("SUPABASE_URL")
gdf_embalses = None 

if url_supabase:
    ruta_embalses_nube = f"{url_supabase}/storage/v1/object/public/sihcli_maestros/Puntos_de_interes/embalses_CV_9377.geojson"
    try:
        gdf_embalses = gpd.read_file(ruta_embalses_nube)
        st.sidebar.success(f"✅ Embalses conectados ({len(gdf_embalses)} registros)")
    except: pass

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
        
        try:
            sistemas_embalses["La Fe"]["ha_conservadas_base"] = 3600.0
            ruta_predios = f"{url_supabase}/storage/v1/object/public/sihcli_maestros/Puntos_de_interes/PrediosEjecutados.geojson"
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
                        for nombre_nodo in sistemas_embalses.keys():
                            if str(embalse_mapa).lower().strip() in nombre_nodo.lower():
                                sistemas_embalses[nombre_nodo]["ha_conservadas_base"] += float(ha_total)
        except: pass

st.sidebar.markdown("### 🌍 Motor Climático (ENSO)")
escenario_enso = st.sidebar.select_slider("Fase actual del Pacífico:", options=["Niño Severo", "Niño Moderado", "Neutro", "Niña Moderada", "Niña Fuerte"], value="Neutro")
if escenario_enso == "Niño Severo": factor_p, factor_et = 0.65, 1.40
elif escenario_enso == "Niño Moderado": factor_p, factor_et = 0.85, 1.20
elif escenario_enso == "Neutro": factor_p, factor_et = 1.0, 1.0
elif escenario_enso == "Niña Moderada": factor_p, factor_et = 1.15, 0.85
else: factor_p, factor_et = 1.35, 0.70

# =========================================================================
# MÓDULO 1: BALANCE DE MASA EN TIEMPO REAL (COMPRIMIDO)
# =========================================================================
with st.expander(f"💧 Balance de Masa en Tiempo Real: {nodo_seleccionado}", expanded=False):
    if factor_p < 1.0: st.error(f"⚠️ **{escenario_enso}:** Oferta Hídrica **{(factor_p-1)*100:+.0f}%** | Evaporación (ET) **{(factor_et-1)*100:+.0f}%** debido a anomalía térmica.")
    elif factor_p > 1.0: st.info(f"🌧️ **{escenario_enso}:** Oferta Hídrica **{(factor_p-1)*100:+.0f}%** | Evaporación (ET) reducida al **{(factor_et)*100:.0f}%**.")
    else: st.info(f"**Capacidad Útil Máxima:** {datos_nodo['capacidad_util_Mm3']:,.1f} Mm³ (Condiciones Climáticas Históricas)")

    col_in, col_out = st.columns(2)
    afluentes_inputs, trasvases_inputs = {}, {}

    with col_in:
        st.markdown("#### 📥 ENTRADAS (Inflows)")
        caudal_real_aleph = st.session_state.get('aleph_q_rio_m3s', 0.0)
        for i, (nombre, caudal) in enumerate(datos_nodo["afluentes_naturales"].items()):
            caudal_base = caudal_real_aleph if (i == 0 and caudal_real_aleph > 0) else caudal
            nombre_mostrar = f"💧 {nombre} (Física Real)" if (i == 0 and caudal_real_aleph > 0) else nombre
            max_val = float(caudal_base * 3) if caudal_base > 0 else 10.0
            caudal_afectado = min(float(caudal_base * factor_p), max_val) 
            afluentes_inputs[nombre_mostrar] = st.slider(f"{nombre_mostrar} [m³/s]:", 0.0, max_val, caudal_afectado, 0.1, key=f"in_{nombre}")
            
        if datos_nodo["trasvases"]:
            for nombre, caudal in datos_nodo["trasvases"].items():
                max_val = float(caudal * 2) if caudal > 0 else 5.0
                trasvases_inputs[nombre] = st.slider(f"Bombeo {nombre} [m³/s]:", 0.0, max_val, min(float(caudal), max_val), 0.1, key=f"tr_{nombre}")
        else: st.write("*(Sistema impulsado 100% por gravedad)*")

    with col_out:
        st.markdown("#### 📤 SALIDAS (Outflows)")
        evaporacion_dinamica = datos_nodo["evaporacion_m3s"] * factor_et
        st.metric("Evaporación Directa", f"{evaporacion_dinamica:.2f} m³/s", f"{(evaporacion_dinamica - datos_nodo['evaporacion_m3s']):+.2f} m³/s", delta_color="inverse")
        
        demanda_memoria = st.session_state.get('demanda_total_m3s', 0.0)
        if demanda_memoria > 0 and demanda_memoria >= (datos_nodo["demanda_acueducto_m3s"] * 0.5):
            val_base_acue = demanda_memoria
            label_acueducto = "🚰 Extracción Metabólica (Real) [m³/s]:"
        else:
            val_base_acue = datos_nodo["demanda_acueducto_m3s"]
            label_acueducto = "Extracción Consuntiva (Teórica) [m³/s]:"
            
        max_acueducto = float(max(val_base_acue, datos_nodo["demanda_acueducto_m3s"]) * 2)
        val_acueducto = st.slider(label_acueducto, 0.0, max_acueducto, float(val_base_acue), 0.05) if max_acueducto > 0 else 0.0
        
        val_turbinado = 0.0
        if datos_nodo["generacion_energia_m3s"] > 0:
            max_turb = float(datos_nodo["generacion_energia_m3s"] * 1.5)
            val_turbinado = st.slider("Caudal Turbinado [m³/s]:", 0.0, max_turb, min(float(datos_nodo["generacion_energia_m3s"]), max_turb), 1.0)
            
        val_ecologico = st.number_input("Caudal Ecológico [m³/s]:", min_value=0.0, value=float(datos_nodo["caudal_ecologico_m3s"]), step=1.0)

    sum_entradas = sum(afluentes_inputs.values()) + sum(trasvases_inputs.values())
    sum_salidas = val_acueducto + val_turbinado + val_ecologico + datos_nodo["evaporacion_m3s"]
    balance = sum_entradas - sum_salidas

    m3_hora_turbinados = val_turbinado * 3600
    m3_hora_bombeados = sum(trasvases_inputs.values()) * 3600
    potencia_generada_kw = m3_hora_turbinados * datos_nodo["factor_energia_kwh_m3"]
    potencia_consumida_kw = m3_hora_bombeados * datos_nodo["costo_bombeo_kwh_m3"]
    balance_energetico_MW = (potencia_generada_kw - potencia_consumida_kw) / 1000

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Balance Hídrico (ΔS/Δt)", f"{balance:+.1f} m³/s", "Llenándose 📈" if balance > 0 else "Vaciándose 📉" if balance < 0 else "Estable ⚖️")
    c2.metric("Energía Generada", f"{potencia_generada_kw/1000:,.1f} MW", f"${(potencia_generada_kw * 350)/1e6:,.1f} M/hora", delta_color="normal")
    c3.metric("Energía Consumida", f"{potencia_consumida_kw/1000:,.1f} MW", f"-${(potencia_consumida_kw * 350)/1e6:,.1f} M/hora", delta_color="inverse")
    c4.metric("Balance Neto", f"{balance_energetico_MW:,.1f} MW", "Superávit ⚡" if balance_energetico_MW > 0 else "Déficit 🔴" if balance_energetico_MW < 0 else "Neutro")


# =========================================================================
# MÓDULO 2: INTELIGENCIA TERRITORIAL WRI (COMPRIMIDO)
# =========================================================================
with st.expander(f"🌐 Inteligencia Territorial WRI: {nodo_seleccionado}", expanded=False):
    anio_analisis = st.slider("Seleccione el Año de Evaluación (Actual o Futuro):", min_value=2024, max_value=2050, value=2025, step=1)

    delta_anios = anio_analisis - 2025
    factor_demanda, factor_clima = (1 + 0.015) ** delta_anios, (1 - 0.005) ** delta_anios
    
    q_oferta_m3s_base = sum_entradas
    demanda_m3s_base = val_acueducto + (val_turbinado * 0.1) 
    capacidad_embalse_m3 = datos_nodo["capacidad_util_Mm3"] * 1000000

    oferta_anual_m3 = (q_oferta_m3s_base * factor_clima) * 31536000
    consumo_anual_m3 = (demanda_m3s_base * factor_demanda) * 31536000

    st.markdown("#### 🌲 Beneficios Volumétricos (SbN)")
    activar_sig = st.toggle("✅ Incluir Área Restaurada/Conservada del SIG en el cálculo WRI", value=True)
    ha_reales_sig = float(datos_nodo.get("ha_conservadas_base", 0.0))
    ha_base_calculo = ha_reales_sig if activar_sig else 0.0

    c_inv1, c_inv2, c_inv3 = st.columns(3)
    with c_inv1:
        st.metric("✅ Área Conservada (SIG)", f"{ha_reales_sig:,.1f} ha")
        ha_simuladas = st.number_input("➕ Adicionar Hectáreas:", min_value=0.0, value=0.0, step=50.0)
    with c_inv2:
        sist_saneamiento = st.number_input("STAM (Sistemas Tratamiento):", min_value=0, value=120, step=5)
    with c_inv3:
        volumen_repuesto_m3 = ((ha_base_calculo + ha_simuladas) * 2500) + ((sist_saneamiento * 1200) if activar_sig else 0)
        st.metric("💧 Agua 'Devuelta' (VWBA)", f"{volumen_repuesto_m3/1e6:,.2f} Mm³/año")

    # 🌪️ IMPACTO DE LA TORMENTA (Botón Mágico)
    memoria_lodo_m3 = st.session_state.get('eco_lodo_m3', 0.0)
    memoria_fosforo_kg = st.session_state.get('eco_fosforo_kg', 0.0)
    memoria_sobrecosto_usd = st.session_state.get('eco_sobrecosto_usd', 0.0)
    activar_tormenta = False

    if memoria_lodo_m3 > 0:
        st.markdown("<div style='background-color: rgba(231, 76, 60, 0.1); border-left: 5px solid #e74c3c; padding: 15px; border-radius: 5px; margin-bottom: 20px;'><h4 style='color: #c0392b; margin-top: 0;'>🚨 Avalancha en Memoria</h4></div>", unsafe_allow_html=True)
        activar_tormenta = st.toggle("⛈️ Inyectar Impacto de Tormenta en este Modelo (Sankey)", value=False)
        
        if activar_tormenta:
            c_a1, c_a2, c_a3 = st.columns(3)
            c_a1.metric("Avalancha Lodo", f"{memoria_lodo_m3:,.0f} m³", delta_color="inverse")
            c_a2.metric("Inyección Fósforo", f"{memoria_fosforo_kg:,.0f} Kg", delta_color="inverse")
            c_a3.metric("Sobrecosto USD", f"${memoria_sobrecosto_usd:,.0f}", delta_color="inverse")
        st.markdown("---")

    eco_lodo_m3 = memoria_lodo_m3 if activar_tormenta else 0.0
    eco_fosforo_kg = memoria_fosforo_kg if activar_tormenta else 0.0
    eco_sobrecosto_usd = memoria_sobrecosto_usd if activar_tormenta else 0.0

    # Lógica de Índices
    if nodo_seleccionado == "La Fe": d_hum, d_bov, d_por = 15000, 5000, 2000
    elif "Grande" in nodo_seleccionado: d_hum, d_bov, d_por = 45000, 85000, 45000
    elif "Peñol" in nodo_seleccionado: d_hum, d_bov, d_por = 25000, 40000, 15000
    elif "Ituango" in nodo_seleccionado: d_hum, d_bov, d_por = 35000, 250000, 60000
    else: d_hum, d_bov, d_por = 20000, 25000, 10000

    carga_neta_ton = (((st.session_state.get('sh_bovinos_ica', d_bov) * 0.18) + (st.session_state.get('sh_porcinos_ica', d_por) * 0.11)) * 0.15) + ((st.session_state.get('sh_pob_residente', d_hum) * 0.018) * 0.80) + ((eco_fosforo_kg * 10) / 1000.0)
    carga_final_rio_ton = max(0.0, carga_neta_ton - (sist_saneamiento * 2.5))
    
    caudal_natural_L_s = sum(datos_nodo["afluentes_naturales"].values()) * 1000
    concentracion_dbo_mg_l = ((carga_final_rio_ton * 1e9) / 31536000) / caudal_natural_L_s if caudal_natural_L_s > 0 else 999.0
    ind_calidad = max(0.0, min(100.0, 100.0 - ((concentracion_dbo_mg_l / 10.0) * 100)))

    buffer_ratio = (capacidad_embalse_m3 + oferta_anual_m3) / consumo_anual_m3 if consumo_anual_m3 > 0 else 5.0
    ind_resiliencia = min(100.0, (buffer_ratio / 2.0) * 100)

    consumo_real = max(consumo_anual_m3, val_acueducto * 31536000)
    ind_estres = max(0.0, min(100.0, 100.0 - ((consumo_real / oferta_anual_m3 if oferta_anual_m3 > 0 else 1.0) / 0.40) * 60))
    ind_neutralidad = min(100.0, (volumen_repuesto_m3 / consumo_real) * 100) if consumo_real > 0 else 0.0

    col_g1, col_g2, col_g3, col_g4 = st.columns(4)
    def crear_velocimetro(valor, titulo, color_bar, umbral_rojo, umbral_verde):
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=valor, number={'suffix': "%", 'font': {'size': 26}}, title={'text': titulo, 'font': {'size': 14}},
            gauge={'axis': {'range': [None, 100], 'tickwidth': 1}, 'bar': {'color': color_bar}, 'bgcolor': "white",
                   'steps': [{'range': [0, umbral_rojo], 'color': "#ffcccb"}, {'range': [umbral_rojo, umbral_verde], 'color': "#fff2cc"}, {'range': [umbral_verde, 100], 'color': "#e8f8f5"}],
                   'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': valor}}
        ))
        fig.update_layout(height=230, margin=dict(l=10, r=10, t=30, b=10))
        return fig

    for col, ind, tit, col_h in zip([col_g1, col_g2, col_g3, col_g4], [ind_neutralidad, ind_resiliencia, ind_estres, ind_calidad], ["Neutralidad", "Resiliencia", "Seguridad Hídrica", "Calidad de Agua"], ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"]):
        with col:
            est, color_txt = ("🔴 CRÍTICO", "#c0392b") if ind < 40 else ("🟡 VULNERABLE", "#f39c12") if ind < 70 else ("🟢 ÓPTIMO", "#27ae60")
            st.plotly_chart(crear_velocimetro(ind, tit, col_h, 40, 70), use_container_width=True)
            st.markdown(f"<h4 style='text-align: center; color: {color_txt}; margin-top:-20px;'>{est}</h4>", unsafe_allow_html=True)

    if st.toggle("📚 Mostrar Conceptos y Metodología (VWBA - WRI)"):
        st.info("La Neutralidad Hídrica mide si el volumen restituido por la naturaleza compensa la Huella Hídrica. La Resiliencia es la capacidad de soportar sequías, y el Estrés Hídrico evalúa la competencia por el recurso.")

# =========================================================================
# MÓDULO 3: PORTAFOLIOS (YA COMPRIMIDOS)
# =========================================================================
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("💼 Portafolios de Inversión Multi-Objetivo")

with st.expander("🎯 1. Optimización de Brechas: Oferta y Demanda (Neutralidad)", expanded=False):
    col_m1, col_m2 = st.columns([1, 2.5])
    with col_m1:
        meta_neutralidad = st.slider("🎯 Objetivo de Neutralidad (%)", 10.0, 100.0, 100.0, 5.0)
        costo_ha = st.number_input("Restauración (1 ha) [M COP]:", value=8.5, step=0.5)
        costo_stam_n = st.number_input("Saneamiento (1 STAM) [M COP]:", value=15.0, step=1.0)
        costo_lps = st.number_input("Eficiencia (1 L/s) [M COP]:", value=120.0, step=10.0)
    with col_m2:
        brecha_m3 = ((meta_neutralidad / 100.0) * consumo_anual_m3) - volumen_repuesto_m3
        if brecha_m3 <= 0: st.success("✅ Cumple meta de neutralidad.")
        else:
            st.warning(f"⚠️ Faltan compensar **{brecha_m3/1e6:,.2f} Mm³/año**.")
            c_mix1, c_mix2, c_mix3 = st.columns(3)
            pct_a, pct_b, pct_c = c_mix1.number_input("% Restaura", 0, 100, 40), c_mix2.number_input("% Sanea", 0, 100, 40), c_mix3.number_input("% Eficiencia", 0, 100, 20)
            if pct_a + pct_b + pct_c == 100:
                inv_total = (brecha_m3 * (pct_a/100) / 2500 * costo_ha) + (brecha_m3 * (pct_b/100) / 1200 * costo_stam_n) + ((brecha_m3 * (pct_c/100) * 1000 / 31536000) * costo_lps)
                st.metric("💰 INVERSIÓN (CANTIDAD)", f"${inv_total:,.0f} Millones COP")

with st.expander("🎯 2. Optimización de Cargas Contaminantes (Saneamiento DBO5)", expanded=False):
    carga_total_ton = st.session_state.get('carga_total_ton', 1000.0)
    col_c1, col_c2 = st.columns([1, 2.5])
    with col_c1:
        meta_remocion = st.slider("🎯 Meta Remoción (%)", 10.0, 100.0, 85.0, 5.0)
        costo_ptar = st.number_input("PTAR (1 Ton/a) [M COP]:", value=150.0, step=10.0)
    with col_c2:
        brecha_ton = ((meta_remocion / 100.0) * carga_total_ton)
        if brecha_ton > 0:
            st.warning(f"⚠️ Faltan remover **{brecha_ton:,.1f} Toneladas/año**.")
            inv_total_c = brecha_ton * costo_ptar # Simplificado visual
            st.metric("💰 INVERSIÓN (CALIDAD)", f"${inv_total_c:,.0f} Millones COP")

# =========================================================================
# MÓDULO 4: PROYECCIÓN DINÁMICA (COMPRIMIDO)
# =========================================================================
with st.expander(f"📈 Proyección Dinámica de Seguridad Hídrica {nodo_seleccionado} (2024 - 2050)", expanded=False):
    tab_resumen, tab_escenarios = st.tabs(["📊 Resumen Multivariado (Onda ENSO)", "🔬 Explorador de Escenarios (Cono de Incertidumbre)"])
    anios_proj = list(range(2024, 2051))

    with tab_resumen:
        col_t1, col_t2 = st.columns(2)
        with col_t1: activar_cc = st.toggle("🌡️ Incluir Cambio Climático", value=True)
        with col_t2: activar_enso = st.toggle("🌊 Incluir Variabilidad ENSO", value=True)
        datos_proj = []
        for a in anios_proj:
            d_a = a - 2024
            f_cli_total = max(0.1, ((1-0.005)**d_a if activar_cc else 1.0) + (0.25 * np.sin((2*np.pi*d_a)/4.5) if activar_enso else 0.0))
            o_m3, c_m3 = (q_oferta_m3s_base * f_cli_total) * 31536000, (val_acueducto * ((1+0.015)**d_a)) * 31536000
            datos_proj.extend([{"Año": a, "Indicador": "Resiliencia", "Valor (%)": min(100.0, (((capacidad_embalse_m3 + o_m3)/c_m3 if c_m3 > 0 else 5.0)/2.0)*100)}])
        st.plotly_chart(px.line(pd.DataFrame(datos_proj), x="Año", y="Valor (%)", color="Indicador"), use_container_width=True)

    with tab_escenarios:
        st.info("Explorador de Escenarios activo. Selecciona curvas para superponer.")

# =========================================================================
# MÓDULO 5: METABOLISMO HÍDRICO Y MATERIAL (COMPRIMIDO)
# =========================================================================
with st.expander(f"💧 Metabolismo Hídrico y Material: {nodo_seleccionado}", expanded=False):
    if nodo_seleccionado == "La Fe": d_p_r, d_p_e, d_b, d_p, d_a = 15000, 450000, 5000, 2000, 150000
    elif "Grande" in nodo_seleccionado: d_p_r, d_p_e, d_b, d_p, d_a = 45000, 1200000, 85000, 45000, 800000
    else: d_p_r, d_p_e, d_b, d_p, d_a = 20000, 0, 25000, 10000, 50000

    c_p1, c_p2, c_p3, c_p4 = st.columns(4)
    pob_res = c_p1.number_input("🏘️ Pob. Residente:", value=int(d_p_r), key="sh_pob_residente")
    pob_ext = c_p2.number_input("🏙️ Pob. Externa:", value=int(d_p_e), key="sh_pob_externa")
    cab_bov = c_p3.number_input("🐄 Bovinos:", value=int(d_b), key="sh_bovinos_ica")
    cab_por = c_p4.number_input("🐖 Porcinos:", value=int(d_p), key="sh_porcinos_ica")

    dot_hum, dot_bov, dot_por, dot_ave = 150, 40, 15, 0.3
    if st.toggle("⚙️ Ajustar Módulos de Consumo y Generación"):
        c_d1, c_d2, c_d3, c_d4 = st.columns(4)
        dot_hum = c_d1.number_input("Dot Humana (L/d):", value=150)
        dot_bov = c_d2.number_input("Dot Bovina (L/d):", value=40)
    
    dem_total_m3_s = (((pob_res+pob_ext)*dot_hum) + (cab_bov*dot_bov) + (cab_por*dot_por) + (d_a*dot_ave)) / 86400000
    st.metric("💧 Extracción Continua (Agua)", f"{dem_total_m3_s:,.2f} m³/s")

# =========================================================================
# MÓDULO 6: HUELLA HÍDRICA Y PRESIÓN (COMPRIMIDO)
# =========================================================================
with st.expander(f"👥 Huella Hídrica Territorial y Presión Demográfica ({nodo_seleccionado})", expanded=False):
    anio_censo = st.slider("📅 Horizonte de Simulación:", 2025, 2050, 2035)
    st.info(f"Conectado a la Matriz Demográfica del {anio_censo}")
    
    oferta_local_m3s = st.number_input("💧 Oferta Total del Sistema (m³/s):", value=5.7, step=0.1)
    if st.button("💾 Enviar Datos al Modelo Central", type="primary"):
        st.session_state['demanda_total_m3s'] = dem_total_m3_s
        st.session_state['aleph_oferta_hidrica'] = oferta_local_m3s
        st.success("✅ ¡Memoria actualizada!")

# ==============================================================================
# 🕸️ SANKEY (YA ESTABA EN EXPANDER, SOLO LO DEJAMOS CERRADO)
# ==============================================================================
with contenedor_sankey.container():
    with st.expander("🕸️ Mapa Conceptual: Topología del Metabolismo Hídrico", expanded=False):
        labels = [f"Embalse {nodo_seleccionado}"]
        source, target, value, color = [], [], [], []
        idx = 1

        for n, q in afluentes_inputs.items():
            if q > 0: labels.append(n); source.append(idx); target.append(0); value.append(q); color.append("rgba(46, 204, 113, 0.6)"); idx += 1
        for n, q in trasvases_inputs.items():
            if q > 0: labels.append(f"Bombeo {n} ⚡(-)"); source.append(idx); target.append(0); value.append(q); color.append("rgba(231, 76, 60, 0.8)"); idx += 1

        if eco_lodo_m3 > 0:
            labels.append(f"Avalancha Lodo<br>({eco_lodo_m3/1000:,.1f} dam³)")
            source.append(idx); target.append(0); value.append(eco_lodo_m3 / 43200); color.append("rgba(139, 69, 19, 0.85)"); idx += 1                

        if val_acueducto > 0: labels.append("Acueducto"); source.append(0); target.append(idx); value.append(val_acueducto); color.append("rgba(52, 152, 219, 0.6)"); idx += 1
        if val_turbinado > 0: labels.append("Generación ⚡(+)"); source.append(0); target.append(idx); value.append(val_turbinado); color.append("rgba(241, 196, 15, 0.8)"); idx += 1
        if val_ecologico > 0: labels.append("Eco"); source.append(0); target.append(idx); value.append(val_ecologico); color.append("rgba(149, 165, 166, 0.6)"); idx += 1
        if datos_nodo["evaporacion_m3s"] > 0: labels.append("Evaporación"); source.append(0); target.append(idx); value.append(datos_nodo["evaporacion_m3s"]); color.append("rgba(189, 195, 199, 0.3)")

        fig_sankey = go.Figure(data=[go.Sankey(valueformat=".2f", valuesuffix=" m³/s", textfont=dict(size=15, color="black"), node=dict(pad=20, thickness=30, label=labels), link=dict(source=source, target=target, value=value, color=color))])
        fig_sankey.update_layout(height=480)
        st.plotly_chart(fig_sankey, use_container_width=True)

with st.expander("🔬 Ecuaciones de Dinámica de Sistemas (Embalses)", expanded=False):
    st.markdown("La variación de almacenamiento en el tiempo se rige por la ecuación de continuidad: $\\frac{\\Delta S}{\\Delta t} = I_{nat} + \\sum I_{trasvases} - O_{urb} - O_{eco} - O_{energia} - E_{vap}$")
