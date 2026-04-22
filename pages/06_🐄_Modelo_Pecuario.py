# pages/06_🐄_Modelo_Pecuario.py

import os
import sys
import warnings
import re
import unicodedata

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from shapely.geometry import shape
from scipy.optimize import curve_fit # 🔥 Agregado para que los modelos matemáticos funcionen

import streamlit as st

# --- 1. CONFIGURACIÓN DE PÁGINA (SIEMPRE PRIMERO) ---
st.set_page_config(page_title="Modelo Pecuario", page_icon="🐄", layout="wide")
warnings.filterwarnings('ignore')

# --- 📂 IMPORTACIÓN ROBUSTA DE MÓDULOS ---
try:
    from modules import selectors
    from modules.utils import encender_gemelo_digital
except ImportError:
    # Fallback de rutas por si hay problemas de lectura entre carpetas
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from modules import selectors
    from modules.utils import encender_gemelo_digital

# ==========================================
# 📂 NUEVO: MENÚ DE NAVEGACIÓN PERSONALIZADO
# ==========================================
# Llama al menú expandible y resalta la página actual (Corregido)
selectors.renderizar_menu_navegacion("Modelo Pecuario")

# ====================================================================
# 🧽 MOTOR DE ESTANDARIZACIÓN UNIVERSAL (Sincronía con Pág 07)
# ====================================================================
def limpiar_texto_maestro(texto):
    """Limpia tildes, pasa a minúsculas, elimina guiones y colapsa espacios."""
    if pd.isna(texto): return ""
    t = unicodedata.normalize('NFKD', str(texto).lower().strip()).encode('ascii', 'ignore').decode('utf-8')
    t = re.sub(r'[^a-z0-9]', ' ', t) # Todo lo que no sea letra o número se vuelve espacio
    return " ".join(t.split()) # Colapsa múltiples espacios

# ====================================================================
# 💉 ENCENDIDO DEL SISTEMA INMUNOLÓGICO Y VARIABLES GLOBALES
# ====================================================================
try:
    from modules.utils import encender_gemelo_digital
    encender_gemelo_digital()
except Exception:
    pass

@st.cache_data(ttl=3600)
def cargar_historico_pecuario():
    # URL directa a tu archivo maestro en Supabase
    url = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/sihcli_maestros/Censo_Pecuario_Historico_Cuencas.csv"
    return pd.read_csv(url)

df_pecuario = cargar_historico_pecuario()

# =====================================================================
# PESTAÑA PRINCIPAL: GENERADOR DE MATRIZ MAESTRA PECUARIA
# =====================================================================
st.title("🐄 Motor Demográfico Pecuario (Bovinos, Porcinos, Aves)")
st.markdown("""
Este motor lee los censos del ICA (2018-2025) y ejecuta un análisis de **Dasimetría Espacial** utilizando el mapa de usos del suelo (Pastos). Luego entrena modelos predictivos, blindándolos 
con la **Llave Universal**.
""")

if st.button("⚙️ Iniciar Forja Pecuaria Integral (Espacial + Matemática)", type="primary"):
    texto_progreso = st.empty()
    barra_progreso = st.progress(0)
    
    try:
        # --- IMPORTACIONES EXTRAS ---
        import geopandas as gpd
        from sqlalchemy import text
        from rasterstats import zonal_stats
        import tempfile
        import urllib.request
        import gc
        from modules.db_manager import get_engine
        engine_geo = get_engine()

        # =================================================================
        # 📍 FASE 1: DASIMETRÍA ESPACIAL (EL BISTURÍ MULTI-HÁBITAT)
        # =================================================================
        texto_progreso.info("📍 Fase 1/2: Descargando Raster de Usos del Suelo (2022)...")
        URL_RASTER = "https://ldunpssoxvifemoyeuac.supabase.co/storage/v1/object/public/rasters/Cob25m_WGS84.tif"
        tmp_raster = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
        urllib.request.urlretrieve(URL_RASTER, tmp_raster.name)
        raster_path = tmp_raster.name

        texto_progreso.info("🗺️ Fase 1/2: Cruzando cartografía (Municipios y Cuencas)...")
        gdf_mun = gpd.read_postgis("SELECT * FROM municipios", engine_geo, geom_col="geometry").to_crs(epsg=4326)
        q_cue = text("SELECT COALESCE(NULLIF(TRIM(nom_nss3), ''), 'Cuenca Sin Nombre') AS subc_lbl, geometry FROM cuencas")
        gdf_cue = gpd.read_postgis(q_cue, engine_geo, geom_col="geometry").to_crs(epsg=4326)

        col_mun = 'mpio_cnmbr' if 'mpio_cnmbr' in gdf_mun.columns else ('MPIO_CNMBR' if 'MPIO_CNMBR' in gdf_mun.columns else 'municipio')
        gdf_mun['mun_norm'] = gdf_mun[col_mun].astype(str).str.upper().str.strip()
        
        gdf_mun['geometry'] = gdf_mun.geometry.buffer(0)
        gdf_cue['geometry'] = gdf_cue.geometry.buffer(0)
        
        inter_mc = gpd.overlay(gdf_mun[['mun_norm', 'geometry']], gdf_cue[['subc_lbl', 'geometry']], how='intersection')
        inter_mc = inter_mc[inter_mc.geometry.area > 0.000001].copy()
        inter_mc['area_geo_frag'] = inter_mc.geometry.area
        
        del gdf_mun, gdf_cue; gc.collect()

        # 4. ESCÁNER MULTI-HÁBITAT (Raster Stats)
        texto_progreso.info("🌱 Fase 1/2: Escaneando usos del suelo para hábitats específicos...")
        stats = zonal_stats(inter_mc, raster_path, categorical=True, nodata=-9999)
        
        # 🎯 Clasificación de Hábitats según Leyenda Oficial (land_cover.py)
        CLASES_BOVINOS = [7] # Pastos
        CLASES_GRANJAS = [2, 3, 4, 5, 6, 8] # Porcinos y Aves (Zonas degradadas, artificiales, industriales y cultivos)
        
        # Sumamos los píxeles correspondientes a cada especie en cada pedazo de cuenca
        inter_mc['pixeles_bovinos'] = [sum(stat.get(c, 0) for c in CLASES_BOVINOS) for stat in stats]
        inter_mc['pixeles_granjas'] = [sum(stat.get(c, 0) for c in CLASES_GRANJAS) for stat in stats]

        # 5. CÁLCULO DE PESOS DASIMÉTRICOS (Con Fallback Independiente)
        texto_progreso.info("⚖️ Fase 1/2: Calculando pesos dasimétricos por especie...")
        
        bovinos_municipio = inter_mc.groupby('mun_norm')['pixeles_bovinos'].transform('sum')
        granjas_municipio = inter_mc.groupby('mun_norm')['pixeles_granjas'].transform('sum')
        area_geo_municipio = inter_mc.groupby('mun_norm')['area_geo_frag'].transform('sum')

        # Peso para Vacas (Busca pastos [7], si no hay, usa área geográfica)
        inter_mc['peso_bovinos'] = np.where(
            bovinos_municipio > 0,
            inter_mc['pixeles_bovinos'] / bovinos_municipio,
            inter_mc['area_geo_frag'] / area_geo_municipio
        )
        
        # Peso para Cerdos y Aves (Busca galpones/cultivos [2,3,4,5,6,8], si no hay, usa área geográfica)
        inter_mc['peso_granjas'] = np.where(
            granjas_municipio > 0,
            inter_mc['pixeles_granjas'] / granjas_municipio,
            inter_mc['area_geo_frag'] / area_geo_municipio
        )

        import os
        os.remove(raster_path) # Limpieza del servidor
        barra_progreso.progress(0.4)

        # =================================================================
        # 🧠 FASE 2: ENTRENAMIENTO MATEMÁTICO MULTIESCALA
        # =================================================================
        texto_progreso.info("🧠 Fase 2/2: Entrenando modelos matemáticos multiescala...")
        
        def f_log(t, k, a, r): return k / (1 + a * np.exp(-r * t))
        def f_exp(t, a, b): return a * np.exp(b * t)
        def calcular_r2(y_real, y_pred):
            ss_res = np.sum((y_real - y_pred) ** 2)
            ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        matriz_resultados = []
        
        def ajustar_modelos(x, y, nivel, territorio, especie):
            if len(x) < 3 or max(y) <= 0: return 
            
            x_offset = x[0]
            x_norm = x - x_offset
            p0_val = max(1, y[0])
            max_y = max(y)
            es_creciente = y[-1] >= p0_val
            
            log_k, log_a, log_r, log_r2 = 0, 0, 0, 0
            try:
                k_max = max_y * 3.0 if es_creciente else max_y * 1.1
                k_guess = max_y * 1.2 if es_creciente else (y[-1] * 0.9 if y[-1] > 0 else max_y)
                a_guess = (k_guess - p0_val) / p0_val if p0_val > 0 else 1
                r_guess = 0.02 if es_creciente else -0.02
                k_min = max_y * 0.8 if es_creciente else max_y * 0.1
                limites = ([k_min, 0, -0.2], [k_max, np.inf, 0.3])
                popt_log, _ = curve_fit(f_log, x_norm, y, p0=[k_guess, a_guess, r_guess], bounds=limites, maxfev=50000)
                log_k, log_a, log_r = popt_log
                log_r2 = calcular_r2(y, f_log(x_norm, *popt_log))
            except Exception: pass

            exp_a, exp_b, exp_r2 = 0, 0, 0
            try:
                popt_exp, _ = curve_fit(f_exp, x_norm, y, p0=[p0_val, 0.01], maxfev=50000)
                exp_a, exp_b = popt_exp
                exp_r2 = calcular_r2(y, f_exp(x_norm, *popt_exp))
            except Exception: pass

            # 3. POLINOMIAL (Grado 3)
            poly_A, poly_B, poly_C, poly_D, poly_r2 = 0, 0, 0, 0, 0
            try:
                coefs = np.polyfit(x_norm, y, 3)
                poly_A, poly_B, poly_C, poly_D = coefs
                poly_r2 = calcular_r2(y, np.polyval(coefs, x_norm))
            except Exception: pass

            # 🚀 NUEVO: 4. LINEAL (Grado 1)
            lin_m, lin_b, lin_r2 = 0, 0, 0
            try:
                coefs_lin = np.polyfit(x_norm, y, 1)
                lin_m, lin_b = coefs_lin
                lin_r2 = calcular_r2(y, np.polyval(coefs_lin, x_norm))
            except Exception: pass

            # ⚖️ JUEZ ACTUALIZADO: MEJOR MODELO
            dic_modelos = {'Logístico': log_r2, 'Exponencial': exp_r2, 'Polinomial_3': poly_r2, 'Lineal': lin_r2}
            mejor_modelo = max(dic_modelos, key=dic_modelos.get)
            mejor_r2 = dic_modelos[mejor_modelo]

            # 🔥 LLAVE UNIVERSAL
            llave_u = f"{nivel}_{territorio}_TOTAL".upper().replace(" ", "_")

            matriz_resultados.append({
                'Especie': especie, 'Nivel': nivel, 'Territorio': territorio,
                'LLAVE_UNIVERSAL': llave_u, 'Año_Base': int(x_offset), 'Poblacion_Base': round(p0_val, 0),
                'Log_K': log_k, 'Log_a': log_a, 'Log_r': log_r, 'Log_R2': round(log_r2, 4),
                'Exp_a': exp_a, 'Exp_b': exp_b, 'Exp_R2': round(exp_r2, 4),
                'Poly_A': poly_A, 'Poly_B': poly_B, 'Poly_C': poly_C, 'Poly_D': poly_D, 'Poly_R2': round(poly_r2, 4),
                'Lin_m': lin_m, 'Lin_b': lin_b, 'Lin_R2': round(lin_r2, 4), # <-- INYECCIÓN LINEAL
                'Modelo_Recomendado': mejor_modelo, 'Mejor_R2': round(mejor_r2, 4)
            })

# =================================================================
        # B. ENTRENAMIENTO DE CUENCAS (Dasimetría Multiespecie + Cascada Hídrica)
        # =================================================================
        texto_progreso.info("🧮 Fase 2/2: Aplicando Pesos de Hábitat y Escalando Cuencas...")
        fragmentos_pecuarios = []
        df_censo = df_pecuario.copy()
        col_censo_mun = 'Municipio_Norm' if 'Municipio_Norm' in df_censo.columns else 'Municipio'
        df_censo['mun_norm'] = df_censo[col_censo_mun].astype(str).str.upper().str.strip()

        # 1. Repartición Dasimétrica Base (Nivel Fragmento)
        for mpio in inter_mc['mun_norm'].unique():
            df_animales_mpio = df_censo[df_censo['mun_norm'] == mpio].copy()
            if df_animales_mpio.empty: continue
            
            pedazos = inter_mc[inter_mc['mun_norm'] == mpio]
            for _, pedazo in pedazos.iterrows():
                df_frag = df_animales_mpio.copy()
                # 🧬 Repartición dasimétrica multiespecie
                df_frag['Bovinos'] = df_frag['Bovinos'] * pedazo['peso_bovinos']
                df_frag['Porcinos'] = df_frag['Porcinos'] * pedazo['peso_granjas']
                df_frag['Aves'] = df_frag['Aves'] * pedazo['peso_granjas']
                df_frag['subc_lbl'] = pedazo['subc_lbl']
                fragmentos_pecuarios.append(df_frag)

        if fragmentos_pecuarios:
            df_hist_pecuario = pd.concat(fragmentos_pecuarios)
            df_hist_pecuario['subc_lbl'] = df_hist_pecuario['subc_lbl'].astype(str).str.strip()
            
            # 2. Descargamos la genealogía hídrica de la base de datos
            q_jerarquia = text("""
                SELECT DISTINCT nomah, nomzh, nom_szh, nom_nss1, nom_nss2, nom_nss3,
                COALESCE(NULLIF(TRIM(nom_nss3), ''), 'Cuenca Sin Nombre') AS subc_lbl FROM cuencas
            """)
            df_arbol = pd.read_sql(q_jerarquia, engine_geo)
            for c in df_arbol.columns: df_arbol[c] = df_arbol[c].astype(str).str.strip()
            
            # 3. Fusión Hídrica (Pegamos los padres a los fragmentos base)
            df_hidro_completo = pd.merge(df_hist_pecuario, df_arbol.drop_duplicates(subset=['subc_lbl']), on='subc_lbl', how='left')
            
            # 4. Entrenamiento Top-Down (Cascada Hidrológica: AH -> NSS3)
            niveles_hidro = {
                'nomah': 'AH', 'nomzh': 'ZH', 'nom_szh': 'SZH', 
                'nom_nss1': 'NSS1', 'nom_nss2': 'NSS2', 'subc_lbl': 'NSS3'
            }
            
            for col_nivel, etiqueta in niveles_hidro.items():
                if col_nivel in df_hidro_completo.columns:
                    df_nivel = df_hidro_completo.dropna(subset=[col_nivel]).groupby([col_nivel, 'Anio'])[['Bovinos', 'Porcinos', 'Aves']].sum().reset_index()
                    for terr in df_nivel[col_nivel].unique():
                        if terr in ["", "None", "nan", "Cuenca Sin Nombre"]: continue
                        df_t = df_nivel[df_nivel[col_nivel] == terr].sort_values('Anio')
                        for esp in ['Bovinos', 'Porcinos', 'Aves']:
                            ajustar_modelos(df_t['Anio'].values, df_t[esp].values, etiqueta, terr, esp)

        barra_progreso.progress(0.7)

        # =================================================================
        # C. ENTRENAMIENTO ADMINISTRATIVO (Mpio, Depto, Región, Macroregión)
        # =================================================================
        texto_progreso.info("🏢 Fase 2/2: Entrenando Escalas Administrativas Superiores...")
        
        # 1. Recuperamos metadata de los municipios desde PostGIS para saber sus regiones
        try:
            df_meta_mun = gpd.read_postgis("SELECT mpio_cnmbr, subregion, macroregion FROM municipios", engine_geo)
            df_meta_mun['mun_norm'] = df_meta_mun['mpio_cnmbr'].astype(str).str.upper().str.strip()
            df_censo = pd.merge(df_censo, df_meta_mun[['mun_norm', 'subregion', 'macroregion']].drop_duplicates(), on='mun_norm', how='left')
        except Exception:
            pass # Si falla la conexión, igual entrenará Municipios y Depto

        # Municipios
        df_mpios = df_censo.groupby(['Anio', 'mun_norm'])[['Bovinos', 'Porcinos', 'Aves']].sum().reset_index()
        for mpio in df_mpios['mun_norm'].unique():
            df_t = df_mpios[df_mpios['mun_norm'] == mpio].sort_values('Anio')
            for esp in ['Bovinos', 'Porcinos', 'Aves']:
                ajustar_modelos(df_t['Anio'].values, df_t[esp].values, 'MUNICIPIO', mpio, esp)

        # Subregiones
        if 'subregion' in df_censo.columns:
            df_reg = df_censo.groupby(['Anio', 'subregion'])[['Bovinos', 'Porcinos', 'Aves']].sum().reset_index()
            for reg in df_reg['subregion'].dropna().unique():
                if reg in ["", "None", "nan"]: continue
                df_t = df_reg[df_reg['subregion'] == reg].sort_values('Anio')
                for esp in ['Bovinos', 'Porcinos', 'Aves']:
                    ajustar_modelos(df_t['Anio'].values, df_t[esp].values, 'REGION', reg, esp)
                    
        # Macroregiones
        if 'macroregion' in df_censo.columns:
            df_mac = df_censo.groupby(['Anio', 'macroregion'])[['Bovinos', 'Porcinos', 'Aves']].sum().reset_index()
            for mac in df_mac['macroregion'].dropna().unique():
                if mac in ["", "None", "nan"]: continue
                df_t = df_mac[df_mac['macroregion'] == mac].sort_values('Anio')
                for esp in ['Bovinos', 'Porcinos', 'Aves']:
                    ajustar_modelos(df_t['Anio'].values, df_t[esp].values, 'MACROREGION', mac, esp)

        # Departamento
        df_depto = df_censo.groupby('Anio')[['Bovinos', 'Porcinos', 'Aves']].sum().reset_index()
        for esp in ['Bovinos', 'Porcinos', 'Aves']:
            ajustar_modelos(df_depto['Anio'].values, df_depto[esp].values, 'DEPARTAMENTO', 'Antioquia', esp)

        # =================================================================
        # D. CARGA A LA MEMORIA VIVA
        # =================================================================
        df_matriz_pec = pd.DataFrame(matriz_resultados)
        st.session_state['df_matriz_pecuaria'] = df_matriz_pec 
        
        barra_progreso.progress(1.0)
        texto_progreso.success(f"✅ ¡Forja Pecuaria Integral Exitosa! {len(df_matriz_pec)} modelos blindados.")

    except Exception as e:
        st.error(f"🚨 Error en el Motor Pecuario: {e}")
        
# =====================================================================
# 🔬 VALIDADOR VISUAL COMPARATIVO Y SINCRONIZADOR
# =====================================================================
if 'df_matriz_pecuaria' in st.session_state:
    st.divider()
    st.subheader("🔬 Validador Visual Pecuario y Sincronización Hídrica")
    
    df_mat = st.session_state['df_matriz_pecuaria']
    
    c_nav1, c_nav2, c_nav3 = st.columns([1, 1.5, 1])
    with c_nav1:
        niveles_disp = list(df_mat['Nivel'].unique())
        idx_mun = niveles_disp.index('NSS3') if 'NSS3' in niveles_disp else 0
        nivel_val = st.selectbox("1. Escala Espacial:", niveles_disp, index=idx_mun)
    with c_nav2:
        territorios_disp = sorted(df_mat[df_mat['Nivel'] == nivel_val]['Territorio'].unique())
        idx_terr = 0
        terr_val = st.selectbox("2. Territorio Hídrico / Administrativo:", territorios_disp, index=idx_terr)
    with c_nav3:
        anio_futuro = st.slider("3. Proyectar hasta el año:", min_value=2025, max_value=2050, value=2035, step=1)

    st.markdown("---")
    
    # Filtramos los datos exactos del territorio que el usuario eligió
    df_filtrado = df_mat[(df_mat['Nivel'] == nivel_val) & (df_mat['Territorio'] == terr_val)]
    
    if df_filtrado.empty:
        st.warning(f"No hay datos proyectados para {terr_val} en el nivel {nivel_val}.")
    else:
        import plotly.graph_objects as go
        import numpy as np
        
        # Organizamos las gráficas en 3 pestañas limpias
        tabs = st.tabs(["🐄 Bovinos", "🐖 Porcinos", "🐔 Aves"])
        especies = ["Bovinos", "Porcinos", "Aves"]
        
        for tab, especie in zip(tabs, especies):
            with tab:
                fila_esp = df_filtrado[df_filtrado['Especie'] == especie]
                if fila_esp.empty:
                    st.info(f"No se detectó vocación de {especie} en esta zona según el mapa de hábitats.")
                    continue
                    
                fila_terr = fila_esp.iloc[0]
                mejor_modelo = fila_terr.get('Modelo_Recomendado', 'Logístico')
                
                x_offset = fila_terr.get('Año_Base', 2018)
                x_pred = np.arange(x_offset, anio_futuro + 1)
                x_norm_pred = x_pred - x_offset
                
                fig = go.Figure()
                
                def config_linea(nombre_mod, color):
                    es_ganador = mejor_modelo == nombre_mod
                    return dict(color=color, width=4 if es_ganador else 2, dash='solid' if es_ganador else 'dash'), 1.0 if es_ganador else 0.4

                # 1. Logístico
                if 'Log_K' in fila_terr and not pd.isna(fila_terr['Log_K']) and fila_terr['Log_K'] > 0:
                    y_log = fila_terr['Log_K'] / (1 + fila_terr['Log_a'] * np.exp(-fila_terr['Log_r'] * x_norm_pred))
                    line_log, op_log = config_linea('Logístico', '#2980b9')
                    fig.add_trace(go.Scatter(x=x_pred, y=y_log, mode='lines', name=f"Logístico (R²: {fila_terr.get('Log_R2',0):.4f})", line=line_log, opacity=op_log))
                    
                # 2. Exponencial
                if 'Exp_a' in fila_terr and not pd.isna(fila_terr['Exp_a']):
                    y_exp = fila_terr['Exp_a'] * np.exp(fila_terr['Exp_b'] * x_norm_pred)
                    line_exp, op_exp = config_linea('Exponencial', '#e67e22')
                    fig.add_trace(go.Scatter(x=x_pred, y=y_exp, mode='lines', name=f"Exponencial (R²: {fila_terr.get('Exp_R2',0):.4f})", line=line_exp, opacity=op_exp))
                    
                # 3. Polinomial (Grado 3)
                if 'Poly_A' in fila_terr and not pd.isna(fila_terr['Poly_A']):
                    y_poly = fila_terr['Poly_A']*(x_norm_pred**3) + fila_terr['Poly_B']*(x_norm_pred**2) + fila_terr['Poly_C']*x_norm_pred + fila_terr['Poly_D']
                    line_poly, op_poly = config_linea('Polinomial_3', '#27ae60')
                    fig.add_trace(go.Scatter(x=x_pred, y=y_poly, mode='lines', name=f"Polinomial (R²: {fila_terr.get('Poly_R2',0):.4f})", line=line_poly, opacity=op_poly))

                # 🚀 4. LINEAL (El nuevo modelo estable)
                if 'Lin_m' in fila_terr and not pd.isna(fila_terr['Lin_m']):
                    y_lin = fila_terr['Lin_m'] * x_norm_pred + fila_terr['Lin_b']
                    line_lin, op_lin = config_linea('Lineal', '#8e44ad') # Color Morado
                    fig.add_trace(go.Scatter(x=x_pred, y=y_lin, mode='lines', name=f"Lineal (R²: {fila_terr.get('Lin_R2',0):.4f})", line=line_lin, opacity=op_lin))

                fig.update_layout(
                    title=f"Proyección de {especie} - {terr_val}", 
                    xaxis_title="Año", yaxis_title="Inventario (Cabezas/Aves)", hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 🔑 Mostramos la llave para auditoría
                llave = fila_terr.get('LLAVE_UNIVERSAL', 'NO_GENERADA')
                st.caption(f"🔑 **Llave Universal:** `{llave}` | 🏆 Modelo Recomendado: **{mejor_modelo}**")

        st.success(f"🔗 Carga pecuaria de **{terr_val}** para el año **{anio_futuro}** lista en memoria RAM.")
        
    # ==============================================================================
    # 🧠 TRANSMISIÓN AL CEREBRO GLOBAL (EL ALEPH)
    # ==============================================================================
    def calcular_proyeccion_especie(df, nivel, terr, esp, anio_obj):
        df_f = df[(df['Nivel'] == nivel) & (df['Territorio'] == terr) & (df['Especie'] == esp)]
        if df_f.empty: return 0.0
        f = df_f.iloc[0]
        x_norm = anio_obj - f['Año_Base']
        mod = f['Modelo_Recomendado']
        
        # Evaluador Matemático Universal
        if mod == 'Logístico': return f['Log_K'] / (1 + f['Log_a'] * np.exp(-f['Log_r'] * x_norm))
        elif mod == 'Exponencial': return f['Exp_a'] * np.exp(f['Exp_b'] * x_norm)
        elif mod == 'Lineal': return f.get('Lin_m', 0) * x_norm + f.get('Lin_b', 0)
        else: return f['Poly_A']*(x_norm**3) + f['Poly_B']*(x_norm**2) + f['Poly_C']*x_norm + f['Poly_D']

    # Extraer valores exactos para inyección a la memoria viva de la app
    res_bov = calcular_proyeccion_especie(df_mat, nivel_val, terr_val, 'Bovinos', anio_futuro)
    res_por = calcular_proyeccion_especie(df_mat, nivel_val, terr_val, 'Porcinos', anio_futuro)
    res_ave = calcular_proyeccion_especie(df_mat, nivel_val, terr_val, 'Aves', anio_futuro)

    st.session_state['ica_bovinos_calc_met'] = float(max(0, res_bov))
    st.session_state['ica_porcinos_calc_met'] = float(max(0, res_por))
    st.session_state['ica_aves_calc_met'] = float(max(0, res_ave))
    st.session_state['aleph_lugar_pecuario'] = terr_val
    
    st.success(f"🔗 Carga pecuaria de **{terr_val}** para el año **{anio_futuro}** lista en memoria RAM.")
    st.markdown("---")
    
    # --- RENDERIZADO VISUAL A DOBLE PANEL ---
    def renderizar_panel_pecuario(especie_sel, key_suffix):
        df_filtrado = df_mat[(df_mat['Nivel'] == nivel_val) & (df_mat['Territorio'] == terr_val) & (df_mat['Especie'] == especie_sel)]
        if df_filtrado.empty:
            st.warning(f"No hay registros o modelos viables para {especie_sel} en {terr_val}.")
            return
            
        fila_terr = df_filtrado.iloc[0]
        mejor_modelo = fila_terr['Modelo_Recomendado']
        
        # Reconstruir Histórico
        if nivel_val == 'DEPARTAMENTO': df_hist = df_pecuario.groupby('Anio')[especie_sel].sum().reset_index()
        elif nivel_val == 'MUNICIPIO': df_hist = df_pecuario[df_pecuario['Municipio_Norm'] == terr_val].groupby('Anio')[especie_sel].sum().reset_index()
        elif nivel_val == 'NSS3': df_hist = df_pecuario[df_pecuario['Subcuenca'] == terr_val].groupby('Anio')[especie_sel].sum().reset_index()
        elif nivel_val == 'REGION': df_hist = df_pecuario[df_pecuario['subregion'] == terr_val].groupby('Anio')[especie_sel].sum().reset_index() if 'subregion' in df_pecuario.columns else pd.DataFrame()
        elif nivel_val == 'MACROREGION': df_hist = df_pecuario[df_pecuario['macroregion'] == terr_val].groupby('Anio')[especie_sel].sum().reset_index() if 'macroregion' in df_pecuario.columns else pd.DataFrame()
        else: df_hist = df_pecuario[df_pecuario['Sistema'] == terr_val].groupby('Anio')[especie_sel].sum().reset_index() if 'Sistema' in df_pecuario.columns else pd.DataFrame()
            
        if not df_hist.empty:
            df_hist = df_hist.sort_values(by='Anio')
            x_hist = df_hist['Anio'].values
            y_hist = df_hist[especie_sel].values
        else:
            x_hist, y_hist = [], []
        
        x_offset = fila_terr['Año_Base']
        x_pred = np.arange(x_offset, anio_futuro + 1)
        x_norm_pred = x_pred - x_offset
        
        fig = go.Figure()
        color_data = {'Bovinos': 'brown', 'Porcinos': 'deeppink', 'Aves': 'goldenrod'}
        icono = {'Bovinos': '🐄', 'Porcinos': '🐖', 'Aves': '🐔'}
        
        if len(x_hist) > 0:
            fig.add_trace(go.Scatter(x=x_hist, y=y_hist, mode='markers', name='Censo ICA', marker=dict(color=color_data[especie_sel], size=10, symbol='diamond')))
        
        def config_linea(nombre_mod, color_mod):
            es_ganador = mejor_modelo == nombre_mod
            return dict(color=color_mod, width=4 if es_ganador else 2, dash='solid' if es_ganador else 'dash'), 1.0 if es_ganador else 0.4
            
        # 1. Logístico
        if 'Log_K' in fila_terr and not pd.isna(fila_terr['Log_K']) and fila_terr['Log_K'] > 0:
            y_log = fila_terr['Log_K'] / (1 + fila_terr['Log_a'] * np.exp(-fila_terr['Log_r'] * x_norm_pred))
            line_log, op_log = config_linea('Logístico', '#2980b9')
            fig.add_trace(go.Scatter(x=x_pred, y=y_log, mode='lines', name=f"Logístico (R²: {fila_terr.get('Log_R2', 0):.4f})", line=line_log, opacity=op_log))
            
        # 2. Exponencial
        if 'Exp_a' in fila_terr and not pd.isna(fila_terr['Exp_a']):
            y_exp = fila_terr['Exp_a'] * np.exp(fila_terr['Exp_b'] * x_norm_pred)
            line_exp, op_exp = config_linea('Exponencial', '#e67e22')
            fig.add_trace(go.Scatter(x=x_pred, y=y_exp, mode='lines', name=f"Exponencial (R²: {fila_terr.get('Exp_R2', 0):.4f})", line=line_exp, opacity=op_exp))
            
        # 3. Polinomial
        if 'Poly_A' in fila_terr and not pd.isna(fila_terr['Poly_A']):
            y_poly = fila_terr['Poly_A']*(x_norm_pred**3) + fila_terr['Poly_B']*(x_norm_pred**2) + fila_terr['Poly_C']*x_norm_pred + fila_terr['Poly_D']
            line_poly, op_poly = config_linea('Polinomial_3', '#27ae60')
            fig.add_trace(go.Scatter(x=x_pred, y=y_poly, mode='lines', name=f"Polinomial 3 (R²: {fila_terr.get('Poly_R2', 0):.4f})", line=line_poly, opacity=op_poly))
            
        # 🚀 4. Lineal
        if 'Lin_m' in fila_terr and not pd.isna(fila_terr['Lin_m']):
            y_lin = fila_terr['Lin_m'] * x_norm_pred + fila_terr['Lin_b']
            line_lin, op_lin = config_linea('Lineal', '#8e44ad')
            fig.add_trace(go.Scatter(x=x_pred, y=y_lin, mode='lines', name=f"Lineal (R²: {fila_terr.get('Lin_R2', 0):.4f})", line=line_lin, opacity=op_lin))
        
        llave_u = fila_terr.get('LLAVE_UNIVERSAL', 'Generando...')
        fig.update_layout(
            title=f"Proyección de {icono[especie_sel]} {especie_sel} (Ganador: {mejor_modelo})<br><sup>🔑 Llave: {llave_u}</sup>", 
            xaxis_title="Año", yaxis_title="Número de Animales", hovermode="x unified", 
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True, key=f"plot_{key_suffix}")
        
        lado = "Panel Izquierdo" if key_suffix == "g1" else "Panel Derecho"
        with st.expander(f"📐 Parámetros del Modelo de {especie_sel} ({lado})", expanded=False):
            df_coefs = pd.DataFrame([
                {"Modelo": "Logístico", "R²": f"{fila_terr.get('Log_R2', 0):.4f}", "Parámetros": f"K={fila_terr.get('Log_K', 0):.0f}, a={fila_terr.get('Log_a', 0):.4f}, r={fila_terr.get('Log_r', 0):.4f}"},
                {"Modelo": "Exponencial", "R²": f"{fila_terr.get('Exp_R2', 0):.4f}", "Parámetros": f"a={fila_terr.get('Exp_a', 0):.0f}, b={fila_terr.get('Exp_b', 0):.4f}"},
                {"Modelo": "Polinomial 3", "R²": f"{fila_terr.get('Poly_R2', 0):.4f}", "Parámetros": f"A={fila_terr.get('Poly_A', 0):.4e}, B={fila_terr.get('Poly_B', 0):.4e}, C={fila_terr.get('Poly_C', 0):.4f}, D={fila_terr.get('Poly_D', 0):.0f}"},
                {"Modelo": "Lineal", "R²": f"{fila_terr.get('Lin_R2', 0):.4f}", "Parámetros": f"m={fila_terr.get('Lin_m', 0):.4f}, b={fila_terr.get('Lin_b', 0):.0f}"}
            ])
            def highlight_winner(row): return ['background-color: #d4edda' if row['Modelo'].startswith(mejor_modelo[:4]) else '' for _ in row]
            st.dataframe(df_coefs.style.apply(highlight_winner, axis=1), use_container_width=True)

    col_graf_1, col_graf_2 = st.columns(2)
    with col_graf_1:
        esp_1 = st.selectbox("Especie (Panel Izquierdo):", ["Bovinos", "Porcinos", "Aves"], index=0, key="sel_esp1")
        renderizar_panel_pecuario(esp_1, "g1")
    with col_graf_2:
        esp_2 = st.selectbox("Especie (Panel Derecho):", ["Bovinos", "Porcinos", "Aves"], index=1, key="sel_esp2")
        renderizar_panel_pecuario(esp_2, "g2")

# ==============================================================================
# 💾 EXPORTACIÓN AUTOMÁTICA A SQL Y DESCARGA (PRODUCCIÓN)
# ==============================================================================
if 'df_matriz_pecuaria' in st.session_state:
    st.markdown("---")
    st.subheader("💾 Exportar Cerebro Pecuario (Para Producción)")
    
    df_matriz_pec = st.session_state['df_matriz_pecuaria']
    
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        if st.button("🚀 Inyectar a Base de Datos (SQL)", type="primary", use_container_width=True):
            with st.spinner("Forjando Cerebro Pecuario y conectando con PostgreSQL..."):
                try:
                    from modules.db_manager import get_engine
                    from sqlalchemy import text 
                    engine_sql = get_engine()
                    
                    df_export = df_matriz_pec.copy()
                    
                    if 'Pob_Base' in df_export.columns and 'Poblacion_Base' not in df_export.columns:
                        df_export.rename(columns={'Pob_Base': 'Poblacion_Base'}, inplace=True)
                    
                    with engine_sql.begin() as conn:
                        conn.execute(text('ALTER TABLE matriz_maestra_pecuaria ADD COLUMN IF NOT EXISTS "LLAVE_UNIVERSAL" TEXT;'))
                        conn.execute(text('ALTER TABLE matriz_maestra_pecuaria ADD COLUMN IF NOT EXISTS "Lin_m" FLOAT, ADD COLUMN IF NOT EXISTS "Lin_b" FLOAT, ADD COLUMN IF NOT EXISTS "Lin_R2" FLOAT;'))
                        conn.execute(text("DELETE FROM matriz_maestra_pecuaria;"))
                        
                    df_export.to_sql('matriz_maestra_pecuaria', engine_sql, if_exists='append', index=False)
                    
                    st.success(f"✅ ¡Inyección Exitosa! {len(df_export)} registros blindados con LLAVE_UNIVERSAL actualizados en PostgreSQL.")
                    st.session_state['df_matriz_pecuaria'] = df_export
                    
                except Exception as e:
                    st.error(f"Error SQL: {e}")
                    
    with col_btn2:
        csv_matriz = df_matriz_pec.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar Matriz (CSV de Respaldo)", 
            data=csv_matriz, 
            file_name="Matriz_Multimodelo_Pecuaria.csv", 
            mime='text/csv',
            use_container_width=True
        )
