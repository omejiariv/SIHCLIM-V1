
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import curve_fit
import warnings
import os

warnings.filterwarnings("ignore") 

st.set_page_config(page_title="Demografía y Población", page_icon="👥", layout="wide")

st.title("👥 Dinámica Demográfica y Modelación Poblacional")
st.markdown("""
Motor de Inferencia Demográfica: Agregación territorial dinámica, ajuste paramétrico automático, proyecciones polinómicas y estructura por edades.
""")
st.divider()

# --- 1. LECTURA DE DATOS MAESTROS (AUTODETECCIÓN Y LIMPIEZA PROFUNDA) ---
def leer_csv_robusto(ruta):
    try:
        df = pd.read_csv(ruta, sep=None, engine='python')
        df.columns = df.columns.str.replace('\ufeff', '').str.strip()
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data
def cargar_macro():
    ruta1 = "data/Pob_Col_Ant_Amva_Med.xlsx"
    ruta2 = "data/Pob_Col_Ant_Amva_Med.csv"
    if os.path.exists(ruta1): df = pd.read_excel(ruta1)
    elif os.path.exists(ruta2): df = leer_csv_robusto(ruta2)
    else: return pd.DataFrame()
    df.columns = [str(c).replace('Pob_', '').replace('\ufeff', '').strip() for c in df.columns]
    return df

@st.cache_data
def cargar_municipios():
    ruta = "data/Pob_mpios_colombia.csv"
    if os.path.exists(ruta):
        df = leer_csv_robusto(ruta)
        if not df.empty and 'depto_nom' in df.columns and 'municipio' in df.columns:
            df.dropna(subset=['depto_nom', 'municipio'], inplace=True)
            return df
    return pd.DataFrame()

@st.cache_data
def cargar_edades():
    ruta = "data/Pob_sexo_edad_Colombia_1950-2070.csv"
    if os.path.exists(ruta):
        df = leer_csv_robusto(ruta)
        if not df.empty:
            if 'sexo' in df.columns: df['sexo'] = df['sexo'].astype(str).str.strip().str.lower()
            if 'area_geografica' in df.columns: df['area_geografica'] = df['area_geografica'].astype(str).str.strip()
            return df
    return pd.DataFrame()

@st.cache_data
def cargar_veredas():
    ruta = "data/veredas_Antioquia.xlsx"
    return pd.read_excel(ruta) if os.path.exists(ruta) else pd.DataFrame()

df_macro = cargar_macro()
df_mpios = cargar_municipios()
df_edades = cargar_edades()
df_veredas = cargar_veredas()

# --- MOTOR DE AGREGACIÓN MATEMÁTICA ---
def obtener_serie_historica(df_mp, df_ed, nivel, nombre_lugar, area_geo):
    if nivel == "Nacional (Colombia)":
        if df_ed.empty: return pd.DataFrame(), 'Colombia'
        mapa_area = {"Total": "Total", "Urbano": "Cabecera", "Rural": "Centros Poblados y Rural Disperso"}
        area_ed = mapa_area.get(area_geo, "Total")
        
        df_f = df_ed[(df_ed['area_geografica'].str.lower() == area_ed.lower()) & (df_ed['sexo'] == 'total')].copy()
        if df_f.empty: return pd.DataFrame(), 'Colombia'
        
        edades_cols = [str(i) for i in range(101) if str(i) in df_f.columns]
        df_f[edades_cols] = df_f[edades_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        df_f['Poblacion'] = df_f[edades_cols].sum(axis=1)
        
        serie = df_f.groupby('año')['Poblacion'].sum().reset_index()
        serie.rename(columns={'Poblacion': 'Colombia'}, inplace=True)
        return serie, 'Colombia'
        
    else:
        if df_mp.empty: return pd.DataFrame(), nombre_lugar
        if area_geo.lower() == "total":
            df_f = df_mp[df_mp['area_geografica'].str.lower().isin(['urbano', 'rural', 'cabecera', 'resto'])]
        else:
            df_f = df_mp[df_mp['area_geografica'].str.lower() == area_geo.lower()]
            
        if nivel == "Departamental":
            serie = df_f[df_f['depto_nom'] == nombre_lugar].groupby('año')['Poblacion'].sum().reset_index()
            serie.rename(columns={'Poblacion': nombre_lugar}, inplace=True)
            return serie, nombre_lugar
        elif nivel == "Municipal":
            serie = df_f[df_f['municipio'] == nombre_lugar].groupby('año')['Poblacion'].sum().reset_index()
            serie.rename(columns={'Poblacion': nombre_lugar}, inplace=True)
            return serie, nombre_lugar
            
    return pd.DataFrame(), nombre_lugar

# ==============================================================================
# ESTRUCTURA MAESTRA DE PESTAÑAS
# ==============================================================================
tab_datos, tab_modelos, tab_piramides, tab_anidados, tab_espacial = st.tabs([
    "📊 1. Censos Históricos", 
    "📈 2. Modelos & Optimización", 
    "🏗️ 3. Pirámides (100 Años)",
    "🌍 4. Downscaling (Anidado)",
    "🗺️ 5. Visor Espacial"
])

# Variables de seguridad
df_plot_model = pd.DataFrame()
nombre_col_model = ""
col_anio_model = ""

# ------------------------------------------------------------------------------
# TAB 1: CENSOS HISTÓRICOS Y AGREGACIÓN
# ------------------------------------------------------------------------------
with tab_datos:
    st.header("📊 Evolución Histórica Territorial")
    
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        escalas = ["Macro (Mundial/Histórico)", "Nacional (Colombia)", "Regional", "Departamental", "Municipal", "Corregimental", "Veredal"]
        nivel_sel = st.selectbox("Nivel Territorial:", escalas)
        
    with col_t2:
        lugar_sel = "N/A"
        if nivel_sel == "Macro (Mundial/Histórico)":
            if not df_macro.empty:
                opciones = [col for col in df_macro.columns if col.lower() != 'año']
                lugar_sel = st.selectbox("Zona Macro/Global:", opciones)
            else: st.error("Archivo Macro no encontrado.")
            
        elif nivel_sel == "Departamental":
            if not df_mpios.empty and 'depto_nom' in df_mpios.columns:
                opciones = sorted([str(x) for x in df_mpios['depto_nom'].unique()])
                lugar_sel = st.selectbox("Departamento:", opciones)
            else: st.warning("Datos departamentales no disponibles.")
            
        elif nivel_sel == "Municipal":
            if not df_mpios.empty and 'municipio' in df_mpios.columns:
                opciones = sorted([str(x) for x in df_mpios['municipio'].unique()])
                idx = opciones.index('Medellín') if 'Medellín' in opciones else 0
                lugar_sel = st.selectbox("Municipio:", opciones, index=idx)
            else: st.warning("Datos municipales no disponibles.")
            
        elif nivel_sel == "Veredal":
            if not df_veredas.empty:
                mpio_vereda = st.selectbox("Municipio para ver Veredas:", sorted(df_veredas['Municipio'].dropna().unique()))
                lugar_sel = mpio_vereda
            else: st.error("Archivo de veredas no encontrado.")
            
        elif nivel_sel in ["Regional", "Corregimental"]:
            st.info(f"Buscador {nivel_sel} en construcción.")
            
        else:
            lugar_sel = "Colombia"
            st.info("Extraído desde BD Edades (1950-2070)")
            
    with col_t3:
        if nivel_sel == "Macro (Mundial/Histórico)":
            area_sel = "Total"
            st.info("Datos Macro agregados.")
        elif nivel_sel in ["Regional", "Corregimental", "Veredal"]:
            area_sel = "Rural"
        else:
            area_sel = st.selectbox("Área Geográfica:", ["Total", "Urbano", "Rural"])
        
    # --- LÓGICA DE GRAFICADO SEGÚN ESCALA ---
    if nivel_sel == "Macro (Mundial/Histórico)":
        if not df_macro.empty and lugar_sel != "N/A":
            df_plot = df_macro[["Año", lugar_sel]].dropna()
            fig1 = px.line(df_plot, x="Año", y=lugar_sel, title=f"Historia de Largo Plazo: {lugar_sel}", markers=True)
            st.plotly_chart(fig1, use_container_width=True)
            df_plot_model = df_plot 
            nombre_col_model = lugar_sel
            col_anio_model = "Año"
            
    elif nivel_sel == "Veredal":
        if not df_veredas.empty and lugar_sel != "N/A":
            df_v_filtrado = df_veredas[df_veredas['Municipio'] == lugar_sel].dropna(subset=['Poblacion_hab'])
            if not df_v_filtrado.empty:
                fig1 = px.bar(df_v_filtrado.sort_values('Poblacion_hab', ascending=False), x='Vereda', y='Poblacion_hab', title=f"Población Veredal: {lugar_sel}", color='Poblacion_hab')
                st.plotly_chart(fig1, use_container_width=True)
            else: st.warning("No hay datos veredales para este municipio.")
            
    elif nivel_sel in ["Regional", "Corregimental"]:
        st.warning(f"⚠️ Para visualizar escalas {nivel_sel} se requiere agregar dichas clasificaciones a la base de datos municipal o veredal.")
        
    else: 
        df_plot, nombre_col = obtener_serie_historica(df_mpios, df_edades, nivel_sel, lugar_sel, area_sel)
        if not df_plot.empty:
            fig1 = px.line(df_plot, x="año", y=nombre_col, title=f"Crecimiento Histórico DANE: {nombre_col} ({area_sel})", markers=True)
            st.plotly_chart(fig1, use_container_width=True)
            df_plot_model = df_plot
            nombre_col_model = nombre_col
            col_anio_model = "año"

# ------------------------------------------------------------------------------
# TAB 2: MODELOS Y OPTIMIZACIÓN MATEMÁTICA
# ------------------------------------------------------------------------------
with tab_modelos:
    st.header("📈 Ajuste de Modelos Evolutivos")
    
    if df_plot_model.empty or not nombre_col_model:
        st.info("👆 Selecciona una escala válida en el Tab 1 que contenga datos de serie de tiempo.")
    elif nivel_sel in ["Veredal", "Regional", "Corregimental"]:
        st.error(f"❌ El motor de optimización requiere una serie de tiempo. La escala {nivel_sel} no posee historia validada.")
    else:
        col_opt1, col_opt2 = st.columns([1, 2.5])
        with col_opt1:
            st.subheader("Configuración")
            st.info(f"Modelando: **{nombre_col_model} ({area_sel})**")
            
            t_data_raw = df_plot_model[col_anio_model].values
            p_data = df_plot_model[nombre_col_model].values
            t_data = t_data_raw - t_data_raw.min()
            p0_val = float(p_data[0]) 
                
            t_max = st.slider("Años a proyectar (Horizonte):", 10, 100, 30)
            st.markdown("---")
            modelos_sel = st.multiselect("Curvas a evaluar:", ["Exponencial", "Logístico", "Geométrico", "Polinómico (Grado 2)", "Polinómico (Grado 3)"], default=["Logístico", "Polinómico (Grado 2)"])
            opt_auto = st.button("✨ Optimizar Parámetros Automáticamente", type="primary", use_container_width=True)

            st.caption("Parámetros Manuales:")
            r_man = st.number_input("Tasa (r):", value=0.02, format="%.4f")
            k_man = st.number_input("Capacidad (K):", value=float(p_data.max() * 2.0), step=1000.0)

        with col_opt2:
            def f_exp(t, p0, r): return p0 * np.exp(r * t)
            def f_log(t, k, p0, r): return k / (1 + ((k-p0)/p0) * np.exp(-r * t))
            def f_geom(t, p0, r): return p0 * (1 + r)**t
            def f_poly2(t, a, b, c): return a*t**2 + b*t + c
            def f_poly3(t, a, b, c, d): return a*t**3 + b*t**2 + c*t + d

            t_total = np.arange(0, max(t_data) + t_max + 1)
            anios_totales = t_total + t_data_raw.min()
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=t_data_raw, y=p_data, mode='markers', name='Datos Históricos', marker=dict(color='black', size=8)))

            res_text = []
            for mod in modelos_sel:
                y_pred = np.zeros_like(t_total, dtype=float)
                try:
                    if mod == "Exponencial":
                        if opt_auto: popt, _ = curve_fit(f_exp, t_data, p_data, p0=[p0_val, 0.01]); y_pred = f_exp(t_total, *popt); res_text.append(f"**Exp**: r={popt[1]:.4f}")
                        else: y_pred = f_exp(t_total, p0_val, r_man)
                    elif mod == "Logístico":
                        if opt_auto: popt, _ = curve_fit(f_log, t_data, p_data, p0=[max(p_data)*1.5, p0_val, 0.01], maxfev=10000); y_pred = f_log(t_total, *popt); res_text.append(f"**Log**: K={popt[0]:,.0f}, r={popt[2]:.4f}")
                        else: y_pred = f_log(t_total, k_man, p0_val, r_man)
                    elif mod == "Geométrico":
                        if opt_auto: popt, _ = curve_fit(f_geom, t_data, p_data, p0=[p0_val, 0.01]); y_pred = f_geom(t_total, *popt)
                        else: y_pred = f_geom(t_total, p0_val, r_man)
                    elif mod == "Polinómico (Grado 2)":
                        if opt_auto: popt, _ = curve_fit(f_poly2, t_data, p_data); y_pred = f_poly2(t_total, *popt)
                        else: y_pred = f_poly2(t_total, 1, 10, p0_val)
                    elif mod == "Polinómico (Grado 3)":
                        if opt_auto: popt, _ = curve_fit(f_poly3, t_data, p_data); y_pred = f_poly3(t_total, *popt)
                        else: y_pred = f_poly3(t_total, 1, 10, p0_val, 0)

                    fig2.add_trace(go.Scatter(x=anios_totales, y=y_pred, mode='lines', name=mod, line=dict(width=3, dash='dot' if opt_auto else 'solid')))
                except: pass

            fig2.update_layout(title="Proyección de Modelos", xaxis_title="Año", yaxis_title="Población", hovermode="x unified", height=550)
            st.plotly_chart(fig2, use_container_width=True)
            if opt_auto and res_text: st.success("✅ **Parámetros Óptimos:** " + " | ".join(res_text))

# ------------------------------------------------------------------------------
# TAB 3: ESTRUCTURAS Y PIRÁMIDES (100 Años)
# ------------------------------------------------------------------------------
with tab_piramides:
    st.header("🏗️ Pirámides y Estructura Poblacional (1950 - 2070)")
    col_p1, col_p2 = st.columns([1, 3])
    
    with col_p1:
        if not df_edades.empty:
            area_pir = st.selectbox("Área Geográfica:", df_edades['area_geografica'].unique())
            anio_pir = st.slider("Año de la Estructura:", int(df_edades['año'].min()), int(df_edades['año'].max()), 2024)
            modo_pir = st.radio("Modo de Visualización:", ["Pirámide (Hombres vs Mujeres)", "Total (Ambos Sexos combinados)"])
            
            df_f_pir = df_edades[(df_edades['año'] == anio_pir) & (df_edades['area_geografica'].str.lower() == area_pir.lower())]
        else: st.warning("Datos de edades no disponibles.")
        
    with col_p2:
        if not df_edades.empty and not df_f_pir.empty:
            edades_cols = [str(i) for i in range(101) if str(i) in df_f_pir.columns]
            try:
                fig_pir = go.Figure()
                
                if modo_pir == "Pirámide (Hombres vs Mujeres)":
                    df_h = df_f_pir[df_f_pir['sexo'] == 'hombres'][edades_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
                    df_m = df_f_pir[df_f_pir['sexo'] == 'mujeres'][edades_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
                    hombres = df_h.sum(axis=0).values
                    mujeres = df_m.sum(axis=0).values
                    
                    fig_pir.add_trace(go.Bar(y=edades_cols, x=hombres * -1, name='Hombres', orientation='h', marker=dict(color='#3498db'), hovertext=hombres))
                    fig_pir.add_trace(go.Bar(y=edades_cols, x=mujeres, name='Mujeres', orientation='h', marker=dict(color='#e74c3c'), hovertext=mujeres))
                    fig_pir.update_layout(title=f"Pirámide Demográfica ({anio_pir})", barmode='relative', yaxis_title='Edad Simple', xaxis_title='Población', height=600)
                else:
                    df_t = df_f_pir[df_f_pir['sexo'] == 'total'][edades_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
                    total_pob = df_t.sum(axis=0).values
                    
                    fig_pir.add_trace(go.Bar(x=edades_cols, y=total_pob, name='Total Población', marker=dict(color='#9b59b6')))
                    fig_pir.update_layout(title=f"Distribución Total por Edades ({anio_pir})", xaxis_title='Edad Simple', yaxis_title='Población', height=600)

                st.plotly_chart(fig_pir, use_container_width=True)
            except Exception as e:
                st.error(f"Error al graficar la estructura: {e}")

# ------------------------------------------------------------------------------
# TAB 4: DOWNSCALING (EL PUENTE A LAS VEREDAS)
# ------------------------------------------------------------------------------
with tab_anidados:
    st.header("🌍 Modelos Anidados: Municipio ➔ Vereda")
    st.markdown("Calcula la proyección futura de una vereda basándose en la tendencia de su municipio anfitrión.")
    
    col_a1, col_a2 = st.columns([1, 2])
    with col_a1:
        if not df_veredas.empty:
            macro_mpio = st.selectbox("Municipio Anfitrión (Macro):", sorted([str(x) for x in df_veredas['Municipio'].dropna().unique()]))
            v_disp = df_veredas[df_veredas['Municipio'] == macro_mpio].dropna(subset=['Poblacion_hab'])
            
            if not v_disp.empty:
                micro_ver = st.selectbox("Vereda (Micro):", sorted(v_disp['Vereda'].unique()))
                pob_v = v_disp[v_disp['Vereda'] == micro_ver]['Poblacion_hab'].values[0]
                
                df_mp_raw = obtener_serie_historica(df_mpios, df_edades, "Municipal", macro_mpio, "Total")[0]
                pob_m = df_mp_raw['Poblacion'].iloc[-1] if not df_mp_raw.empty else pob_v * 10
                
                share_calc = (pob_v / pob_m) * 100
                st.metric("Población Veredal Actual", f"{pob_v:,.0f}")
                cuota_base = st.slider(f"% de participación en {macro_mpio}:", 0.0, 100.0, float(share_calc), step=0.1) / 100.0
                tendencia = st.radio("Comportamiento del Share:", ["Constante", "Decreciente (Migración a cabecera)"])
            else: st.warning("No hay veredas válidas.")
        else: st.error("Archivo de veredas requerido.")

    with col_a2:
        if not df_veredas.empty and not v_disp.empty and not df_mp_raw.empty:
            t_ani = np.arange(2024, 2060)
            pob_macro = pob_m * (1 + 0.015)**(t_ani - 2020) 
            
            if tendencia == "Constante": pob_micro = pob_macro * cuota_base
            else: pob_micro = pob_macro * np.linspace(cuota_base, cuota_base * 0.8, len(t_ani))
                
            fig_ani = go.Figure()
            fig_ani.add_trace(go.Scatter(x=t_ani, y=pob_macro, mode='lines', fill='tozeroy', name=f'Macro: {macro_mpio}', line=dict(color='#bdc3c7')))
            fig_ani.add_trace(go.Scatter(x=t_ani, y=pob_micro, mode='lines', fill='tozeroy', name=f'Micro: {micro_ver}', line=dict(color='#2ecc71', width=3)))
            fig_ani.update_layout(title="Proyección Anidada (Downscaling)", hovermode="x unified")
            st.plotly_chart(fig_ani, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 5: VISOR ESPACIAL (CENSOS DETALLADOS Y RANKINGS)
# ------------------------------------------------------------------------------
with tab_espacial:
    st.header("🗺️ Visor de Censos Detallados y Rankings")
    st.markdown("Generación dinámica de bases de datos departamentales y municipales con análisis de ranking (Top/Bottom).")
    
    col_v_sel1, col_v_sel2 = st.columns(2)
    with col_v_sel1:
        visor_sel = st.selectbox("Selecciona la base de datos a explorar:", ["Departamentos de Colombia (DANE)", "Municipios de Colombia (DANE)", "Veredas de Antioquia"])
    with col_v_sel2:
        orden_ranking = st.radio("Filtro de Ranking:", ["Top 15 (Mayor Población)", "Bottom 15 (Menor Población)"], horizontal=True)
        es_top = "Top 15" in orden_ranking
    
    st.divider()

    if visor_sel == "Departamentos de Colombia (DANE)":
        if not df_mpios.empty and 'depto_nom' in df_mpios.columns:
            # Eliminada la dependencia de 'id_dp' para evitar KeyErrors
            df_dept = df_mpios.groupby(['depto_nom', 'año', 'area_geografica'])['Poblacion'].sum().reset_index()
            st.success(f"Base de datos departamental generada dinámicamente ({len(df_dept)} registros).")
            
            col_v1, col_v2 = st.columns([2, 1.5])
            with col_v1: 
                st.dataframe(df_dept, use_container_width=True)
            with col_v2:
                max_anio = df_dept['año'].max()
                df_rank = df_dept[(df_dept['año'] == max_anio) & (df_dept['area_geografica'].str.lower().isin(['urbano', 'rural']))]
                df_rank = df_rank.groupby('depto_nom')['Poblacion'].sum().reset_index()
                
                df_plot = df_rank.nlargest(15, 'Poblacion') if es_top else df_rank.nsmallest(15, 'Poblacion')
                
                fig = px.bar(df_plot, x='Poblacion', y='depto_nom', orientation='h', color='Poblacion', title=f"Ranking Departamental ({max_anio})")
                fig.update_layout(yaxis={'categoryorder':'total ascending' if es_top else 'total descending'})
                st.plotly_chart(fig, use_container_width=True)
        else: st.warning("⚠️ No hay datos municipales para generar la vista departamental.")
            
    elif visor_sel == "Municipios de Colombia (DANE)":
        if not df_mpios.empty and 'municipio' in df_mpios.columns:
            st.success(f"Base de datos municipal cargada ({len(df_mpios)} registros).")
            
            col_v1, col_v2 = st.columns([2, 1.5])
            with col_v1: 
                st.dataframe(df_mpios, use_container_width=True)
            with col_v2:
                max_anio = df_mpios['año'].max()
                df_rank = df_mpios[(df_mpios['año'] == max_anio) & (df_mpios['area_geografica'].str.lower().isin(['urbano', 'rural']))]
                df_rank = df_rank.groupby('municipio')['Poblacion'].sum().reset_index()
                
                df_plot = df_rank.nlargest(15, 'Poblacion') if es_top else df_rank.nsmallest(15, 'Poblacion')
                
                fig = px.bar(df_plot, x='Poblacion', y='municipio', orientation='h', color='Poblacion', title=f"Ranking Municipal ({max_anio})")
                fig.update_layout(yaxis={'categoryorder':'total ascending' if es_top else 'total descending'})
                st.plotly_chart(fig, use_container_width=True)
        else: st.warning("⚠️ No se encontró el archivo de municipios.")
            
    elif visor_sel == "Veredas de Antioquia":
        if not df_veredas.empty:
            st.success(f"Base de datos veredal cargada ({len(df_veredas)} registros).")
            
            col_v1, col_v2 = st.columns([2, 1.5])
            with col_v1: 
                st.dataframe(df_veredas, use_container_width=True)
            with col_v2:
                if 'Poblacion_hab' in df_veredas.columns and 'Vereda' in df_veredas.columns:
                    df_ver_clean = df_veredas.dropna(subset=['Poblacion_hab'])
                    df_plot = df_ver_clean.nlargest(15, 'Poblacion_hab') if es_top else df_ver_clean.nsmallest(15, 'Poblacion_hab')
                    
                    fig_ver = px.bar(df_plot, x='Poblacion_hab', y='Vereda', orientation='h', color='Poblacion_hab', title="Ranking Veredal (Dato Reciente)")
                    fig_ver.update_layout(yaxis={'categoryorder':'total ascending' if es_top else 'total descending'})
                    st.plotly_chart(fig_ver, use_container_width=True)
        else: st.warning("⚠️ No se encontró el archivo de veredas.")
