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
Motor de Inferencia Demográfica: Agregación territorial dinámica, ajuste paramétrico, proyecciones polinómicas, comparativas multiescalar y estructura por edades.
""")
st.divider()

# --- 1. LECTURA DE DATOS MAESTROS ---
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
    # Estandarizar nombre de columna de año
    if 'Año' in df.columns: df.rename(columns={'Año': 'año'}, inplace=True)
    return df

@st.cache_data
def cargar_municipios():
    ruta = "data/Pob_mpios_colombia.csv"
    if os.path.exists(ruta):
        df = leer_csv_robusto(ruta)
        if not df.empty and 'municipio' in df.columns:
            df.dropna(subset=['municipio'], inplace=True)
            if 'region' in df.columns: df['region'] = df['region'].astype(str).str.strip()
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
        areas_str = df_ed['area_geografica'].astype(str).str.lower()
        if area_geo.lower() == "total":
            df_f = df_ed[areas_str.str.contains('cabecera|urbano|rural|centro', na=False)].copy()
            if df_f.empty: df_f = df_ed.copy()
        elif area_geo.lower() == "urbano":
            df_f = df_ed[areas_str.str.contains('cabecera|urbano', na=False)].copy()
        else:
            df_f = df_ed[areas_str.str.contains('centro|rural|resto', na=False)].copy()
            
        if df_f.empty: return pd.DataFrame(), 'Colombia'
        sexo_str = df_f['sexo'].astype(str).str.lower()
        if sexo_str.isin(['hombres', 'mujeres']).any(): df_f = df_f[sexo_str.isin(['hombres', 'mujeres'])]
        else: df_f = df_f[sexo_str == 'total']
            
        edades_cols = [str(i) for i in range(101) if str(i) in df_f.columns]
        df_f[edades_cols] = df_f[edades_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        df_f['Poblacion'] = df_f[edades_cols].sum(axis=1)
        
        serie = df_f.groupby('año')['Poblacion'].sum().reset_index()
        serie.rename(columns={'Poblacion': f'Colombia ({area_geo})'}, inplace=True)
        return serie, f'Colombia ({area_geo})'
        
    else: 
        if df_mp.empty: return pd.DataFrame(), nombre_lugar
        areas_str_mp = df_mp['area_geografica'].astype(str).str.lower()
        if area_geo.lower() == "total":
            areas_validas = ['urbano', 'rural', 'cabecera', 'resto', 'centros poblados y rural disperso']
            df_f = df_mp[areas_str_mp.isin(areas_validas)]
            if df_f.empty: df_f = df_mp 
        elif area_geo.lower() == "urbano":
            df_f = df_mp[areas_str_mp.str.contains('urbano|cabecera', na=False)]
        else:
            df_f = df_mp[areas_str_mp.str.contains('rural|resto|centro', na=False)]
            
        col_name = f"{nombre_lugar} ({area_geo})"
        if nivel == "Regional":
            if 'region' not in df_f.columns: return pd.DataFrame(), col_name
            serie = df_f[df_f['region'] == nombre_lugar].groupby('año')['Poblacion'].sum().reset_index()
            serie.rename(columns={'Poblacion': col_name}, inplace=True)
            return serie, col_name
        elif nivel == "Departamental":
            if 'depto_nom' not in df_f.columns: return pd.DataFrame(), col_name
            serie = df_f[df_f['depto_nom'] == nombre_lugar].groupby('año')['Poblacion'].sum().reset_index()
            serie.rename(columns={'Poblacion': col_name}, inplace=True)
            return serie, col_name
        elif nivel == "Municipal":
            serie = df_f[df_f['municipio'] == nombre_lugar].groupby('año')['Poblacion'].sum().reset_index()
            serie.rename(columns={'Poblacion': col_name}, inplace=True)
            return serie, col_name
            
    return pd.DataFrame(), nombre_lugar

# ==============================================================================
# PESTAÑAS
# ==============================================================================
tab_datos, tab_modelos, tab_piramides, tab_anidados, tab_espacial = st.tabs([
    "📊 1. Censos Históricos", 
    "📈 2. Modelos & Optimización", 
    "🏗️ 3. Pirámides",
    "🌍 4. Downscaling",
    "🗺️ 5. Visor Espacial"
])

df_plot_model = pd.DataFrame()
nombre_col_model = ""
col_anio_model = "año"

# ------------------------------------------------------------------------------
# TAB 1: CENSOS HISTÓRICOS Y COMPARATIVAS
# ------------------------------------------------------------------------------
with tab_datos:
    st.header("📊 Evolución Histórica Territorial")
    
    # --- SECCIÓN A: COMPARATIVA MULTIESCALAR ---
    with st.expander("🌍 Comparador Multiescalar (Ver varias unidades a la vez)", expanded=False):
        st.markdown("Selecciona múltiples unidades territoriales de diferentes escalas para compararlas en un solo gráfico.")
        c1, c2, c3 = st.columns(3)
        with c1:
            macros_disp = [c for c in df_macro.columns if c.lower() != 'año'] if not df_macro.empty else []
            sel_macros = st.multiselect("Zonas Macro / Mundiales:", macros_disp)
        with c2:
            deptos_disp = sorted([str(x) for x in df_mpios['depto_nom'].unique() if pd.notna(x)]) if not df_mpios.empty and 'depto_nom' in df_mpios.columns else []
            sel_deptos = st.multiselect("Departamentos:", deptos_disp)
        with c3:
            mpios_disp = sorted([str(x) for x in df_mpios['municipio'].unique() if pd.notna(x)]) if not df_mpios.empty and 'municipio' in df_mpios.columns else []
            sel_mpios = st.multiselect("Municipios:", mpios_disp)
            
        if sel_macros or sel_deptos or sel_mpios:
            df_comp = pd.DataFrame(columns=['año'])
            
            # Agregar Macros
            for m in sel_macros:
                temp = df_macro[['año', m]].dropna()
                df_comp = pd.merge(df_comp, temp, on='año', how='outer') if not df_comp.empty else temp
            
            # Agregar Departamentos (Total por defecto para comparativas)
            for d in sel_deptos:
                temp, col_n = obtener_serie_historica(df_mpios, df_edades, "Departamental", d, "Total")
                if not temp.empty:
                    df_comp = pd.merge(df_comp, temp, on='año', how='outer') if not df_comp.empty else temp
                    
            # Agregar Municipios
            for mp in sel_mpios:
                temp, col_n = obtener_serie_historica(df_mpios, df_edades, "Municipal", mp, "Total")
                if not temp.empty:
                    df_comp = pd.merge(df_comp, temp, on='año', how='outer') if not df_comp.empty else temp
            
            df_comp = df_comp.sort_values('año').reset_index(drop=True)
            cols_to_plot = [c for c in df_comp.columns if c != 'año']
            fig_comp = px.line(df_comp, x="año", y=cols_to_plot, markers=True, title="Comparativa Territorial (Área Total)")
            fig_comp.update_layout(yaxis_title="Población", legend_title="Unidades")
            st.plotly_chart(fig_comp, use_container_width=True)
            
    st.divider()

    # --- SECCIÓN B: ANÁLISIS INDIVIDUAL PROFUNDO (Múltiples Áreas) ---
    st.subheader("Análisis Detallado por Unidad")
    col_t1, col_t2, col_t3 = st.columns(3)
    
    with col_t1:
        escalas = ["Macro (Mundial/Histórico)", "Nacional (Colombia)", "Regional", "Departamental", "Municipal", "Veredal"]
        nivel_sel = st.selectbox("Nivel Territorial:", escalas)
        
    with col_t2:
        lugar_sel = "N/A"
        if nivel_sel == "Macro (Mundial/Histórico)":
            if not df_macro.empty:
                opciones = [col for col in df_macro.columns if col.lower() != 'año']
                lugar_sel = st.selectbox("Zona Macro/Global:", opciones)
            else: st.error("Archivo Macro no encontrado.")
            
        elif nivel_sel == "Regional":
            if not df_mpios.empty and 'region' in df_mpios.columns and 'depto_nom' in df_mpios.columns:
                # Lógica Vinculada: Departamento -> Regiones
                depto_reg_sel = st.selectbox("1. Selecciona el Departamento:", sorted([str(x) for x in df_mpios['depto_nom'].unique() if pd.notna(x)]))
                regs_disponibles = df_mpios[df_mpios['depto_nom'] == depto_reg_sel]['region'].dropna().unique()
                if len(regs_disponibles) > 0:
                    lugar_sel = st.selectbox("2. Selecciona la Región:", sorted(regs_disponibles))
                else:
                    st.warning("No hay regiones definidas para este departamento.")
            else: st.warning("Columna 'region' o 'depto_nom' no encontrada.")
            
        elif nivel_sel == "Departamental":
            if not df_mpios.empty and 'depto_nom' in df_mpios.columns:
                opciones = sorted([str(x) for x in df_mpios['depto_nom'].unique() if pd.notna(x)])
                lugar_sel = st.selectbox("Departamento:", opciones)
                
        elif nivel_sel == "Municipal":
            if not df_mpios.empty and 'municipio' in df_mpios.columns:
                opciones = sorted([str(x) for x in df_mpios['municipio'].unique() if pd.notna(x)])
                idx = opciones.index('Medellín') if 'Medellín' in opciones else 0
                lugar_sel = st.selectbox("Municipio:", opciones, index=idx)
                
        elif nivel_sel == "Veredal":
            if not df_veredas.empty:
                mpio_vereda = st.selectbox("Municipio para ver Veredas:", sorted(df_veredas['Municipio'].dropna().unique()))
                lugar_sel = mpio_vereda
                
        else:
            lugar_sel = "Colombia"
            st.info("Extraído desde BD Edades (1950-2070)")
            
    with col_t3:
        if nivel_sel == "Macro (Mundial/Histórico)":
            areas_sel = ["Total"]
            st.info("Las bases globales solo reportan valores Totales.")
        elif nivel_sel == "Veredal":
            areas_sel = ["Rural"]
            st.info("Nivel estrictamente Rural.")
        else:
            # MAGIA: Multiselect para ver Total, Urbano y Rural en el mismo gráfico
            areas_sel = st.multiselect("Área(s) Geográfica(s) a graficar:", ["Total", "Urbano", "Rural"], default=["Total"])
        
    # --- GRAFICADOR MULTI-LÍNEA ---
    if areas_sel and lugar_sel != "N/A":
        if nivel_sel == "Macro (Mundial/Histórico)":
            if not df_macro.empty:
                df_plot_indiv = df_macro[["año", lugar_sel]].dropna()
                fig1 = px.line(df_plot_indiv, x="año", y=lugar_sel, title=f"Historia de Largo Plazo: {lugar_sel}", markers=True)
                st.plotly_chart(fig1, use_container_width=True)
                df_plot_model = df_plot_indiv 
                nombre_col_model = lugar_sel
                
        elif nivel_sel == "Veredal":
            if not df_veredas.empty:
                df_v_filtrado = df_veredas[df_veredas['Municipio'] == lugar_sel].dropna(subset=['Poblacion_hab'])
                if not df_v_filtrado.empty:
                    fig1 = px.bar(df_v_filtrado.sort_values('Poblacion_hab', ascending=False), x='Vereda', y='Poblacion_hab', title=f"Población Veredal: {lugar_sel}")
                    st.plotly_chart(fig1, use_container_width=True)
                
        else: 
            # Combinar las áreas seleccionadas
            df_plot_indiv = pd.DataFrame(columns=['año'])
            cols_graficar = []
            
            for ar in areas_sel:
                temp, col_n = obtener_serie_historica(df_mpios, df_edades, nivel_sel, lugar_sel, ar)
                if not temp.empty:
                    df_plot_indiv = pd.merge(df_plot_indiv, temp, on='año', how='outer') if not df_plot_indiv.empty else temp
                    cols_graficar.append(col_n)
            
            if not df_plot_indiv.empty:
                df_plot_indiv = df_plot_indiv.sort_values('año')
                fig1 = px.line(df_plot_indiv, x="año", y=cols_graficar, title=f"Crecimiento Histórico: {lugar_sel}", markers=True)
                fig1.update_layout(yaxis_title="Población", legend_title="Áreas")
                st.plotly_chart(fig1, use_container_width=True)
                
                # Pasamos la primera área seleccionada al modelo matemático
                df_plot_model = df_plot_indiv[['año', cols_graficar[0]]].dropna()
                nombre_col_model = cols_graficar[0]

# ------------------------------------------------------------------------------
# TAB 2: MODELOS Y OPTIMIZACIÓN MATEMÁTICA
# ------------------------------------------------------------------------------
with tab_modelos:
    st.header("📈 Ajuste de Modelos Evolutivos")
    
    if df_plot_model.empty or not nombre_col_model:
        st.info("👆 Selecciona una escala válida en el Tab 1 que contenga datos de serie de tiempo.")
    elif nivel_sel == "Veredal":
        st.error(f"❌ La escala Veredal no posee historia validada para modelación pura. Usa el Tab 4.")
    else:
        col_opt1, col_opt2 = st.columns([1, 2.5])
        with col_opt1:
            st.subheader("Configuración")
            st.success(f"Modelando base principal: **{nombre_col_model}**")
            st.caption("(Si elegiste varias áreas en el Tab 1, el modelo tomará la primera de la lista para la optimización).")
            
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
                    df_t = df_f_pir[df_f_pir['sexo'].isin(['hombres', 'mujeres'])][edades_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
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
    st.markdown("Generación dinámica de bases de datos territoriales con análisis de ranking.")
    
    col_v_sel1, col_v_sel2 = st.columns(2)
    with col_v_sel1:
        visor_sel = st.selectbox("Selecciona la base de datos a explorar:", [
            "Departamentos de Colombia (DANE)", 
            "Regiones (Agrupación Municipal)", 
            "Municipios de Colombia (DANE)", 
            "Veredas de Antioquia"
        ])
    with col_v_sel2:
        orden_ranking = st.radio("Filtro de Ranking:", ["Top 15 (Mayor Población)", "Bottom 15 (Menor Población)"], horizontal=True)
        es_top = "Top 15" in orden_ranking
    
    st.divider()

    if visor_sel == "Departamentos de Colombia (DANE)":
        if not df_mpios.empty and 'depto_nom' in df_mpios.columns:
            df_dept = df_mpios.groupby(['depto_nom', 'año', 'area_geografica'])['Poblacion'].sum().reset_index()
            col_v1, col_v2 = st.columns([2, 1.5])
            with col_v1: 
                st.dataframe(df_dept, use_container_width=True)
            with col_v2:
                max_anio = df_dept['año'].max()
                df_rank = df_dept[(df_dept['año'] == max_anio) & (df_dept['area_geografica'].str.lower().isin(['urbano', 'rural', 'cabecera', 'resto']))]
                df_rank = df_rank.groupby('depto_nom')['Poblacion'].sum().reset_index()
                df_plot = df_rank.nlargest(15, 'Poblacion') if es_top else df_rank.nsmallest(15, 'Poblacion')
                
                fig = px.bar(df_plot, x='Poblacion', y='depto_nom', orientation='h', color='Poblacion', title=f"Ranking Departamental ({max_anio})")
                fig.update_layout(yaxis={'categoryorder':'total ascending' if es_top else 'total descending'})
                st.plotly_chart(fig, use_container_width=True)
        else: st.warning("Datos municipales insuficientes.")
            
    elif visor_sel == "Regiones (Agrupación Municipal)":
        if not df_mpios.empty and 'region' in df_mpios.columns and 'depto_nom' in df_mpios.columns:
            # Lógica jerárquica: Agrupamos por depto y region
            df_reg = df_mpios.groupby(['depto_nom', 'region', 'año', 'area_geografica'])['Poblacion'].sum().reset_index()
            col_v1, col_v2 = st.columns([2, 1.5])
            with col_v1: 
                st.dataframe(df_reg, use_container_width=True)
            with col_v2:
                max_anio = df_reg['año'].max()
                df_rank = df_reg[(df_reg['año'] == max_anio) & (df_reg['area_geografica'].str.lower().isin(['urbano', 'rural', 'cabecera', 'resto']))]
                
                # Crear etiqueta combinada: Región (Depto)
                df_rank['region_label'] = df_rank['region'] + " (" + df_rank['depto_nom'] + ")"
                df_rank = df_rank.groupby('region_label')['Poblacion'].sum().reset_index()
                
                df_plot = df_rank.nlargest(15, 'Poblacion') if es_top else df_rank.nsmallest(15, 'Poblacion')
                fig = px.bar(df_plot, x='Poblacion', y='region_label', orientation='h', color='Poblacion', title=f"Ranking Regional ({max_anio})")
                fig.update_layout(yaxis={'categoryorder':'total ascending' if es_top else 'total descending'})
                st.plotly_chart(fig, use_container_width=True)
        else: st.warning("Columna 'region' o 'depto_nom' no encontrada.")

    elif visor_sel == "Municipios de Colombia (DANE)":
        if not df_mpios.empty and 'municipio' in df_mpios.columns:
            col_v1, col_v2 = st.columns([2, 1.5])
            with col_v1: 
                st.dataframe(df_mpios, use_container_width=True)
            with col_v2:
                max_anio = df_mpios['año'].max()
                df_rank = df_mpios[(df_mpios['año'] == max_anio) & (df_mpios['area_geografica'].str.lower().isin(['urbano', 'rural', 'cabecera', 'resto']))]
                
                label_col = 'municipio'
                if 'depto_nom' in df_rank.columns:
                    df_rank['mpio_label'] = df_rank['municipio'] + " (" + df_rank['depto_nom'] + ")"
                    label_col = 'mpio_label'
                    
                df_rank = df_rank.groupby(label_col)['Poblacion'].sum().reset_index()
                df_plot = df_rank.nlargest(15, 'Poblacion') if es_top else df_rank.nsmallest(15, 'Poblacion')
                
                fig = px.bar(df_plot, x='Poblacion', y=label_col, orientation='h', color='Poblacion', title=f"Ranking Municipal ({max_anio})")
                fig.update_layout(yaxis={'categoryorder':'total ascending' if es_top else 'total descending'})
                st.plotly_chart(fig, use_container_width=True)
        else: st.warning("Archivo de municipios no válido.")
            
    elif visor_sel == "Veredas de Antioquia":
        if not df_veredas.empty:
            col_v1, col_v2 = st.columns([2, 1.5])
            with col_v1: 
                st.dataframe(df_veredas, use_container_width=True)
            with col_v2:
                if 'Poblacion_hab' in df_veredas.columns and 'Vereda' in df_veredas.columns:
                    df_ver_clean = df_veredas.dropna(subset=['Poblacion_hab'])
                    
                    label_col = 'Vereda'
                    if 'Municipio' in df_ver_clean.columns:
                        df_ver_clean['ver_label'] = df_ver_clean['Vereda'] + " (" + df_ver_clean['Municipio'].astype(str) + ")"
                        label_col = 'ver_label'
                        
                    df_plot = df_ver_clean.nlargest(15, 'Poblacion_hab') if es_top else df_ver_clean.nsmallest(15, 'Poblacion_hab')
                    fig_ver = px.bar(df_plot, x='Poblacion_hab', y=label_col, orientation='h', color='Poblacion_hab', title="Ranking Veredal")
                    fig_ver.update_layout(yaxis={'categoryorder':'total ascending' if es_top else 'total descending'})
                    st.plotly_chart(fig_ver, use_container_width=True)
        else: st.warning("Archivo de veredas no encontrado.")
