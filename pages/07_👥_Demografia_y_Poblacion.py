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
Motor de Inferencia Demográfica: Agregación territorial dinámica, ajuste paramétrico automático (Curve Fitting), 
proyecciones polinómicas/exponenciales y estructura por edades con resolución veredal (Downscaling).
""")
st.divider()

# --- 1. LECTURA DE DATOS MAESTROS ---
def leer_csv_robusto(ruta):
    """Intenta leer con coma, si falla o detecta solo 1 columna, lee con punto y coma."""
    try:
        df = pd.read_csv(ruta, sep=',')
        if len(df.columns) < 2:
            df = pd.read_csv(ruta, sep=';')
        return df
    except Exception:
        try:
            return pd.read_csv(ruta, sep=';')
        except:
            return pd.DataFrame()

@st.cache_data
def cargar_municipios():
    ruta = "data/Pob_mpios_colombia.csv"
    if os.path.exists(ruta):
        return leer_csv_robusto(ruta)
    return pd.DataFrame()

@st.cache_data
def cargar_edades():
    ruta = "data/Pob_sexo_edad_Colombia_1950-2070.csv"
    if os.path.exists(ruta):
        return leer_csv_robusto(ruta)
    return pd.DataFrame()
    
# --- MOTOR DE AGREGACIÓN MATEMÁTICA ---
def obtener_serie_historica(df, nivel, nombre_lugar, area_geo):
    df_filtrado = df[df['area_geografica'].str.lower() == area_geo.lower()]
    
    if nivel == "Nacional (Colombia)":
        serie = df_filtrado.groupby('año')['Poblacion'].sum().reset_index()
        serie.rename(columns={'Poblacion': 'Colombia'}, inplace=True)
        return serie, 'Colombia'
    elif nivel == "Departamental":
        df_dept = df_filtrado[df_filtrado['depto_nom'] == nombre_lugar]
        serie = df_dept.groupby('año')['Poblacion'].sum().reset_index()
        serie.rename(columns={'Poblacion': nombre_lugar}, inplace=True)
        return serie, nombre_lugar
    elif nivel == "Municipal":
        df_mpio = df_filtrado[df_filtrado['municipio'] == nombre_lugar]
        serie = df_mpio.groupby('año')['Poblacion'].sum().reset_index()
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
    "🌍 4. Downscaling (Veredal)",
    "🗺️ 5. Visor Espacial"
])

# ------------------------------------------------------------------------------
# TAB 1: CENSOS HISTÓRICOS Y AGREGACIÓN
# ------------------------------------------------------------------------------
with tab_datos:
    st.header("📊 Evolución Histórica Territorial")
    
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        nivel_sel = st.selectbox("Nivel Territorial:", ["Nacional (Colombia)", "Departamental", "Municipal", "Veredal (Punto Estático)"])
    with col_t2:
        if nivel_sel == "Departamental":
            lugar_sel = st.selectbox("Departamento:", sorted(df_mpios['depto_nom'].unique()))
        elif nivel_sel == "Municipal":
            lugar_sel = st.selectbox("Municipio:", sorted(df_mpios['municipio'].unique()), index=sorted(df_mpios['municipio'].unique()).index('Medellín') if 'Medellín' in df_mpios['municipio'].values else 0)
        elif nivel_sel == "Veredal (Punto Estático)":
            if not df_veredas.empty:
                mpio_vereda = st.selectbox("Selecciona un Municipio para ver sus Veredas:", sorted(df_veredas['Municipio'].dropna().unique()))
                lugar_sel = mpio_vereda
            else:
                st.error("No se encontró el archivo de veredas.")
                lugar_sel = "N/A"
        else:
            lugar_sel = "Colombia"
            st.info("Escala Nacional Seleccionada")
            
    with col_t3:
        if nivel_sel != "Veredal (Punto Estático)":
            area_sel = st.selectbox("Área Geográfica:", ["Total", "Urbano", "Rural"])
        else:
            area_sel = "Rural"
            st.info("Escala Rural Fija")
        
    if nivel_sel == "Veredal (Punto Estático)":
        if not df_veredas.empty:
            df_v_filtrado = df_veredas[df_veredas['Municipio'] == lugar_sel].dropna(subset=['Poblacion_hab'])
            if not df_v_filtrado.empty:
                fig1 = px.bar(df_v_filtrado.sort_values('Poblacion_hab', ascending=False), x='Vereda', y='Poblacion_hab', title=f"Población por Veredas: {lugar_sel} (Dato más reciente)", color='Poblacion_hab')
                st.plotly_chart(fig1, use_container_width=True)
                st.info("💡 Como las veredas solo tienen un dato estático, usa el **Tab 4 (Downscaling)** para proyectar su crecimiento basado en el municipio.")
            else:
                st.warning("No hay datos de población veredal para este municipio.")
    else:
        df_plot, nombre_col = obtener_serie_historica(df_mpios, nivel_sel, lugar_sel, area_sel)
        if not df_plot.empty:
            fig1 = px.line(df_plot, x="año", y=nombre_col, title=f"Crecimiento Histórico: {nombre_col} ({area_sel})", markers=True)
            st.plotly_chart(fig1, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: MODELOS Y OPTIMIZACIÓN MATEMÁTICA
# ------------------------------------------------------------------------------
with tab_modelos:
    st.header("📈 Ajuste de Modelos Evolutivos")
    if nivel_sel == "Veredal (Punto Estático)":
        st.error("❌ El motor de optimización requiere una serie de tiempo. Las veredas solo tienen un dato puntual. Usa el nivel Municipal para modelar y el Tab 4 para anidar la vereda.")
    else:
        col_opt1, col_opt2 = st.columns([1, 2.5])
        with col_opt1:
            st.subheader("Configuración")
            st.info(f"Modelando: **{lugar_sel} ({area_sel})**")
            
            t_data_raw = df_plot["año"].values
            p_data = df_plot[nombre_col].values
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
    st.header("🏗️ Pirámides Poblacionales (1950 - 2070)")
    col_p1, col_p2 = st.columns([1, 3])
    
    with col_p1:
        area_pir = st.selectbox("Área:", df_edades['area_geografica'].unique())
        anio_pir = st.slider("Año de la Pirámide:", int(df_edades['año'].min()), int(df_edades['año'].max()), 2024)
        
        df_f_pir = df_edades[(df_edades['año'] == anio_pir) & (df_edades['area_geografica'] == area_pir)]
        
    with col_p2:
        if not df_f_pir.empty:
            edades_cols = [str(i) for i in range(101)]
            try:
                hombres = df_f_pir[df_f_pir['sexo'].str.lower() == 'hombres'][edades_cols].values.flatten()
                mujeres = df_f_pir[df_f_pir['sexo'].str.lower() == 'mujeres'][edades_cols].values.flatten()
                
                fig_pir = go.Figure()
                fig_pir.add_trace(go.Bar(y=edades_cols, x=hombres * -1, name='Hombres', orientation='h', marker=dict(color='#3498db'), hovertext=hombres))
                fig_pir.add_trace(go.Bar(y=edades_cols, x=mujeres, name='Mujeres', orientation='h', marker=dict(color='#e74c3c'), hovertext=mujeres))
                fig_pir.update_layout(title=f"Estructura Demográfica ({anio_pir})", barmode='relative', yaxis_title='Edad Simple', xaxis_title='Población', height=600)
                st.plotly_chart(fig_pir, use_container_width=True)
            except Exception as e:
                st.error("Error al graficar la pirámide. Revisa la estructura del archivo de edades.")

# ------------------------------------------------------------------------------
# TAB 4: DOWNSCALING (EL PUENTE A LAS VEREDAS)
# ------------------------------------------------------------------------------
with tab_anidados:
    st.header("🌍 Modelos Anidados: Municipio ➔ Vereda")
    st.markdown("Calcula la proyección futura de una vereda basándose en la tendencia de su municipio anfitrión.")
    
    col_a1, col_a2 = st.columns([1, 2])
    with col_a1:
        if not df_veredas.empty:
            macro_mpio = st.selectbox("Municipio Anfitrión (Macro):", sorted(df_veredas['Municipio'].dropna().unique()))
            v_disp = df_veredas[df_veredas['Municipio'] == macro_mpio].dropna(subset=['Poblacion_hab'])
            
            if not v_disp.empty:
                micro_ver = st.selectbox("Vereda (Micro):", sorted(v_disp['Vereda'].unique()))
                pob_v = v_disp[v_disp['Vereda'] == micro_ver]['Poblacion_hab'].values[0]
                
                # Buscar población municipal más reciente
                df_mp = obtener_serie_historica(df_mpios, "Municipal", macro_mpio, "Total")[0]
                pob_m = df_mp['Poblacion'].iloc[-1] if not df_mp.empty else pob_v * 10
                
                share_calc = (pob_v / pob_m) * 100
                st.metric("Población Veredal Actual", f"{pob_v:,.0f}")
                cuota_base = st.slider(f"% de participación en {macro_mpio}:", 0.0, 100.0, float(share_calc), step=0.1) / 100.0
                
                # Simulador de tendencia
                tendencia = st.radio("Comportamiento del Share:", ["Constante", "Decreciente (Migración a cabecera)"])
            else:
                st.warning("No hay veredas válidas.")
        else:
            st.error("Sube el archivo de veredas a GitHub para activar este módulo.")

    with col_a2:
        if not df_veredas.empty and not v_disp.empty and not df_mp.empty:
            t_ani = np.arange(2024, 2060)
            # Curva macro simulada basada en el último dato
            pob_macro = pob_m * (1 + 0.015)**(t_ani - 2020) 
            
            if tendencia == "Constante": pob_micro = pob_macro * cuota_base
            else: pob_micro = pob_macro * np.linspace(cuota_base, cuota_base * 0.8, len(t_ani))
                
            fig_ani = go.Figure()
            fig_ani.add_trace(go.Scatter(x=t_ani, y=pob_macro, mode='lines', fill='tozeroy', name=f'Macro: {macro_mpio}', line=dict(color='#bdc3c7')))
            fig_ani.add_trace(go.Scatter(x=t_ani, y=pob_micro, mode='lines', fill='tozeroy', name=f'Micro: {micro_ver}', line=dict(color='#2ecc71', width=3)))
            fig_ani.update_layout(title="Proyección Anidada (Downscaling)", hovermode="x unified")
            st.plotly_chart(fig_ani, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 5: VISOR ESPACIAL (CENSOS DETALLADOS DANE Y VEREDAL)
# ------------------------------------------------------------------------------
with tab_espacial:
    st.header("🗺️ Visor de Censos Detallados (DANE y Veredal)")
    st.markdown("Exploración de bases de datos detalladas por Departamento, Municipio y Vereda para análisis cruzado.")
    
    visor_sel = st.selectbox(
        "Selecciona la base de datos a explorar:", 
        ["Departamentos de Colombia (DANE)", "Municipios de Colombia (DANE)", "Veredas de Antioquia"]
    )
    
    if visor_sel == "Departamentos de Colombia (DANE)":
        ruta_dept = "data/Departamentos_Colombia.xlsx"
        if os.path.exists(ruta_dept):
            # skiprows=7 ignora los títulos iniciales del DANE
            df_dept = pd.read_excel(ruta_dept, skiprows=7) 
            st.success(f"Base de datos departamental cargada ({len(df_dept)} registros).")
            st.dataframe(df_dept, use_container_width=True)
        else:
            st.warning(f"⚠️ No se encontró el archivo en GitHub. Sube el archivo con el nombre exacto: `Departamentos_Colombia.xlsx` a la carpeta `data/`")
            
    elif visor_sel == "Municipios de Colombia (DANE)":
        ruta_mpio = "data/Municipios_Colombia.xlsx"
        if os.path.exists(ruta_mpio):
            # skiprows=5 ignora los títulos iniciales del DANE
            df_mpio = pd.read_excel(ruta_mpio, skiprows=5)
            st.success(f"Base de datos municipal cargada ({len(df_mpio)} registros).")
            st.dataframe(df_mpio, use_container_width=True)
        else:
            st.warning(f"⚠️ No se encontró el archivo en GitHub. Sube el archivo con el nombre exacto: `Municipios_Colombia.xlsx` a la carpeta `data/`")
            
    elif visor_sel == "Veredas de Antioquia":
        ruta_ver = "data/veredas_Antioquia.xlsx"
        if os.path.exists(ruta_ver):
            df_ver = pd.read_excel(ruta_ver)
            st.success(f"Base de datos veredal cargada ({len(df_ver)} registros).")
            
            col_v1, col_v2 = st.columns([2, 1])
            with col_v1:
                st.dataframe(df_ver, use_container_width=True)
            with col_v2:
                # Mini-gráfico automático para las Veredas
                if 'Poblacion_hab' in df_ver.columns and 'Vereda' in df_ver.columns:
                    st.subheader("📊 Top 15 Veredas")
                    df_ver_clean = df_ver.dropna(subset=['Poblacion_hab'])
                    df_top = df_ver_clean.sort_values(by='Poblacion_hab', ascending=False).head(15)
                    fig_ver = px.bar(df_top, x='Poblacion_hab', y='Vereda', orientation='h', color='Municipio')
                    fig_ver.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, height=400)
                    st.plotly_chart(fig_ver, use_container_width=True)
        else:
            st.warning(f"⚠️ No se encontró el archivo en GitHub. Sube el archivo con el nombre exacto: `veredas_Antioquia.xlsx` a la carpeta `data/`")
