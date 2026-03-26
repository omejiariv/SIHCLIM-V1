# 06a_🐄_Modelo_Pecuario.py

import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go

# =====================================================================
# CONFIGURACIÓN DE PÁGINA Y CARGA DE DATOS
# =====================================================================
st.set_page_config(page_title="Modelo Demográfico Pecuario", page_icon="🐄", layout="wide")

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
Este motor lee la historia de los censos del ICA (2018-2025) distribuida en las cuencas y entrena 
tres modelos matemáticos predictivos para proyectar la carga contaminante animal hacia el futuro.
""")

if st.button("⚙️ Iniciar Entrenamiento Multimodelo Pecuario", type="primary"):
    with st.spinner("Entrenando modelos para Bovinos, Porcinos y Aves en todas las escalas... Esto tomará unos segundos."):
        
        # --- FUNCIONES MATEMÁTICAS ---
        def f_log(t, k, a, r): return k / (1 + a * np.exp(-r * t))
        def f_exp(t, a, b): return a * np.exp(b * t)
        
        def calcular_r2(y_real, y_pred):
            ss_res = np.sum((y_real - y_pred) ** 2)
            ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        matriz_resultados = []
        
        # --- FUNCIÓN ENTRENADORA ---
        def ajustar_modelos(x, y, nivel, territorio, especie):
            # Filtro de seguridad: Si no hay animales en toda la historia, saltamos
            if len(x) < 4 or max(y) <= 0: return 
            
            x_offset = x[0]
            x_norm = x - x_offset
            p0_val = max(1, y[0])
            max_y = max(y)
            es_creciente = y[-1] >= p0_val
            
            # 1. LOGÍSTICO
            log_k, log_a, log_r, log_r2 = 0, 0, 0, 0
            try:
                k_max = max_y * 3.0 if es_creciente else max_y * 1.1
                # Si el hato decrece, apuntamos la proyección hacia abajo
                k_guess = max_y * 1.2 if es_creciente else (y[-1] * 0.9 if y[-1] > 0 else max_y)
                a_guess = (k_guess - p0_val) / p0_val if p0_val > 0 else 1
                r_guess = 0.02 if es_creciente else -0.02
                    
                # 🛡️ EL BISTURÍ: Permitir que la capacidad de carga baje hasta un 10% si hay reducción ganadera
                k_min = max_y * 0.8 if es_creciente else max_y * 0.1
                limites = ([k_min, 0, -0.2], [k_max, np.inf, 0.3])
                    
                popt_log, _ = curve_fit(f_log, x_norm, y, p0=[k_guess, a_guess, r_guess], bounds=limites, maxfev=50000)
                log_k, log_a, log_r = popt_log
                log_r2 = calcular_r2(y, f_log(x_norm, *popt_log))
            except Exception: pass

            # 2. EXPONENCIAL
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

            # JUEZ: MEJOR MODELO
            dic_modelos = {'Logístico': log_r2, 'Exponencial': exp_r2, 'Polinomial_3': poly_r2}
            mejor_modelo = max(dic_modelos, key=dic_modelos.get)
            mejor_r2 = dic_modelos[mejor_modelo]

            matriz_resultados.append({
                'Especie': especie, 
                'Nivel': nivel, 'Territorio': territorio,
                'Año_Base': int(x_offset), 'Poblacion_Base': round(p0_val, 0),
                'Log_K': log_k, 'Log_a': log_a, 'Log_r': log_r, 'Log_R2': round(log_r2, 4),
                'Exp_a': exp_a, 'Exp_b': exp_b, 'Exp_R2': round(exp_r2, 4),
                'Poly_A': poly_A, 'Poly_B': poly_B, 'Poly_C': poly_C, 'Poly_D': poly_D, 'Poly_R2': round(poly_r2, 4),
                'Modelo_Recomendado': mejor_modelo, 'Mejor_R2': round(mejor_r2, 4)
            })

        # --- PREPARACIÓN DE DATOS Y AGRUPACIONES ---
        # 1. Nivel Departamental (Antioquia Completo)
        df_depto = df_pecuario.groupby('Anio')[['Bovinos', 'Porcinos', 'Aves']].sum().reset_index()
        for especie in ['Bovinos', 'Porcinos', 'Aves']:
            ajustar_modelos(df_depto['Anio'].values, df_depto[especie].values, 'Departamental', 'Antioquia', especie)

        # 2. Nivel Municipal
        df_mpios = df_pecuario.groupby(['Anio', 'Municipio_Norm'])[['Bovinos', 'Porcinos', 'Aves']].sum().reset_index()
        for mpio in df_mpios['Municipio_Norm'].unique():
            df_temp = df_mpios[df_mpios['Municipio_Norm'] == mpio].sort_values(by='Anio')
            for especie in ['Bovinos', 'Porcinos', 'Aves']:
                ajustar_modelos(df_temp['Anio'].values, df_temp[especie].values, 'Municipal', mpio, especie)

        # 3. Nivel Subcuenca
        df_subcuencas = df_pecuario.groupby(['Anio', 'Subcuenca'])[['Bovinos', 'Porcinos', 'Aves']].sum().reset_index()
        for sub in df_subcuencas['Subcuenca'].unique():
            if sub == "No Definido": continue
            df_temp = df_subcuencas[df_subcuencas['Subcuenca'] == sub].sort_values(by='Anio')
            for especie in ['Bovinos', 'Porcinos', 'Aves']:
                ajustar_modelos(df_temp['Anio'].values, df_temp[especie].values, 'Subcuenca', sub, especie)
                
        # 4. Nivel Sistema Hídrico
        df_sistemas = df_pecuario.groupby(['Anio', 'Sistema'])[['Bovinos', 'Porcinos', 'Aves']].sum().reset_index()
        for sis in df_sistemas['Sistema'].unique():
            if sis == "No Definido": continue
            df_temp = df_sistemas[df_sistemas['Sistema'] == sis].sort_values(by='Anio')
            for especie in ['Bovinos', 'Porcinos', 'Aves']:
                ajustar_modelos(df_temp['Anio'].values, df_temp[especie].values, 'Sistema Hidrico', sis, especie)

        # ==============================================================================
        # 💾 EXPORTACIÓN AUTOMÁTICA A SQL (1 CLIC A PRODUCCIÓN)
        # ==============================================================================
        df_matriz_pec = pd.DataFrame(matriz_resultados)
        st.session_state['df_matriz_pecuaria'] = df_matriz_pec 
        
        try:
            from modules.db_manager import get_engine
            engine_sql = get_engine()
            
            # Escribir directamente en la base de datos PostgreSQL
            df_matriz_pec.to_sql('matriz_maestra_pecuaria', engine_sql, if_exists='replace', index=False)
            
            st.success(f"✅ ¡Migración a SQL Exitosa! Matriz Pecuaria ({len(df_matriz_pec)} modelos) alojada automáticamente en la base de datos maestra.")
        except Exception as e:
            st.error(f"Error conectando a la base de datos SQL: {e}")
        
        # Mantenemos el botón de descarga solo como respaldo
        csv_matriz = df_matriz_pec.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Descargar Respaldo (CSV)", data=csv_matriz, file_name="Matriz_Multimodelo_Pecuaria.csv", mime='text/csv')
        
# =====================================================================
# 🔬 VALIDADOR VISUAL COMPARATIVO (DOBLE VENTANA)
# =====================================================================
if 'df_matriz_pecuaria' in st.session_state:
    st.divider()
    st.subheader("🔬 Validador Visual Pecuario (Especie vs Especie)")
    
    df_mat = st.session_state['df_matriz_pecuaria']
    
    c_nav1, c_nav2, c_nav3 = st.columns([1, 1.5, 1])
    with c_nav1:
        niveles_disp = list(df_mat['Nivel'].unique())
        idx_mun = niveles_disp.index('Subcuenca') if 'Subcuenca' in niveles_disp else 0
        nivel_val = st.selectbox("1. Escala Espacial:", niveles_disp, index=idx_mun)
    with c_nav2:
        territorios_disp = sorted(df_mat[df_mat['Nivel'] == nivel_val]['Territorio'].unique())
        idx_terr = territorios_disp.index('R. Chico') if 'R. Chico' in territorios_disp else 0
        terr_val = st.selectbox("2. Territorio Hídrico / Administrativo:", territorios_disp, index=idx_terr)
    with c_nav3:
        anio_futuro = st.slider("3. Proyectar hasta el año:", min_value=2025, max_value=2050, value=2035, step=1)
        
    st.markdown("---")
    
    def renderizar_panel_pecuario(especie_sel, key_suffix):
        df_filtrado = df_mat[(df_mat['Nivel'] == nivel_val) & (df_mat['Territorio'] == terr_val) & (df_mat['Especie'] == especie_sel)]
        if df_filtrado.empty:
            st.warning(f"No hay registros o modelos viables para {especie_sel} en {terr_val}.")
            return
            
        fila_terr = df_filtrado.iloc[0]
        mejor_modelo = fila_terr['Modelo_Recomendado']
        
        # Reconstruir Histórico
        if nivel_val == 'Departamental':
            df_hist = df_pecuario.groupby('Anio')[especie_sel].sum().reset_index()
        elif nivel_val == 'Municipal':
            df_hist = df_pecuario[df_pecuario['Municipio_Norm'] == terr_val].groupby('Anio')[especie_sel].sum().reset_index()
        elif nivel_val == 'Subcuenca':
            df_hist = df_pecuario[df_pecuario['Subcuenca'] == terr_val].groupby('Anio')[especie_sel].sum().reset_index()
        else: # Sistema Hidrico
            df_hist = df_pecuario[df_pecuario['Sistema'] == terr_val].groupby('Anio')[especie_sel].sum().reset_index()
            
        df_hist = df_hist.sort_values(by='Anio')
        x_hist = df_hist['Anio'].values
        y_hist = df_hist[especie_sel].values
        
        x_offset = fila_terr['Año_Base']
        x_pred = np.arange(x_offset, anio_futuro + 1)
        x_norm_pred = x_pred - x_offset
        
        # Ecuaciones
        y_log = fila_terr['Log_K'] / (1 + fila_terr['Log_a'] * np.exp(-fila_terr['Log_r'] * x_norm_pred))
        y_exp = fila_terr['Exp_a'] * np.exp(fila_terr['Exp_b'] * x_norm_pred)
        y_poly = fila_terr['Poly_A']*(x_norm_pred**3) + fila_terr['Poly_B']*(x_norm_pred**2) + fila_terr['Poly_C']*x_norm_pred + fila_terr['Poly_D']
        
        fig = go.Figure()
        
        # Diccionario de colores e íconos para la gráfica
        color_data = {'Bovinos': 'brown', 'Porcinos': 'deeppink', 'Aves': 'goldenrod'}
        icono = {'Bovinos': '🐄', 'Porcinos': '🐖', 'Aves': '🐔'}
        
        fig.add_trace(go.Scatter(x=x_hist, y=y_hist, mode='markers', name='Censo ICA', marker=dict(color=color_data[especie_sel], size=10, symbol='diamond')))
        
        def config_linea(nombre_mod, color_mod):
            es_ganador = mejor_modelo == nombre_mod
            return dict(color=color_mod, width=4 if es_ganador else 2, dash='solid' if es_ganador else 'dash'), 1.0 if es_ganador else 0.4
            
        line_log, op_log = config_linea('Logístico', '#2980b9')
        fig.add_trace(go.Scatter(x=x_pred, y=y_log, mode='lines', name=f"Logístico (R²: {fila_terr['Log_R2']})", line=line_log, opacity=op_log))
        
        line_exp, op_exp = config_linea('Exponencial', '#e67e22')
        fig.add_trace(go.Scatter(x=x_pred, y=y_exp, mode='lines', name=f"Exponencial (R²: {fila_terr['Exp_R2']})", line=line_exp, opacity=op_exp))
        
        line_poly, op_poly = config_linea('Polinomial_3', '#27ae60')
        fig.add_trace(go.Scatter(x=x_pred, y=y_poly, mode='lines', name=f"Polinomial 3 (R²: {fila_terr['Poly_R2']})", line=line_poly, opacity=op_poly))
        
        fig.update_layout(
            title=f"Proyección de {icono[especie_sel]} {especie_sel} (Ganador: {mejor_modelo})", 
            xaxis_title="Año", yaxis_title="Número de Animales", hovermode="x unified", 
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True, key=f"plot_{key_suffix}")
        
        # Le agregamos el lado (Izquierdo/Derecho) al título para que nunca se repita
        lado = "Panel Izquierdo" if key_suffix == "g1" else "Panel Derecho"
        with st.expander(f"📐 Parámetros del Modelo de {especie_sel} ({lado})", expanded=False):
            df_coefs = pd.DataFrame([
                {"Modelo": "Logístico", "R²": f"{fila_terr['Log_R2']:.4f}", "Parámetros": f"K={fila_terr['Log_K']:.0f}, a={fila_terr['Log_a']:.4f}, r={fila_terr['Log_r']:.4f}"},
                {"Modelo": "Exponencial", "R²": f"{fila_terr['Exp_R2']:.4f}", "Parámetros": f"a={fila_terr['Exp_a']:.0f}, b={fila_terr['Exp_b']:.4f}"},
                {"Modelo": "Polinomial 3", "R²": f"{fila_terr['Poly_R2']:.4f}", "Parámetros": f"A={fila_terr['Poly_A']:.4e}, B={fila_terr['Poly_B']:.4e}, C={fila_terr['Poly_C']:.4f}, D={fila_terr['Poly_D']:.0f}"}
            ])
            def highlight_winner(row): return ['background-color: #d4edda' if row['Modelo'] == mejor_modelo else '' for _ in row]
            st.dataframe(df_coefs.style.apply(highlight_winner, axis=1), use_container_width=True)

    col_graf_1, col_graf_2 = st.columns(2)
    with col_graf_1:
        esp_1 = st.selectbox("Especie (Panel Izquierdo):", ["Bovinos", "Porcinos", "Aves"], index=0, key="sel_esp1")
        renderizar_panel_pecuario(esp_1, "g1")
    with col_graf_2:
        esp_2 = st.selectbox("Especie (Panel Derecho):", ["Bovinos", "Porcinos", "Aves"], index=1, key="sel_esp2")
        renderizar_panel_pecuario(esp_2, "g2")

# ==============================================================================
# 💾 EXPORTACIÓN DEL CEREBRO A PRODUCCIÓN (NUEVA ARQUITECTURA)
# ==============================================================================
if 'df_matriz_pecuaria' in st.session_state:
    st.markdown("---")
    st.subheader("💾 Exportar Cerebro Pecuario (Para Producción)")
    st.info("💡 **Paso a Producción:** Descarga esta matriz ya calculada y súbela a tu bucket de Supabase.")
    
    csv_pecu = st.session_state['df_matriz_pecuaria'].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar Matriz Maestra Pecuaria", 
        data=csv_pecu, 
        file_name="Matriz_Maestra_Pecuaria.csv", 
        mime="text/csv",
        type="primary"
    )
