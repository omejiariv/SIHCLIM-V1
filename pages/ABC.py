import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- CONFIGURACIÓN UI ---
st.set_page_config(page_title="ACB de SbN", layout="wide")

st.markdown("""
<style>
    font-family: 'Georgia', serif !important;
    .metric-card {background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 5px solid #2ecc71; box-shadow: 0 2px 5px rgba(0,0,0,0.05);}
    .metric-card-red {border-left-color: #e74c3c;}
</style>
""", unsafe_allow_html=True)

st.title("🌱 Análisis Costo-Beneficio (ACB): Soluciones basadas en la Naturaleza")
st.info("Traducción de los impactos ecológicos y volumétricos a flujos financieros descontados para justificar la viabilidad económica ante bancas de desarrollo e inversionistas.")

# ==========================================================
# 1. PARÁMETROS DEL PROYECTO (INPUTS)
# ==========================================================
with st.expander("⚙️ 1. Parametrización del Escenario Base", expanded=True):
    col_p1, col_p2, col_p3 = st.columns(3)
    
    with col_p1:
        st.markdown("#### 📏 Escala Física")
        ha_sbn = st.number_input("Hectáreas a Restaurar (SbN):", min_value=10.0, value=1500.0, step=50.0)
        horizonte = st.slider("Horizonte de Evaluación (Años)", 10, 50, 20)
        tasa_descuento = st.slider("Tasa de Descuento (Costo de Capital %)", 1.0, 15.0, 8.0) / 100
        
    with col_p2:
        st.markdown("#### 💰 Estructura de Costos (USD)")
        costo_ha_usd = st.number_input("CAPEX: Costo Restauración (USD/ha):", value=2100)
        opex_pct = st.slider("OPEX: Mantenimiento Anual (% del CAPEX):", 1.0, 10.0, 2.5) / 100
        costo_oportunidad_ha = st.number_input("Costo Oportunidad (USD/ha/año):", value=150, help="Renta agrícola/ganadera dejada de percibir.")

    with col_p3:
        st.markdown("#### 🌍 Valoración de Beneficios (USD)")
        ahorro_agua_ha = st.number_input("Ahorro Tratamiento PTAP (USD/ha/año):", value=300)
        co2_ton_ha = st.number_input("Secuestro Carbono (Ton CO2/ha/año):", value=12.0)
        precio_co2 = st.number_input("Precio Mercado Carbono (USD/Ton):", value=15.0)
        mitigacion_riesgo_ha = st.number_input("Mitigación Desastres (USD/ha/año):", value=250)

# ==========================================================
# 2. MOTOR DE CÁLCULO FINANCIERO (CASHFLOW)
# ==========================================================
# Cálculos de Totales Iniciales
capex_total = ha_sbn * costo_ha_usd
opex_total_anual = (capex_total * opex_pct) + (ha_sbn * costo_oportunidad_ha)
beneficio_agua_anual = ha_sbn * ahorro_agua_ha
beneficio_co2_anual = ha_sbn * (co2_ton_ha * precio_co2)
beneficio_riesgo_anual = ha_sbn * mitigacion_riesgo_ha

beneficio_total_anual = beneficio_agua_anual + beneficio_co2_anual + beneficio_riesgo_anual

# Proyección a N años
años = np.arange(0, horizonte + 1)
flujos = pd.DataFrame({'Año': años})

# Asignación de Costos
flujos['Costos'] = opex_total_anual
flujos.loc[0, 'Costos'] = capex_total  # Año 0 es full CAPEX, sin OPEX
flujos['Desglose_Costos'] = np.where(flujos['Año'] == 0, 'CAPEX', 'OPEX + Costo Oportunidad')

# Asignación de Beneficios (Curva de maduración del bosque)
# Los bosques no dan 100% de beneficios el día 1. Usamos un factor logarítmico simple de maduración
flujos['Factor_Maduracion'] = np.where(flujos['Año'] == 0, 0.0, 1 - np.exp(-0.3 * flujos['Año']))
flujos['Beneficios'] = beneficio_total_anual * flujos['Factor_Maduracion']

# Matemáticas Financieras (Descuento)
flujos['Flujo_Neto'] = flujos['Beneficios'] - flujos['Costos']
flujos['Factor_Descuento'] = 1 / ((1 + tasa_descuento) ** flujos['Año'])

flujos['Costos_Descontados'] = flujos['Costos'] * flujos['Factor_Descuento']
flujos['Beneficios_Descontados'] = flujos['Beneficios'] * flujos['Factor_Descuento']
flujos['Flujo_Neto_Descontado'] = flujos['Flujo_Neto'] * flujos['Factor_Descuento']
flujos['Flujo_Acumulado'] = flujos['Flujo_Neto_Descontado'].cumsum()

# Indicadores Macro
vpn = flujos['Flujo_Neto_Descontado'].sum()
rcb = flujos['Beneficios_Descontados'].sum() / flujos['Costos_Descontados'].sum()
try:
    payback_year = flujos[flujos['Flujo_Acumulado'] >= 0]['Año'].iloc[0]
except:
    payback_year = "No recupera"

# ==========================================================
# 3. DASHBOARD DE RESULTADOS GERENCIALES
# ==========================================================
st.markdown("---")
st.subheader("📊 2. Viabilidad y Métricas de Retorno (ROI Ambiental)")

m1, m2, m3, m4 = st.columns(4)
color_vpn = "metric-card" if vpn > 0 else "metric-card metric-card-red"

m1.markdown(f"<div class='{color_vpn}'><b>Valor Presente Neto (VPN)</b><br><h2 style='margin:0; color:#2c3e50;'>${vpn/1e6:,.2f} M</h2>{'🟢 Viable y Genera Valor' if vpn>0 else '🔴 Destruye Valor'}</div>", unsafe_allow_html=True)
m2.markdown(f"<div class='metric-card'><b>Relación Beneficio/Costo</b><br><h2 style='margin:0; color:#2c3e50;'>{rcb:.2f}x</h2>Retorna ${rcb:.2f} por cada $1</div>", unsafe_allow_html=True)
m3.markdown(f"<div class='metric-card'><b>TIR Social / Payback</b><br><h2 style='margin:0; color:#2c3e50;'>Año {payback_year}</h2>Punto de Equilibrio</div>", unsafe_allow_html=True)
m4.markdown(f"<div class='metric-card'><b>Inversión Inicial (CAPEX)</b><br><h2 style='margin:0; color:#e74c3c;'>${capex_total/1e6:,.2f} M</h2>Capital a levantar hoy</div>", unsafe_allow_html=True)

# --- GRÁFICA PROFESIONAL DE FLUJOS ---
st.markdown("<br>", unsafe_allow_html=True)

fig = make_subplots(specs=[[{"secondary_y": True}]])

# Barras de Costos (Negativas)
fig.add_trace(go.Bar(x=flujos['Año'], y=-flujos['Costos_Descontados'], name='Costos (CAPEX+OPEX)', marker_color='#e74c3c', opacity=0.8), secondary_y=False)
# Barras de Beneficios (Positivas)
fig.add_trace(go.Bar(x=flujos['Año'], y=flujos['Beneficios_Descontados'], name='Beneficios (Agua+CO2+Riesgo)', marker_color='#2ecc71', opacity=0.8), secondary_y=False)
# Línea de Acumulado
fig.add_trace(go.Scatter(x=flujos['Año'], y=flujos['Flujo_Acumulado'], name='Flujo Acumulado (VPN)', mode='lines+markers', line=dict(color='#2980b9', width=3), marker=dict(size=8)), secondary_y=True)

fig.update_layout(
    title='Proyección de Flujos de Caja Descontados y Punto de Equilibrio',
    barmode='relative',
    height=450,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    margin=dict(l=20, r=20, t=50, b=20)
)
fig.update_yaxes(title_text="Flujo Anual ($)", secondary_y=False)
fig.update_yaxes(title_text="Flujo Acumulado ($)", secondary_y=True, showgrid=False)
fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)

st.plotly_chart(fig, use_container_width=True)

# TABLA DESGLOSADA (Resumen)
with st.expander("🧮 Ver Tabla de Flujos Financieros", expanded=False):
    st.dataframe(
        flujos[['Año', 'Costos', 'Beneficios', 'Flujo_Neto', 'Flujo_Acumulado']].style.format({
            'Costos': '${:,.0f}', 'Beneficios': '${:,.0f}', 'Flujo_Neto': '${:,.0f}', 'Flujo_Acumulado': '${:,.0f}'
        }),
        use_container_width=True, hide_index=True
    )
