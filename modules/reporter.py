# modules/reporter.py


import os
import tempfile
import matplotlib
matplotlib.use("Agg") 
from datetime import datetime
import matplotlib.pyplot as plt
import streamlit as st
from fpdf import FPDF
from modules.config import Config

# --- 1. FUNCIÓN DE MAPA ESTÁTICO (DEBE IR PRIMERO) ---
def create_context_map_static(gdf_stations, gdf_municipios=None, gdf_subcuencas=None):
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        if gdf_municipios is not None and not gdf_municipios.empty:
            gdf_municipios.plot(ax=ax, color="none", edgecolor="gray", linewidth=0.5, alpha=0.5)
        if gdf_subcuencas is not None and not gdf_subcuencas.empty:
            gdf_subcuencas.plot(ax=ax, color="#e6f2ff", edgecolor="blue", alpha=0.2, linewidth=1)
        if gdf_stations is not None and not gdf_stations.empty:
            gdf_stations.plot(ax=ax, color="red", markersize=40, edgecolor="white", zorder=3)
        ax.set_title("Mapa de Localización")
        ax.set_axis_off()
        return fig
    except Exception as e:
        print(f"Error mapa estático: {e}")
        return None

class PDFReport(FPDF):
    def header(self):
        # Fondo decorativo sutil en el encabezado
        self.set_fill_color(46, 134, 193) # Azul SIHCLI
        self.rect(0, 0, 210, 3, 'F')
        
        if self.page_no() == 1:
            return # La portada tiene su propio diseño

        if os.path.exists(Config.LOGO_PATH):
            self.image(Config.LOGO_PATH, 10, 8, 20)
        
        self.set_font("Arial", "B", 10)
        self.set_x(35)
        self.cell(0, 10, f"{Config.APP_TITLE} - Inteligencia Territorial", 0, 0, "L")
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Página {self.page_no()}", 0, 1, "R")
        self.ln(5)

    def print_chapter_title(self, label):
        self.set_font("Arial", "B", 14)
        self.set_fill_color(235, 245, 251)
        self.cell(0, 10, f"  {label}", 0, 1, "L", 1)
        self.ln(4)

    def add_matplotlib_figure(self, fig, title=""):
        """Añade una figura de Matplotlib al PDF de forma segura."""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                fig.savefig(tmp.name, format="png", dpi=150, bbox_inches="tight")
                tmp_path = tmp.name

            if self.get_y() + 100 > 270:
                self.add_page()

            if title:
                self.set_font("Arial", "B", 10)
                self.cell(0, 8, title, 0, 1, "C")

            self.image(tmp_path, x=20, w=170)
            self.ln(5)
            os.remove(tmp_path)
            plt.close(fig)
        except Exception as e:
            st.error(f"Error al insertar gráfico en PDF: {e}")

def generate_pdf_report(df_long, gdf_stations, analysis_results, **kwargs):
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # --- PORTADA ESTILO CUENCAVERDE ---
    pdf.set_y(100)
    pdf.set_font("Arial", "B", 28)
    pdf.set_text_color(46, 134, 193)
    pdf.cell(0, 15, "INFORME TÉCNICO", 0, 1, "C")
    pdf.set_font("Arial", "", 18)
    pdf.set_text_color(100)
    pdf.cell(0, 10, "Análisis de Hidrodiversidad y Variabilidad", 0, 1, "C")
    
    pdf.ln(80)
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 10, f"Generado por: SIHCLI-POTER v3.0 | {datetime.now().strftime('%Y-%m-%d')}", 0, 1, "C")

    # --- SECCIÓN 1: EL MAPA DEL SISTEMA ---
    pdf.add_page()
    pdf.print_chapter_title("1. Contexto Geoespacial")
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 6, "Este análisis integra la red de estaciones seleccionada con el Modelo Digital de Elevación y coberturas de suelo, permitiendo una visión borgeana del territorio donde cada punto es un reflejo de la cuenca completa.")
    
    # Insertar el Mapa Estático
    fig_map = create_context_map_static(gdf_stations, kwargs.get("gdf_municipios"), kwargs.get("gdf_subcuencas"))
    if fig_map:
        pdf.add_matplotlib_figure(fig_map, "Distribución Espacial de la Red de Monitoreo")

    # --- SECCIÓN 2: MODELACIÓN AVANZADA (LA SORPRESA) ---
    if 'matrices' in kwargs:
        pdf.add_page()
        pdf.print_chapter_title("2. Resultados de Modelación Distribuida (Aleph)")
        pdf.print_section_body("Se ha ejecutado el motor físico para determinar el balance hídrico distribuido, identificando zonas críticas de rendimiento y vulnerabilidad.")
        
        # Aquí puedes agregar lógica para exportar imágenes de las matrices de física
        # Por ahora, dejamos el espacio para que la App envíe las figuras de Plotly
    
    # --- SECCIÓN 3: SÍNTESIS ESTADÍSTICA ---
    pdf.add_page()
    pdf.print_chapter_title("3. Estadísticas y Récords")
    # Lógica de tablas formateadas (como ya la tenías, pero con diseño limpio)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        return open(tmp.name, "rb").read()