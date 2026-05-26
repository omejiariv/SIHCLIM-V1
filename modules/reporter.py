import os
import tempfile
import matplotlib.pyplot as plt
from datetime import datetime
from fpdf import FPDF
from modules.config import Config

class PDFReport(FPDF):
    def header(self):
        self.set_fill_color(46, 134, 193)
        self.rect(0, 0, 210, 3, 'F')
        if self.page_no() > 1:
            if os.path.exists(Config.LOGO_PATH):
                self.image(Config.LOGO_PATH, 10, 8, 15)
            self.set_font("Arial", "B", 10)
            self.cell(0, 10, f"{Config.APP_TITLE} - Reporte Ejecutivo", 0, 1, "R")
            self.ln(5)

    def print_chapter(self, title, text, figure=None):
        self.add_page()
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, title, 0, 1, "L", 1)
        self.ln(5)
        self.set_font("Arial", "", 11)
        self.multi_cell(0, 6, text)
        self.ln(5)
        if figure:
            self.add_matplotlib_figure(figure)

    def add_matplotlib_figure(self, fig, title=""):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            fig.savefig(tmp.name, format="png", dpi=100, bbox_inches="tight")
            self.image(tmp.name, x=20, w=170)
            os.remove(tmp.name)
            plt.close(fig)

# --- MOTOR DE CONSOLIDACIÓN MODULAR ---
def generate_consolidated_pdf(lista_capitulos):
    """
    Recibe una lista de diccionarios: 
    [{'title': '...', 'text': '...', 'fig': figure_obj}, ...]
    """
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Portada
    pdf.add_page()
    pdf.set_font("Arial", "B", 24)
    pdf.cell(0, 100, "INFORME EJECUTIVO CONSOLIDADO", 0, 1, "C")
    
    # Iterar sobre los fragmentos guardados en sesión
    for cap in lista_capitulos:
        pdf.print_chapter(cap['title'], cap['text'], cap.get('fig'))
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        return open(tmp.name, "rb").read()
