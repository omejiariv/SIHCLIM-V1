import os
import tempfile
import plotly.io as pio
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
            self.cell(0, 10, "Reporte Ejecutivo - SIHCLI-POTER", 0, 1, "R")
            self.ln(5)

    def print_chapter(self, title, text, fig=None):
        self.add_page()
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, title, 0, 1, "L", 1)
        self.ln(5)
        self.set_font("Arial", "", 11)
        self.multi_cell(0, 6, text)
        self.ln(5)
        if fig:
            self.add_plotly_figure(fig)

    def add_plotly_figure(self, fig):
        """Convierte Plotly a imagen temporal e inserta en PDF."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            pio.write_image(fig, tmp.name, format="png", scale=2)
            if self.get_y() + 100 > 270: self.add_page()
            self.image(tmp.name, x=20, w=170)
            os.remove(tmp.name)
            self.ln(5)

def generate_consolidated_pdf(lista_capitulos):
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Portada
    pdf.add_page()
    pdf.set_font("Arial", "B", 24)
    pdf.cell(0, 100, "INFORME TÉCNICO CONSOLIDADO", 0, 1, "C")
    
    for cap in lista_capitulos:
        pdf.print_chapter(cap['title'], cap['text'], cap.get('fig'))
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        return open(tmp.name, "rb").read()
