import streamlit as st
from fpdf import FPDF, HTMLMixin

def generate_pdf(braille):
    class BraillePDF(FPDF, HTMLMixin):
        def __init__(self, *args, **kwargs):
            FPDF.__init__(self, *args, **kwargs)
            self.add_font('braille', '', r'\braille_font\Braille-1lA2.ttf', uni=True)

        def generate_pdf(self, braille):
            self.add_page()
            self.set_auto_page_break(auto=True, margin=20)
            self.set_font("braille", size=16)
            self.multi_cell(196, 10, txt=braille, align='J')
            return self.output(dest='S').encode('latin-1')

    pdf = BraillePDF()
    return pdf.generate_pdf(braille)

#braille = 'hi'
#pdf_data = generate_pdf(braille)
#st.download_button("Download PDF", data=pdf_data, file_name="i-narrate_braille_script.pdf")