from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from PIL import Image
import tempfile
import io
from PyPDF2 import PdfReader, PdfWriter
import streamlit as st
from PyPDF2 import PdfMerger
from pdf1 import generate_pdf

def create_overlay(text, position, margin, image_path=None, image_width=None, image_height=None):
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=letter)

    # Add image to the PDF
    if image_path:
        image = Image.open(image_path)
        if image_width and image_height:
            image = image.resize((image_width, image_height))
        image_x = position[0]
        image_y = position[1]

        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            image.save(temp_file, format='JPEG')
            temp_file.seek(0)

            # Draw the image on the PDF
            can.drawImage(temp_file.name, image_x, image_y, width=image_width, height=image_height)

            # Close the temporary file
            temp_file.close()

    # Set font and font size
    can.setFont("Times-Roman", 15)

    # Calculate the width for text wrapping
    page_width, page_height = letter
    text_width = page_width - 2 * margin * mm

    # Create a text object for wrapping and alignment
    text_object = can.beginText()
    text_object.setTextOrigin(margin * mm, position[1] - 20)
    text_object.setFont("Times-Roman", 15)
    # Set the leading value to 1.5 times the font size
    text_object.setLeading(15 * 1.5)

    # Split the text into lines that fit within the width
    lines = []
    words = text.split()
    current_line = ''
    for word in words:
        if can.stringWidth(current_line + ' ' + word) < text_width:
            current_line += ' ' + word
        else:
            lines.append(current_line.lstrip())
            current_line = word
    lines.append(current_line.lstrip())

    # Add the text in lines, wrapping as needed
    for line in lines:
        text_object.textLine(line)
    can.drawText(text_object)

    can.save()

    # Move to the beginning of the StringIO buffer
    packet.seek(0)
    return packet

def merge_pdfs(base_pdf, overlay_pdf):
    reader = PdfReader(base_pdf)
    overlay_reader = PdfReader(overlay_pdf)
    writer = PdfWriter()

    # Add overlay to each page
    for page_number in range(len(reader.pages)):
        page = reader.pages[page_number]
        overlay_page = overlay_reader.pages[0]
        page.merge_page(overlay_page)
        writer.add_page(page)

    return writer

def generate_merged_pdf(input_pdf_path, text, position, image_path, image_width, image_height):
    # Create an overlay PDF with the desired text and image
    right_margin = 35  # Right margin in mm
    overlay_pdf_stream = create_overlay(text, position, right_margin, image_path, image_width, image_height)

    # Merge the original PDF with the overlay
    writer = merge_pdfs(input_pdf_path, overlay_pdf_stream)

    # Write the modified content to a new PDF file
    output_pdf_path = "static/inarrate.pdf"
    with open(output_pdf_path, "wb") as output_pdf:
        writer.write(output_pdf)

    # Generate the second PDF
    braille = text
    pdf_data = generate_pdf(braille)

    # Merge the two PDFs
    merged_pdf = PdfMerger()
    merged_pdf.append(open(output_pdf_path, 'rb'))
    merged_pdf.append(io.BytesIO(pdf_data))
    
    # Write the merged PDF to a temporary file
    merged_pdf_path = 'static/final.pdf'
    with open(merged_pdf_path, "wb") as merged_pdf_file:
        merged_pdf.write(merged_pdf_file)

    return merged_pdf_path


# Call the function with the required arguments
#input_pdf_path = "Doc1.pdf"
#text = "brown dog is running through tall grass The dog leaps into the air, catching the ball in mid-flight. The dog runs around the backyard, its tail wagging excitedly. As the sun begins to set, the dog curls up in its bed, dreaming of all the fun it had that day. After a while, the dog tires out and lies down in the shade, panting happily. The dog's tail still wags slightly in its sleep, a testament to the joy it experienced during playtime."
#position = (100, 470)  # Position where the text will appear
#image_path = "barkingmain.jpg"
#image_width = 200
#image_height = 200
#merged_pdf = generate_merged_pdf(input_pdf_path, text, position, image_path, image_width, image_height)
# Write the merged PDF to a file
#with open("merged.pdf", "wb") as merged_pdf_file:
#    merged_pdf.write(merged_pdf_file)

# Download the merged PDF
#st.download_button("Download PDF", data=open("merged.pdf", "rb"), file_name="merged.pdf")