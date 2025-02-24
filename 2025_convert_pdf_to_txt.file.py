import os
from pypdf import PdfReader


FILE_PATH = r"C:\\Users\\kuzne\\Desktop\\rag_articles_pdf\\2025 Almanac - Retrieval Augmented Language models for clinical medicine.pdf"
OUTPUT_FOLDER = r"C:\\Users\\kuzne\Desktop\\rag_articles_pdf\\converted_pdf_to_txt_files"

'''
reader = PdfReader(FILE_PATH)
page = reader.pages[0]
print(page.extract_text())

'''


def convert_pdf_to_text(pdf_path, output_folder):
    # Load PDF
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    # Save as a text file
    base_name = os.path.basename(pdf_path).replace(".pdf", "")
    output_path = os.path.join(output_folder, f"{base_name}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Converted {pdf_path} to {output_path}")



convert_pdf_to_text(FILE_PATH, OUTPUT_FOLDER)