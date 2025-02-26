from pypdf import PdfReader
import re
import os

TEST_PDF_PATH = r"C:\\Users\\User\\Desktop\\test_path\\2025 Journal article Toward expert-level medical question answering with LLMs.pdf"

OUTPUT_PATH = r"C:\\Users\\User\\Desktop\\test_path"


def extract_text_from_pdf(pdf_path, output_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")
    

    # The [0] index is used because os.path.splitext() returns a tuple containing two elements:
    # (filename_without_extension, file_extension) = os.path.splitext(os.path.basename(TEST_PDF_PATH))
    filename_without_extension = os.path.splitext(os.path.basename(pdf_path))[0]

    reader = PdfReader(pdf_path)

    all_text = []
    alter_text = []



    for page_number, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            page_text = re.sub(r"\n+", "\n", page_text)
            page_text = re.sub(r'\s+', ' ', page_text.strip())
            formatted_text = f"----- Page {page_number} -----\n{page_text}"
            all_text.append(formatted_text)


            for paragraph in page_text.split('\n'):
                formatted_paragraph = f"----- Paragraph {page_number + 1} -----\n{paragraph}"
                alter_text.append(formatted_paragraph)

        else:
            print(f"--------- Page {page_number + 1} is empty---------")

    save_file = output_path + f"\\{filename_without_extension}_extracted_text.txt"

    with open(save_file, 'w', encoding="utf-8") as f:
        f.write("\n\n".join(all_text))

    alternative_file =  output_path + f"\\{filename_without_extension}_extracted_text_alternative.txt"
    with open(alternative_file, 'w', encoding="utf-8") as f:
        f.write("\n\n".join(alter_text))
    
    print(f"Text extracted and saved as:\n- {save_file}\n- {alternative_file}")

 


extract_text_from_pdf(TEST_PDF_PATH, OUTPUT_PATH)




def format_text_file(input_path, output_path):
    filename_without_extension = os.path.splitext(os.path.basename(input_path))[0]
    # Read the raw text file
    with open(input_path, 'r', encoding="utf-8") as f:
        text = f.read()

    formatted_text = []
    
    for paragraph in text.split("----- Paragraph "):  
        if not paragraph.strip():
            continue

        # Extract paragraph number and content
        match = re.match(r"(\d+) -----(.*)", paragraph, re.DOTALL)
        if match:
            paragraph_number, content = match.groups()
            content = content.strip()

            # Add paragraph header
            formatted_text.append(f"----- Paragraph {paragraph_number} -----\n")

            # Improve readability: Add line breaks after sentences
            content = re.sub(r'(?<=[.!?])\s+', '\n', content)  # Insert line breaks after punctuation

            # Wrap text for better readability
            wrapped_content = "\n".join([line.strip() for line in content.split("\n") if line.strip()])
            formatted_text.append(wrapped_content)

            formatted_text.append("\n")  # Extra space between paragraphs



    save_file = output_path + f"\\{filename_without_extension}_extracted_text_formatted.txt"
    # Save formatted text
    with open(save_file, 'w', encoding="utf-8") as f:
        f.write("\n".join(formatted_text))

    print(f"Formatted text saved: {output_path}")



INPUT_file = r"C:\\Users\\User\\Desktop\\test_path\\2025 Journal article Toward expert-level medical question answering with LLMs_extracted_text_alternative.txt"


format_text_file(INPUT_file, OUTPUT_PATH)
        


    
    

    
    
    



