import PyPDF2
import os

def extract_pdf_text(file_name):
    pdf_text = ""
    base_path = os.getcwd()
    pdf_read_path = base_path + f'/pdf-store/{file_name}'
    # read the pdf and convert it into text
    with open(pdf_read_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in range(len(reader.pages)):
            pdf_text += reader.pages[page].extract_text()

    # store the text into text file
    new_file_name = file_name.split('.')[0]
    with open(f'pdf-store/{new_file_name}.txt', 'w') as f:
        f.write(pdf_text)

if __name__ == "__main__":
    file_name = 'Mahindra_Thar_Car_Manual.pdf'
    extract_pdf_text(file_name)

