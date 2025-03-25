import fitz  # PyMuPDF
import os
import csv

def extract_text_from_pdf(pdf_path):
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    
    # Initialize an empty string to store the extracted text
    extracted_text = ""
    
    # Iterate through each page in the PDF
    for page_num in range(len(pdf_document)):
        # Get the page
        page = pdf_document.load_page(page_num)
        
        # Extract text from the page
        page_text = page.get_text()
        
        # Append the extracted text to the overall text
        extracted_text += page_text
    
    return extracted_text

def save_text_to_csv(text, output_path):
    # Replace newlines with spaces to put all text on one line
    single_line_text = text.replace('\n', ' ').replace('\r', '')
    
    # Write the single line text to a CSV file
    with open(output_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([single_line_text])

def main():
    folder_path = 'screenplay_pdfs'
    
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            output_csv_path = os.path.join(folder_path, filename.replace('.pdf', '.csv'))
            
            # Extract text from the PDF
            extracted_text = extract_text_from_pdf(pdf_path)
            
            # Save the extracted text to a CSV file
            save_text_to_csv(extracted_text, output_csv_path)
            
            print(f"Text extracted and saved to {output_csv_path}")

if __name__ == "__main__":
    main()