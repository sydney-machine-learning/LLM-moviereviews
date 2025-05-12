"""
PDF Screenplay Text Extraction Utility

This script extracts text content from movie screenplay PDFs and converts it to CSV format.
It performs the following operations:
1. Scans the 'screenplay_pdfs' folder for PDF files
2. Extracts all text content from each PDF using PyMuPDF (fitz)
3. Processes the extracted text by:
   - Converting multi-line content to single-line format
   - Removing unnecessary formatting
4. Saves each screenplay's text as a separate CSV file in the 'data/screenplays/' folder
   with the movie title as filename (e.g., "avatar.csv")

The extracted text is used later in the project for:
- Sentiment and emotion analysis
- Comparison with reviews and subtitles
- AI-based analysis of screenplay content

Dependencies: PyMuPDF (fitz), os, csv, re
"""

import fitz  # PyMuPDF
import os
import csv
import re

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

def save_text_to_csv(text, output_path, movie_data):
    # Replace newlines with spaces to put all text on one line
    single_line_text = text.replace('\n', ' ').replace('\r', '')
    
    # Write the data to a CSV file with the desired columns
    with open(output_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(['movie', 'imdb_id', 'year', 'award', 'cleaned_screenplay_text'])
        # Write data row
        writer.writerow([
            movie_data['movie'],
            movie_data['imdb_id'],
            movie_data['year'],
            movie_data['award'],
            single_line_text
        ])

def main():
    folder_path = 'screenplay_pdfs'
    output_folder = 'data/screenplays'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Read movie metadata from selected_movie_info.csv
    movie_metadata = {}
    try:
        with open('selected_movie_info.csv', 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                movie_metadata[row['movie'].lower()] = row
    except Exception as e:
        print(f"Warning: Could not load movie metadata: {e}")
        print("Continuing with limited metadata...")
    
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            
            # Extract the movie title from the filename
            # Remove year and file extension, convert hyphens to spaces, and capitalize properly
            movie_title = re.sub(r'-\d{4}\.pdf$', '', filename)
            movie_title = movie_title.replace('-', ' ').title()
            
            # Create sanitized version for the output filename
            sanitized_title = movie_title.lower().replace(' ', '_').replace(',', '').replace('\'', '').replace('"', '')
            output_csv_path = os.path.join(output_folder, f"{sanitized_title}.csv")
            
            # Extract text from the PDF
            extracted_text = extract_text_from_pdf(pdf_path)
            
            # Get metadata for this movie
            movie_data = {
                'movie': movie_title,
                'imdb_id': '',
                'year': '',
                'award': 'Oscar'  # Default assumption
            }
            
            # Try to match with metadata from the CSV
            movie_key = movie_title.lower()
            if movie_key in movie_metadata:
                movie_data = movie_metadata[movie_key]
            else:
                # Try partial matching for movies with slightly different names
                for meta_key, meta_value in movie_metadata.items():
                    if meta_key in movie_key or movie_key in meta_key:
                        movie_data = meta_value
                        break
                    
                # If year is in the filename, extract it
                year_match = re.search(r'-(\d{4})\.pdf$', filename)
                if year_match:
                    movie_data['year'] = year_match.group(1)
            
        # Save the extracted text to a CSV file
            save_text_to_csv(extracted_text, output_csv_path, movie_data)
            
            print(f"Text extracted from '{movie_title}' and saved to {output_csv_path}")
    


def consolidate_screenplays(screenplay_folder):
    """Consolidate all screenplay CSVs into a single file"""
    all_data = []
    
    for filename in os.listdir(screenplay_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(screenplay_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                csv_reader = csv.DictReader(file)
                for row in csv_reader:
                    all_data.append(row)
    
    if all_data:
        # Write all data to a consolidated CSV
        output_path = 'data/screenplays/cleaned_screenplays.csv'
        with open(output_path, 'w', encoding='utf-8', newline='') as file:
            fieldnames = ['movie', 'imdb_id', 'year', 'award', 'cleaned_screenplay_text']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_data:
                writer.writerow(row)
        print(f"Consolidated all screenplay data to {output_path}")
        
        # Create a version without screenplay text (metadata only)
        metadata_path = 'data/screenplay_metadata.csv'
        with open(metadata_path, 'w', encoding='utf-8', newline='') as file:
            fieldnames = ['movie', 'imdb_id', 'year', 'award']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_data:
                # Create a new row without the screenplay text
                metadata = {k: v for k, v in row.items() if k != 'cleaned_screenplay_text'}
                writer.writerow(metadata)
        print(f"Saved screenplay metadata to {metadata_path}")

if __name__ == "__main__":
    main()