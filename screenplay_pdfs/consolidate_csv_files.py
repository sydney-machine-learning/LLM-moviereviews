"""
Screenplay Consolidation and Cleaning Utility

This script processes individual screenplay CSV files and combines them into a single consolidated dataset.
It performs the following operations:
1. Reads individual screenplay CSV files from the 'screenplay_pdfs' folder
2. Correlates each screenplay with movie metadata from 'selected_movie_info.csv'
3. Cleans the screenplay text by:
   - Removing timestamps and formatting characters
   - Filtering out common words (articles, pronouns, etc.)
   - Removing non-ASCII characters and normalizing whitespace
   - Applying the same cleaning process used for subtitles for consistency
4. Consolidates all screenplay data into a single DataFrame with movie metadata
5. Saves the consolidated data to 'screenplay_pdfs/cleaned_screenplays.csv'


Dependencies: pandas, re, os, collections (Counter)
Input: Individual screenplay CSV files, selected_movie_info.csv
Output: cleaned_screenplays.csv
"""

import pandas as pd
import re
import os
from collections import Counter

common_exclusions = {'-', '♪', 'i', 'you', 'to', 'the', 'a', 'and', 'it', 'is', 'that', 'of', 's', 't', 'what', 'in', 'me', 'this', 'on', 'sir', 'get', 'for', 'she', 'be', 'eve', 'not', 'have', 'all', 'her', 'was', 'my', 'can', 'oh', 'no', 'we', 'well', 'annie', 'be', 'he', 'like', 'don'}

def clean_text(text):
    # Function to clean the text similar to the subtitles cleaning process
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'\n', ' ', text)  # Replace newlines with a space
    text = re.sub(r'\r', '', text)  # Remove carriage returns
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    text = text.strip()  # Remove leading and trailing whitespace
    return text

def parse_srt_excluding_common(content):
    # Extract all timestamps
    timestamps = re.findall(r'\d{2}:\d{2}:\d{2},\d{3}', content)
    if timestamps:
        last_timestamp = timestamps[-1]
        hours, minutes, seconds_milliseconds = last_timestamp.split(':')
        seconds, milliseconds = seconds_milliseconds.split(',')        
        total_minutes = int(hours) * 60 + int(minutes) + int(seconds) / 60 + int(milliseconds) / (60 * 1000)
    else:
        total_minutes = 0

    # Remove timestamps and numbers
    lines = re.sub(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', '', content)
    lines = re.sub(r'\d+', '', lines)
    lines = lines.replace('\n', '')
    lines = re.sub(r'\n\s*\n', '\n', lines).strip()
    lines = lines.replace('-', '')
    lines = lines.replace('♪', '')
    lines = re.sub(r'</?i>', '', lines)
    lines = re.sub(r'</?b>', '', lines)

    # Extract words
    words = re.findall(r'\b\w+\b', lines.lower())
    word_count = len(words)

    # Filter out common exclusions
    filtered_words = [word for word in words if word not in common_exclusions]

    # Get the most common words
    common_words = Counter(filtered_words).most_common(10)
    top_ten_words = [word for word, _ in common_words]

    return word_count, total_minutes, top_ten_words, lines

def main():
    folder_path = 'screenplay_pdfs'
    movie_info_path = 'selected_movie_info.csv'
    
    # Read the movie information from selected_movie_info.csv
    movie_info_df = pd.read_csv(movie_info_path)
    
    data = []
    
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            csv_path = os.path.join(folder_path, filename)
            
            # Extract movie name from the filename
            movie_name = filename.replace('.csv', '')
            
            # Retrieve additional information from the movie_info_df
            movie_info_row = movie_info_df[movie_info_df['movie'] == movie_name]
            if not movie_info_row.empty:
                imdb_id = movie_info_row['imdb_id'].values[0]
                year = movie_info_row['year'].values[0]
                award = movie_info_row['award'].values[0]
            else:
                print(f"Movie information not found for movie: {movie_name}")
                continue
            
            # Read the CSV file
            with open(csv_path, 'r', encoding='utf-8') as file:
                screenplay_text = file.read()
            
            # Clean the screenplay text using parse_srt_excluding_common
            _, _, _, cleaned_screenplay_text = parse_srt_excluding_common(screenplay_text)
            
            # Further clean the text to remove non-ASCII characters
            cleaned_screenplay_text = clean_text(cleaned_screenplay_text)
            
            # Append the data to the list
            data.append({
                'movie': movie_name,
                'imdb_id': imdb_id,
                'year': year,
                'award': award,
                'cleaned_subtitle_text': cleaned_screenplay_text
            })
    
    # Create a DataFrame from the data
    df = pd.DataFrame(data)
    
    # Save the DataFrame to a CSV file
    df.to_csv('screenplay_pdfs/cleaned_screenplays.csv', index=False)
    

if __name__ == "__main__":
    main()