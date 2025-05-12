"""
Subtitle Processing Pipeline for Oscar-nominated Films

This script performs the following operations:
1. Downloads subtitle files (.srt) for selected Oscar-nominated films from Kaggle
2. Retrieves IMDb IDs for each movie using the OMDb API
3. Processes each subtitle file by:
   - Extracting timing information
   - Calculating movie duration in minutes
   - Removing timestamps, numbers, and formatting tags
   - Filtering out common words (like 'the', 'a', 'and', etc.)
   - Identifying the top 10 most frequent words in each subtitle
4. Creates a DataFrame with movie information and cleaned subtitle text
5. Saves the following files:
   - individual movie subtitle files in data/subtitles/[movie_title].csv
   - imdb-reviews/cleaned_subtitles.csv: Contains all data including the full cleaned subtitle text
   - selected_movie_info.csv: Contains only the movie metadata without subtitle content
6. Downloads IMDb reviews for each movie and saves them as individual CSV files in the imdb-reviews folder

Dependencies: pandas, kagglehub, requests, python-dotenv
Environment variables: OMDB_API_KEY (for retrieving IMDb IDs)
"""

import pandas as pd
import kagglehub
import re
import os
from collections import Counter
import requests
from dotenv import load_dotenv

load_dotenv()
OMDB_API_KEY = os.getenv('OMDB_API_KEY')

# Download from Kaggle
path_subtitles = kagglehub.dataset_download("mlopssss/subtitles")
path_imdb_reviews = kagglehub.dataset_download("mlopssss/imdb-movie-reviews-grouped-by-ratings")

# Create imdb-reviews directory if it doesn't exist
if not os.path.exists('imdb-reviews'):
    os.makedirs('imdb-reviews')

# Create data/subtitles directory if it doesn't exist
if not os.path.exists('data/subtitles'):
    os.makedirs('data/subtitles', exist_ok=True)

common_exclusions = {'-','♪','i', 'you', 'to', 'the', 'a', 'and', 'it', 'is', 'that', 'of','s', 't', 'what', 'in', 'me', 'this', 'on', 'sir', 'get','for', 'she', 'be', 'eve', 'not', 'have', 'all', 'her', 'was', 'my','can', 'oh', 'no', 'we', 'well', 'annie', 'be', 'he', 'like', 'don'}

Movies_df = pd.DataFrame([
    ["The Shawshank Redemption", 1995, "Oscar"],
    ["Brokeback Mountain", 2006, "Oscar"],
    ["Avatar", 2010, "Oscar"],
    ["Titanic", 1998, "Oscar"],
    ["Crouching Tiger, Hidden Dragon", 2001, "Oscar"],
    ["Nomadland", 2021, "Oscar"]
    ], columns=["movie", "year", "award"])

def get_imdb_id(movie_title):
    url = f"http://www.omdbapi.com/?t={movie_title}&apikey={OMDB_API_KEY}"
    response = requests.get(url)
    data = response.json()
    
    if response.status_code == 200 and data["Response"] == "True":
        return data["imdbID"]
    else:
        return f"Error: {data.get('Error', 'Unknown error')}"

def parse_srt_excluding_common(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

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

def custom_csv_reader(file_path, imdb_id):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        current_id = None
        current_review_number = None
        current_review = []
        for line in lines[1:]:  # Skip the first line (header)
            if line.startswith('tt'):
                if current_id is not None and current_id == imdb_id:
                    data.append([current_id, current_review_number, ' '.join(current_review)])
                parts = line.split(',', 2)  # Split only on the first two commas
                current_id = parts[0]
                current_review_number = parts[1]
                current_review = [parts[2].strip()]
            else:
                current_review.append(line.strip())
        if current_id is not None and current_id == imdb_id:
            data.append([current_id, current_review_number, ' '.join(current_review)])
    return pd.DataFrame(data, columns=['imdb_id', 'Rating', 'Review'])

def get_movie_reviews(imdb_id, movie_name):
    all_reviews = []
    for i in range(1, 11):
        df = custom_csv_reader(f"{path_imdb_reviews}/reviews_rating_{i}.csv", imdb_id)
        all_reviews.append(df)
    
    # Combine all reviews for this movie into a single DataFrame
    movie_reviews_df = pd.concat(all_reviews, ignore_index=True)
    
    # Create a sanitized filename
    sanitized_name = movie_name.replace(' ', '_').replace(',', '').replace('\'', '').lower()
    
    # Save to CSV file in the imdb-reviews folder
    output_path = f'imdb-reviews/{sanitized_name}.csv'
    movie_reviews_df.to_csv(output_path, index=False)
    print(f"Saved {len(movie_reviews_df)} reviews for {movie_name} to {output_path}")
    
    return movie_reviews_df

# Create a DataFrame with the required information
data = []

for index, row in Movies_df.iterrows():
    movie_name = row['movie']
    year = row['year']
    award = row['award']
    
    # Create the path to the .srt file
    srt_file_name = f"{movie_name}.srt"
    srt_file_path = os.path.join(path_subtitles, "Subtitlesforoscarandblockbusters", award, str(year), srt_file_name)

    if os.path.exists(srt_file_path):
        word_count, total_minutes, top_ten_words, content = parse_srt_excluding_common(srt_file_path)
        
        imdb_id = get_imdb_id(movie_name)
        
        # Get and save reviews for this movie
        if not imdb_id.startswith('Error'):
            movie_reviews = get_movie_reviews(imdb_id, movie_name)
            print(f"Found {len(movie_reviews)} reviews for {movie_name}")
        else:
            print(f"Could not get IMDb ID for {movie_name}: {imdb_id}")

        data.append({
            'movie': movie_name,
            'imdb_id': imdb_id,
            'year': year,
            'award': award,
            'cleaned_subtitle_text': content
        })
        
        # Save individual subtitle file for this movie
        subtitle_df = pd.DataFrame({
            'movie': [movie_name],
            'imdb_id': [imdb_id],
            'year': [year],
            'award': [award],
            'cleaned_subtitle_text': [content]
        })
        
        # Create sanitized filename
        sanitized_name = movie_name.replace(' ', '_').replace(',', '').replace('\'', '').replace('"', '').lower()
        subtitle_output_path = f'data/subtitles/{sanitized_name}.csv'
        subtitle_df.to_csv(subtitle_output_path, index=False)
        print(f"Saved subtitle data for {movie_name} to {subtitle_output_path}")

df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('imdb-reviews/cleaned_subtitles.csv', index=False)
df_without_subtitles = df.drop(columns=['cleaned_subtitle_text'])
df_without_subtitles.to_csv('selected_movie_info.csv', index=False)

