# This script downloads subtitles from Kaggle, processes them to exclude common words,
# and saves the cleaned subtitles along with movie information to 2 CSV files.


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

        data.append({
            'movie': movie_name,
            'imdb_id': imdb_id,
            'year': year,
            'award': award,
            'cleaned_subtitle_text': content
        })

df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('download/cleaned_subtitles_1.csv', index=False)
df_without_subtitles = df.drop(columns=['cleaned_subtitle_text'])
df_without_subtitles.to_csv('selected_movie_info.csv', index=False)

