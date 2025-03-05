import pandas as pd
import kagglehub
import re
import os
from collections import Counter
from google import genai
import requests

# Download from Kaggle
path_subtitles = kagglehub.dataset_download("mlopssss/subtitles")

# For gemini connection
gkey = "AIzaSyB4ys06i3HZcvSgsIOMO1n0VCroTVal5oI"
API_KEY = "9a116dc6"
client = genai.Client(api_key=gkey)

common_exclusions = {'-','♪','i', 'you', 'to', 'the', 'a', 'and', 'it', 'is', 'that', 'of','s', 't', 'what', 'in', 'me', 'this', 'on', 'sir', 'get','for', 'she', 'be', 'eve', 'not', 'have', 'all', 'her', 'was', 'my','can', 'oh', 'no', 'we', 'well', 'annie', 'be', 'he', 'like', 'don'}

Movies_df = pd.DataFrame([
    ["The Shawshank Redemption", 1995, "Oscar"],
    ["Brokeback Mountain", 2006, "Oscar"],
    ["Avatar", 2010, "Oscar"],
    ["Titanic", 1998, "Oscar"],
    ["Crouching Tiger, Hidden Dragon", 2001, "Oscar"],
    ["Nomadland", 2021, "Oscar"]
], columns=["movie", "year", "award"])

# Function to get IMDb ID from movie title
def get_imdb_id(movie_title):
    url = f"http://www.omdbapi.com/?t={movie_title}&apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    
    if response.status_code == 200 and data["Response"] == "True":
        return data["imdbID"]
    else:
        return f"Error: {data.get('Error', 'Unknown error')}"

def generate_gemini_review(movie_title, subtitle_text, question):
    prompt = f"You are a movie reviewer and you can make funny responses. {question}: {subtitle_text}"
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",  
        contents=prompt,
    )
    return response.text

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
        
        question = "Provide a bad review of this movie"
        gemini_review = generate_gemini_review(movie_name, content, question)
        imdb_id = get_imdb_id(movie_name)

        data.append({
            'movie': movie_name,
            'imdb_id': imdb_id,
            'year': year,
            'award': award,
            'numberofwords': word_count,
            'time': f"{total_minutes:.2f}mins",
            'toptenwords': top_ten_words,
            'bodyContent': content,
            'geminiReview': gemini_review
        })

df = pd.DataFrame(data)

# Print the DataFrame to verify the changes
print(df.head(1))

# Save the DataFrame to a CSV file if needed
df.to_csv('movies_with_subtitles_and_aireviews.csv', index=False)


