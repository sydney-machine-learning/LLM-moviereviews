import pandas as pd
import kagglehub
import re
import os
from collections import Counter
from google import genai

# Download from Kaggle
path_subtitles = kagglehub.dataset_download("mlopssss/subtitles")

#for gemini connection
gkey = "AIzaSyB4ys06i3HZcvSgsIOMO1n0VCroTVal5oI"
client = genai.Client(api_key=gkey)

common_exclusions = {'-','♪','i', 'you', 'to', 'the', 'a', 'and', 'it', 'is', 'that', 'of','s', 't', 'what', 'in', 'me', 'this', 'on', 'sir', 'get','for', 'she', 'be', 'eve', 'not', 'have', 'all', 'her', 'was', 'my','can', 'oh', 'no', 'we', 'well', 'annie', 'be', 'he', 'like', 'don'}
years = [1950,1960,1970,1980,1990,2000,2010,2020]

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

    return word_count, total_minutes, top_ten_words,lines

#print(os.listdir(path_subtitles))

for group in ['Oscar', 'Blockbusters']:
    data = []
    for year in years:
        subfolder_path = os.path.join(path_subtitles, "Subtitlesforoscarandblockbusters",group,str(year))
    
        movie_titles = os.listdir(subfolder_path)
        for srt_file in movie_titles[:2]:  # Only the first 2 movie titles for each year
            movie_path = os.path.join(subfolder_path, srt_file)
            movie_name, _ = os.path.splitext(srt_file)
        #print(year, movie_name, movie_path)

            word_count, total_minutes, top_ten_words , content = parse_srt_excluding_common(movie_path)
            
            question = "Provide a review of this movie"
            gemini_review = generate_gemini_review(movie_name, content, question)
            
            data.append({
                        'movie': movie_name,
                        'year': year,
                        'numberofwords': word_count,
                        'time': f"{total_minutes:.2f}mins",
                        'toptenwords': top_ten_words,
                        'bodyContent': content,
                        'geminiReview': gemini_review
                    })
    df = pd.DataFrame(data)

    print(df[['movie', 'year','geminiReview']].tail(2))
    
    # Save Blockbuster movie titles and years to a CSV file
    #df[['movie', 'year','bodyContent']].to_csv(f'{group.lower()}_movies_and_years.csv', index=False, header=False)


