import pandas as pd
import kagglehub
import os
from google import genai
from openai import OpenAI
import requests
from dotenv import load_dotenv

from transformers import GPT2Tokenizer, GPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def print_token_count(prompt):
    tokens = tokenizer.encode(prompt)
    print(f"Number of tokens: {len(tokens)}")

load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DEEPSEEK_API_KEY = os.getenv('DEEP_AI_API_KEY')

# Initialize the API clients
client_gemini = genai.Client(api_key=GEMINI_API_KEY)
client_openai = OpenAI(api_key=OPENAI_API_KEY)
client_deepseek = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")



# Read the CSV files
#imdb_reviews_df = pd.read_csv('all_imdb_reviews.csv',names=['MovieID','Rating','Review'])
#cleaned_subtitles_df = pd.read_csv('download/cleaned_subtitles.csv')
cleaned_subtitles_df = pd.read_csv('data/cleaned_screenplays.csv')

# Filter IMDb reviews to keep only the reviews for the movies in cleaned_subtitles_df
movies_of_interest = cleaned_subtitles_df['imdb_id'].unique()

#imdb_reviews_df = imdb_reviews_df[imdb_reviews_df['MovieID'].isin(movies_of_interest)]



# System context for the AI models

Context_a = ["""
I have subtitle text from a movie. Please read the subtitles and generate
a short movie review written in the voice of a young professional woman in
her late 20s who loves going to the movies and has a sharp, thoughtful 
perspective. She often posts casual but insightful reviews on Letterboxd. 
Her tone is witty, self-aware, and a little emotionally vulnerable. Write 
a review of this movie from her point of view, focusing on the emotional beats, 
themes, characters, the actor performances, and overall experience of watching the film. Make it sound 
like something she might post online the night after seeing it.
""",
"""
I have subtitle text from a movie. Please read the subtitles and generate
a short movie review written in the voice of a professional film critic in
their late 60s who has a dry, to the point, and somewhat cynical perspective. 
Their tone is witty, self-aware, and cuts to the point. Write 
a review of this movie from their point of view, as if they just watched the movie, focusing on the emotional beats, 
themes, characters, the actor performances, and overall experience of watching the film. Make it sound 
like something they might post in an ibdb review after seeing it.
""",
"""I have subtitle text from a movie. Please read the subtitles and generate a movie 
review written in the voice of a 17-year-old high school student who runs their 
school’s film club. They’re very online, love A24 movies, and aren’t afraid to 
speak their mind. Their tone is fast-paced, funny, and packed with pop culture 
references, while still showing an impressive grasp of film analysis. Write a 
review of this movie from their point of view, as if they just watched the movie, mixing humor with surprisingly 
sharp takes on themes, symbolism, and character arcs. Feel free to comment on the actor performances.""",
"""
I have subtitle text from a movie. Please read the subtitles and generate
a short movie review written in the voice of a young right wing man in his 30s. 
His tone is slightly angry, and he is a little aggressive. He rarely gives 
positive reviews, prefering to focus on the negative. Provide a review as if you
 had just been to see the movie. You express comments on the actors,
characters, and the quality of the plot.""",
"""
I have subtitle text from a movie. Please read the subtitles and generate
a short movie review written in the voice of a older professional woman in
her late 40s who loves going to the movies and has a thoughtful and humourous
perspective. Her tone is dry, and a little controversial. Write 
a review of this movie from her point of view, focusing on the 
themes, characters, the actor performances, and overall experience of watching 
the film. Make it sound like something she might post online the night after seeing it.
"""]


Context1 = "You are a young woman and go to the movies a lot and you enjoy providing honest informative reviews."
Context2 = "You are a professional movie critic and you enjoy providing honest informative reviews."
Context3 = "You are a sometimes aggressive man and an action movie buff and you enjoy providing honest informative reviews."
Context4 = "You are an online troll and provide negative reviews for fun unless you really like the movie."
Context5 = "You are a right-wing extremist and movie buff and enjoy providing reviews."
Contexts = [Context1, Context2, Context3, Context4, Context5]

def truncate_text(text, max_tokens):
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        return tokenizer.decode(tokens[:max_tokens])
    return text

# User prompts for the AI models
Prompts = ["Provide a bad review for this movie", "Provide a good review for this movie", "Provide an average review for this movie"]


def generate_review(movie_title, subtitle_text, context, question, ai_client):
    
    prompt = f"{context} {question}. Here is the text for {movie_title}: {subtitle_text}"
    truncated_prompt = truncate_text(prompt, 84000)  # Truncate tokens if needed.

    print_token_count(truncated_prompt)

    if ai_client == "gemini":
        response = client_gemini.models.generate_content(
            model="gemini-2.0-flash",  
            contents=truncated_prompt
        )
        return response.text
    elif ai_client == "chatgpt":
        response = client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": truncated_prompt}],
        )
        return response.choices[0].message.content
    elif ai_client == "deepseek":
        response = client_deepseek.chat.completions.create(
            model="deepseek-chat",  
            messages=[{"role": "user", "content": truncated_prompt}],
        )
        return response.choices[0].message.content


# Create a DataFrame with the required information
data = []

for index, row in cleaned_subtitles_df.iterrows():
    movie_name = row['movie']
    year = row['year']
    award = row['award']
    content = row['cleaned_subtitle_text']
    

    
    reviews = {}
    for reviewer in ["gemini"]:
        for context_index, context in enumerate(Context_a, start=1):
            for question_index, question in enumerate(Prompts, start=1):
                print(movie_name, context_index, question_index, reviewer)
                review = generate_review(movie_name, content, context, question, reviewer)
                reviews[f"{reviewer}_context{context_index}_question{question_index}"] = review

                
    imdb_id = row['imdb_id']

    data.append({
        'movie': movie_name,
        'imdb_id': imdb_id,
        'year': year,
        'award': award,
        **reviews  # Unpack the reviews dictionary into the data dictionary
    })

df = pd.DataFrame(data)

# Print the DataFrame to verify the changes
print(df.head(1))

# Save the DataFrame to a CSV file if needed
df.to_csv('LLM Generated Reviews/screenplays/aireviews_gemini_screenplays.csv', index=False)


