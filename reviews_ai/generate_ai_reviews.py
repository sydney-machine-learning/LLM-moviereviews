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
cleaned_subtitles_df = pd.read_csv('download/cleaned_subtitles_1.csv')


# Filter IMDb reviews to keep only the reviews for the movies in cleaned_subtitles_df
movies_of_interest = cleaned_subtitles_df['imdb_id'].unique()

#imdb_reviews_df = imdb_reviews_df[imdb_reviews_df['MovieID'].isin(movies_of_interest)]



# System context for the AI models
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
    
    prompt = f"{context} {question}. Here is the movie subtitle text: {subtitle_text}"
    truncated_prompt = truncate_text(prompt, 16500)  # Truncate tokens if needed.

    print_token_count(truncated_prompt)

    if ai_client == "gemini":
        response = client_gemini.models.generate_content(
            model="gemini-2.0-flash",  
            contents=truncated_prompt
        )
        return response.text
    elif ai_client == "chatgpt":
        response = client_openai.chat.completions.create(
            model="gpt-3.5-turbo",
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
    for reviewer in ["deepseek"]:
        for context_index, context in enumerate(Contexts, start=1):
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
df.to_csv('reviews_ai/aireviews_deepseek_1.csv', index=False)


