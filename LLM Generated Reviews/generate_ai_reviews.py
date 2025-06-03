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



# System context for the AI models

Context_detailed = ["""
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

# Three user prompt components for full prompt
Prompts = ["Provide a bad review for this movie", 
           "Provide a good review for this movie", 
           "Provide an average review for this movie"]


def generate_review(movie_title, movie_content, context, question, ai_client):
    
    prompt = f"{context} {question}. Here is the text for {movie_title}: {movie_content}"
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


# Process screenplay and subtitle files
def process_movie_files(source_file, output_directory_base, content_field):
    """
    Process all movie CSV files in a directory, model by model.
    For each model, it processes all movies, generates 15 reviews per movie,
    and saves them to a movie-specific CSV in that model's output folder.

    Args:
        source_directory: Directory containing source movie CSV files (e.g., 'data/screenplays')
        output_directory_base: Base directory to save review CSV files (e.g., 'LLM-reviews/screenplays')
        content_field: Field name for the movie content (cleaned_screenplay_text or cleaned_subtitle_text)

    Output CSV Format (per movie, per model):
        - Columns: 'review_id', 'review_text'
        - Each row is one of the 15 generated reviews.
    """

    all_model_configurations = ["gemini_detailed_context","chatgpt", "gemini", "deepseek"]
    
    movie_df = pd.read_csv(source_file)

    for model_config_name in all_model_configurations:
        print(f"\\n=== Processing Model Configuration: {model_config_name} ===")
        data = []
        # Determine the actual AI client and contexts to use
        current_ai_client = model_config_name
        current_contexts_to_use = Contexts
        
        if model_config_name == "gemini_detailed_context":
            current_ai_client = "gemini" # Use the gemini client
            current_contexts_to_use = Context_detailed
        
        
        model_specific_output_dir = output_directory_base
        
        
        for index, row in movie_df.iterrows():                  # loop through movies
            reviews = {}
            # Extract movie information from the row
            movie_name = row['movie']
            year = row['year']
            award = row['award']
            movie_content = row[content_field]
            
            imdb_id = row['imdb_id']
                    
            print(f"\\nProcessing movie: {movie_name} for model configuration: {model_config_name}")

            for context_index, context in enumerate(current_contexts_to_use, start=1):      # loop through 5 contexts
                for question_index, question in enumerate(Prompts, start=1):                    # loop though 3 questions
                    print(movie_name, context_index, question_index, model_config_name)
                    review = generate_review(movie_name, movie_content, context, question, current_ai_client)
                    reviews[f"{current_ai_client}_context{context_index}_question{question_index}"] = review

            data.append({
                        'movie': movie_name,
                        'imdb_id': imdb_id,
                        'year': year,
                        'award': award,
                        **reviews  # Unpack the reviews dictionary into the data dictionary
                    })  

        df = pd.DataFrame(data)
        df.to_csv(f"{output_directory_base}/aireviews_{model_config_name}.csv", index=False)


# Process subtitle files
print("\\n=== Processing Subtitle Files ===")
process_movie_files('data/cleaned_subtitles.csv', 'LLM Generated Reviews/subtitles', 'cleaned_subtitle_text')

# Process screenplay files
print("\\n=== Processing Screenplay Files ===")
process_movie_files('data/cleaned_screenplays.csv', 'LLM Generated Reviews/screenplays', 'cleaned_subtitle_text')


