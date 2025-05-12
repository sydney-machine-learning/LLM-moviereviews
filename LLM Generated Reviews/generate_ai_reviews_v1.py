# Script to generate AI  movie reviews using multiple models and contexts


import pandas as pd
import os
from google import genai
from openai import OpenAI
from dotenv import load_dotenv

from transformers import GPT2Tokenizer
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

# Create model-specific subdirectories
for content_type in ['screenplays', 'subtitles']:
    for model in ['chatgpt', 'gemini', 'deepseek', 'gemini_detailed_context']:
        os.makedirs(f'LLM-reviews/{content_type}/{model}', exist_ok=True)

# Five detailed context inputs for Gemini
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
school's film club. They're very online, love A24 movies, and aren't afraid to 
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

# Five simple context inputs
Context1 = "You are a young woman and go to the movies a lot and you enjoy providing honest informative reviews."
Context2 = "You are a professional movie critic and you enjoy providing honest informative reviews."
Context3 = "You are a sometimes aggressive man and an action movie buff and you enjoy providing honest informative reviews."
Context4 = "You are an online troll and provide negative reviews for fun unless you really like the movie."
Context5 = "You are a right-wing extremist and movie buff and enjoy providing reviews."
Contexts = [Context1, Context2, Context3, Context4, Context5]

def truncate_text(text, max_tokens):
    tokens = tokenizer.encode(text)
    print(f"Number of tokens: {len(tokens)}")
    if len(tokens) > max_tokens:
        print(f"Truncating text to {max_tokens} tokens.")
        return tokenizer.decode(tokens[:max_tokens])
    return text

# Three user prompt components for the full prompt
Prompts = [
    "Write a bad review for this movie.",
    "Write a good review for this movie.", 
    "Write an average review for this movie."
]           

def generate_review(movie_title, movie_content, context, question, ai_client):
    
    prompt = f"{context} {question}. Here is the movie text for the movie {movie_title}: {movie_content}"
    prompt = truncate_text(prompt, 80000)  # Truncate tokens if needed.

    print_token_count(prompt)

    if ai_client == "gemini":
        response = client_gemini.models.generate_content(
            model="gemini-2.0-flash",  
            contents= prompt
        )
        return response.text
    elif ai_client == "chatgpt":
        response = client_openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    elif ai_client == "deepseek":
        response = client_deepseek.chat.completions.create(
            #model="deepseek-v",
            model="deepseek-chat",  
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


# Process screenplay and subtitle files
def process_movie_files(source_directory, output_directory_base, content_field):
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
    #all_model_configurations = ["gemini_detailed_context","chatgpt", "gemini", "deepseek"]
    all_model_configurations = ["gemini_detailed_context"]
    # Check if source directory exists
    if not os.path.exists(source_directory):
        print(f"Source directory {source_directory} not found. Skipping.")
        return

    for model_config_name in all_model_configurations:
        print(f"\\n=== Processing Model Configuration: {model_config_name} ===")
        
        # Determine the actual AI client and contexts to use
        current_ai_client = model_config_name
        current_contexts_to_use = Contexts
        
        if model_config_name == "gemini_detailed_context":
            current_ai_client = "gemini" # Use the gemini client
            current_contexts_to_use = Context_detailed
        
        model_specific_output_dir = os.path.join(output_directory_base, model_config_name)
        # Ensure the specific model's output directory exists (it should have been created at the start of the script)

        for filename in os.listdir(source_directory):
            if filename.endswith('anic.csv'):
            #if filename.endswith('.csv'):    
                file_path = os.path.join(source_directory, filename)
                
                try:
                    movie_df = pd.read_csv(file_path, header=0)
                    if len(movie_df) == 0:
                        print(f"Empty file: {filename} for model {model_config_name}. Skipping.")
                        continue
                    
                    movie_row = movie_df.iloc[0]
                    movie_name = movie_row['movie']
                    movie_content = movie_row[content_field]
                    
                    print(f"\\nProcessing movie: {movie_name} for model configuration: {model_config_name}")
                    
                    sanitized_name = movie_name.lower().replace(' ', '_').replace(',', '').replace('\'', '').replace('"', '')
                    
                    reviews_for_csv = [] # List to hold dictionaries: {'review_id': ..., 'review_text': ...}

                    for context_index, context_text in enumerate(current_contexts_to_use, start=1):
                        for question_index, question_text in enumerate(Prompts, start=1):
                            review_id = f"context{context_index}_question{question_index}"
                            print(f"Generating {model_config_name} review ({review_id}) for {movie_name}...")
                            
                            review_text_content = generate_review(movie_name, movie_content, context_text, question_text, current_ai_client)
                            
                            reviews_for_csv.append({
                                'review_id': review_id,
                                'review_text': review_text_content
                            })
                    
                    # Create DataFrame for the current movie's 15 reviews for this model_config_name
                    reviews_df = pd.DataFrame(reviews_for_csv)
                    
                    output_csv_path = os.path.join(model_specific_output_dir, f"{sanitized_name}.csv")
                    reviews_df.to_csv(output_csv_path, index=False)
                    print(f"Saved {model_config_name} reviews for {movie_name} to {output_csv_path}")

                except Exception as e:
                    print(f"Error processing {filename} for model {model_config_name}: {e}")
                    # Optionally, continue to the next file or model, or handle more gracefully
                    continue # Continue to the next file in the source directory

# Process screenplay files
#print("\\n=== Processing Screenplay Files ===")
#process_movie_files('data/screenplays', 'LLM-reviews/screenplays', 'cleaned_screenplay_text')

# Process subtitle files
print("\\n=== Processing Subtitle Files ===")
process_movie_files('data/subtitles', 'LLM-reviews/subtitles', 'cleaned_subtitle_text')


