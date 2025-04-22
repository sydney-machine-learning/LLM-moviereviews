"""
Trigram Analysis

This script performs trigram analysis on movie reviews from multiple sources:
1. AI-generated reviews (from different AI models like ChatGPT, Deepseek, Gemini)
2. IMDb user reviews


The script processes data through the following workflow:
1. Preprocessing text by tokenizing and removing stopwords/punctuation
2. Generating trigrams  from the processed text
3. Calculating frequency counts for trigrams in each source
4. Saving the results as CSV files
5. Optionally performing analyses for:
   - The Shawshank Redemption movie reviews with rating 10
   - All movies in the dataset


Output files are stored in the trigrams_output directory with naming conventions 
that indicate the source and context of the trigram analysis.

Dependencies: pandas, nltk, collections (Counter), os
"""

import pandas as pd
import nltk
from nltk import trigrams
from collections import Counter
from nltk.corpus import stopwords
import os

# Download necessary NLTK data
# nltk.download('punkt')
# nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

#print("Stop words:", stop_words)

def preprocess_text(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Remove punctuation and stop words
    tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
    return tokens

def generate_trigrams(text):
    tokens = preprocess_text(text)
    trigrams_list = list(trigrams(tokens))
    return trigrams_list

def trigram_analysis(reviews):
    trigram_counts = Counter()
    for review in reviews:
        trigrams_list = generate_trigrams(review)
        trigram_counts.update(trigrams_list)
    return trigram_counts.most_common(3)

def process_reviews(df, ai_client):
    
    results = {}
    for _, row in df.iterrows():
        movie = row['movie']
        results[movie] = {}
        for question_index in range(1, 4):
            question_reviews = row.filter(like=f'{ai_client}_context').filter(like=f'_question{question_index}').tolist()
            trigrams = trigram_analysis(question_reviews)
            results[movie][f'question{question_index}'] = trigrams
    return results

def results_to_dataframe(results):
    data = []
    for movie, questions in results.items():
        for question, trigrams in questions.items():
            for trigram, count in trigrams:
                trigram_str = ' '.join(trigram)
                data.append([movie, question, trigram_str, count])
    return pd.DataFrame(data, columns=['Movie', 'Question', 'Trigram', 'Count'])

def generate_trigrams_for_all_movies(df, movie_info_df):
    results = {}
    
    for movie_id in df['imdb_id'].unique():
        movie_reviews = df[df['imdb_id'] == movie_id]['Review'].tolist()
        trigrams = trigram_analysis(movie_reviews)
        movie_title = movie_info_df[movie_info_df['imdb_id'] == movie_id]['movie'].values[0]
        results[movie_id] = {'title': movie_title, 'trigrams': trigrams}
    return results

def trigrams_to_dataframe(results):
    data = []
    for movie_id, info in results.items():
        movie_title = info['title']
        trigrams = info['trigrams']
        for trigram, count in trigrams:
            trigram_str = ' '.join(trigram)
            data.append([movie_id, movie_title, trigram_str, count])
    return pd.DataFrame(data, columns=['MovieID', 'Title', 'Trigram', 'Count'])


def generate_shawshank_trigrams(selected_movie_info_df, output_file='trigrams_output/shawshank_imdb_10_trigrams.csv'):
    """
    Generate trigrams specifically for "The Shawshank Redemption" from IMDb reviews with a rating of 10.
    
    Args:
        selected_movie_info_df (DataFrame): DataFrame containing movie information
        output_file (str): Path to save the output CSV file
    """
    try:
        # Get the IMDb ID for The Shawshank Redemption
        shawshank_imdb_id = selected_movie_info_df[selected_movie_info_df['movie'] == 'The Shawshank Redemption']['imdb_id'].values[0]
        print(f"Shawshank IMDb ID: {shawshank_imdb_id}")
        # Load all IMDb reviews
        all_imdb_reviews_df = pd.read_csv('download/all_imdb_reviews.csv')

        
        # Filter reviews for Shawshank Redemption with rating 10
        shawshank_reviews = all_imdb_reviews_df[(all_imdb_reviews_df['imdb_id'] == shawshank_imdb_id) & 
                                               (all_imdb_reviews_df['Rating'] == 10)]['Review'].tolist()
        
        if not shawshank_reviews:
            print("No reviews found for The Shawshank Redemption with rating 10")
            return
            
        # Combine all reviews into a single text
        combined_reviews = ' '.join(shawshank_reviews)
        
        # Generate trigrams from the combined reviews
        shawshank_trigrams = trigram_analysis([combined_reviews])
        
        # Convert to DataFrame
        shawshank_trigrams_df = pd.DataFrame(shawshank_trigrams, columns=['Trigram', 'Count'])
        shawshank_trigrams_df['Trigram'] = shawshank_trigrams_df['Trigram'].apply(lambda x: ' '.join(x))
        
        # Save to CSV
        shawshank_trigrams_df.to_csv(output_file, index=False)
        print(f"Shawshank Redemption trigrams saved to {output_file}")
    except Exception as e:
        print(f"Error generating Shawshank trigrams: {e}")

# Function to process and generate trigrams for a given DataFrame and AI client

def process_and_save_trigrams(df, ai_client, output_filename):
    results = process_reviews(df, ai_client)
    #print(f"Results for {ai_client}: {results}")
    df_trigrams = results_to_dataframe(results)
    df_trigrams.to_csv(output_filename, index=False)
    

# Main function to loop through files and generate trigrams

def main(run_shawshank_analysis=False, run_all_movies_analysis=False):
    """
    Main function to process files and generate trigrams.
    
    Args:
        run_shawshank_analysis (bool): Whether to run trigram analysis for The Shawshank Redemption
        run_all_movies_analysis (bool): Whether to run trigram analysis for all movies
    """
    # Get the base directory path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    screenplay_folder = os.path.join(base_dir, 'reviews_ai', 'screenplays')
    subtitles_folder = os.path.join(base_dir, 'reviews_ai', 'subtitles')
    output_folder = os.path.join(base_dir, 'trigrams_output')

    # Process screenplay files
    for filename in os.listdir(screenplay_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(screenplay_folder, filename)
            df = pd.read_csv(file_path)
            output_filename = os.path.join(output_folder, filename.replace('.csv', '_trigrams.csv'))
            ai_client = filename.split('_')[1]  # Extract AI client from filename
            process_and_save_trigrams(df, ai_client, output_filename)

    # Process subtitle files
    for filename in os.listdir(subtitles_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(subtitles_folder, filename)
            df = pd.read_csv(file_path)
            output_filename = os.path.join(output_folder, filename.replace('.csv', '_trigrams.csv'))
            ai_client = filename.split('_')[1].replace('.csv', '')  # Extract AI client from filename without .csv
            process_and_save_trigrams(df, ai_client, output_filename)
    
    # Load movie info dataframe for potential use in the optional analyses
    selected_movie_info_df = pd.read_csv('selected_movie_info.csv')
    
    # Optionally run The Shawshank Redemption analysis
    if run_shawshank_analysis:
        print("Running trigram analysis for The Shawshank Redemption...")
        generate_shawshank_trigrams(selected_movie_info_df)
    
    # Optionally run analysis for all movies
    if run_all_movies_analysis:
        print("Running trigram analysis for all movies...")
        all_imdb_reviews_df = pd.read_csv('download/all_imdb_reviews.csv')
        all_movies_trigrams = generate_trigrams_for_all_movies(all_imdb_reviews_df, selected_movie_info_df)
        all_movies_trigrams_df = trigrams_to_dataframe(all_movies_trigrams)
        all_movies_trigrams_df.to_csv('trigrams_output/all_imdb_review_trigrams.csv', index=False)

if __name__ == "__main__":
    # Set these flags to True to run the respective analyses
    RUN_SHAWSHANK_ANALYSIS = True  # Change to True to run Shawshank Redemption analysis
    RUN_ALL_MOVIES_ANALYSIS = True  # Change to True to run all movies analysis
    
    main(run_shawshank_analysis=RUN_SHAWSHANK_ANALYSIS, 
         run_all_movies_analysis=RUN_ALL_MOVIES_ANALYSIS)

