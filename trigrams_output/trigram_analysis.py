# This script will perform several sentiment analysis tests on movie reviews output from various AI APIs versus human-generated reviews
# from IMDb. The goal is to determine the accuracy of the AI-generated reviews and compare
# them to human-generated reviews. The script will output the results of the sentiment analysis.

import pandas as pd
import nltk
from nltk import trigrams
from collections import Counter
import string
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
    print("Processing reviews for AI client:", ai_client)
    print("DataFrame head:", df.head())
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
    for movie_id in df['MovieID'].unique():
        movie_reviews = df[df['MovieID'] == movie_id]['Review'].tolist()
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



# Code to generate trigrams for "The Shawshank Redemption" from IMDb reviews with a 10 rating


  
#shawshank_imdb_id = selected_movie_info_df[selected_movie_info_df['movie'] == 'The Shawshank Redemption']['imdb_id'].values[0]
#all_imdb_reviews_df = pd.read_csv('download/all_imdb_reviews.csv')
#shawshank_reviews = all_imdb_reviews_df[(all_imdb_reviews_df['MovieID'] == shawshank_imdb_id) & (all_imdb_reviews_df['ReviewNumber'] == 10)]['Review'].tolist()
#combined_reviews = ' '.join(shawshank_reviews)
# Generate trigrams from the combined reviews
##shawshank_trigrams = trigram_analysis([combined_reviews])
##shawshank_trigrams_df = pd.DataFrame(shawshank_trigrams, columns=['Trigram', 'Count'])
##shawshank_trigrams_df['Trigram'] = shawshank_trigrams_df['Trigram'].apply(lambda x: ' '.join(x))
##shawshank_trigrams_df.to_csv('shawshank_imdb_10_trigrams.csv', index=False)

# Function to process and generate trigrams for a given DataFrame and AI client

def process_and_save_trigrams(df, ai_client, output_filename):
    results = process_reviews(df, ai_client)
    #print(f"Results for {ai_client}: {results}")
    df_trigrams = results_to_dataframe(results)
    df_trigrams.to_csv(output_filename, index=False)
    

# Main function to loop through files and generate trigrams

def main():
    screenplay_folder = 'reviews_ai/screenplays'
    subtitles_folder = 'reviews_ai/subtitles'
    output_folder = 'trigrams_output'

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
    selected_movie_info_df = pd.read_csv('selected_movie_info.csv')
    
    #all_imdb_reviews_df = pd.read_csv('download/all_imdb_reviews.csv')
    #all_movies_trigrams = generate_trigrams_for_all_movies(all_imdb_reviews_df, selected_movie_info_df)
    #all_movies_trigrams_df = trigrams_to_dataframe(all_movies_trigrams)
    #all_movies_trigrams_df.to_csv('trigrams_output/all_imdb_review_trigrams.csv', index=False)

if __name__ == "__main__":
    main()
