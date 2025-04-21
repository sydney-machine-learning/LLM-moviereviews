"""
Pairwise Cosine Similarity Analysis for Movie Reviews

This script analyzes the semantic similarity between different types of movie-related text using 
cosine similarity. It processes:
1. IMDb user reviews
2. AI-generated reviews from different models


The analysis follows these steps:
1. Reads review data from IMDb and AI-generated sources
2. Calculates pairwise cosine similarity between different text sources
3. Aggregates similarity metrics (mean, median, max) for each movie
4. Performs targeted comparisons based on:
   - Positive IMDb reviews (rating > 7) vs specific AI responses
   - Negative IMDb reviews (rating < 7) vs specific AI responses
5. Saves the results to CSV files for further analysis

Output files are organized by:
- Movie-level comparisons (cosine_similarity_results_*_by_movie.csv)
- Question-specific comparisons (cosine_similarity_results_*_by_question.csv)

Dependencies: pandas, scikit-learn, numpy, os
"""




import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

def read_reviews(imdb_path, ai_path):
    print(imdb_path, ai_path)
    imdb_reviews_df = pd.read_csv(imdb_path)
    ai_reviews_df = pd.read_csv(ai_path)
    print(imdb_reviews_df.head())
    print(ai_reviews_df.head())
    return imdb_reviews_df, ai_reviews_df

def compute_pairwise_cosine_similarity(imdb_reviews, ai_reviews):
    vectorizer = TfidfVectorizer()
    imdb_vectors = vectorizer.fit_transform(imdb_reviews)
    ai_vectors = vectorizer.transform(ai_reviews)
    
    similarities = cosine_similarity(imdb_vectors, ai_vectors)
    return similarities

def compute_within_imdb_similarity(imdb_reviews):
    vectorizer = TfidfVectorizer()
    imdb_vectors = vectorizer.fit_transform(imdb_reviews)
    
    similarities = cosine_similarity(imdb_vectors, imdb_vectors)
    
    # Exclude self-comparisons by setting the diagonal to 0
    np.fill_diagonal(similarities, 0)
    
    return similarities

def aggregate_similarities(similarities):
    mean_similarity = np.mean(similarities)
    median_similarity = np.median(similarities)
    max_similarity = np.max(similarities)
    return mean_similarity, median_similarity, max_similarity

def main():
    # Get the base directory path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    imdb_path = os.path.join(base_dir, 'download', 'all_imdb_reviews.csv')
       
    # Process reviews in screenplays and subtitles folders separately
    folders = [
        (os.path.join(base_dir, 'reviews_ai', 'screenplays'), os.path.join(base_dir, 'cosine_similarity_and_other_tests', 'cosine_similarity_results_screenplays')),
        (os.path.join(base_dir, 'reviews_ai', 'subtitles'), os.path.join(base_dir, 'cosine_similarity_and_other_tests', 'cosine_similarity_results_subtitles'))
    ]

    for folder_path, output_file_prefix in folders:
        results_by_movie = []
        results_by_question = []
        
        # Process each CSV file in the folder
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):
                ai_file_path = os.path.join(folder_path, file_name)
                print(f"Processing file: {ai_file_path}")
                
                imdb_reviews_df, ai_reviews_df = read_reviews(imdb_path, ai_file_path)

                for movie_id in imdb_reviews_df['imdb_id'].unique():
                    print(f"Processing imdb_id: {movie_id}")  # Updated to uncomment the print statement

                    imdb_reviews = imdb_reviews_df[imdb_reviews_df['imdb_id'] == movie_id]['Review'].tolist()
                    ai_reviews = ai_reviews_df[ai_reviews_df['imdb_id'] == movie_id].filter(like='context').values.flatten().tolist()

                    if not imdb_reviews or not ai_reviews:
                        print(f"No reviews found for imdb_id: {movie_id}")
                        continue

                    # Compute pairwise cosine similarity
                    similarities = compute_pairwise_cosine_similarity(imdb_reviews, ai_reviews)
                    mean_similarity, median_similarity, max_similarity = aggregate_similarities(similarities)

                    # Append results by movie
                    results_by_movie.append({
                        'File': file_name,
                        'imdb_id': movie_id,
                        'MeanSimilarity': mean_similarity,
                        'MedianSimilarity': median_similarity,
                        'MaxSimilarity': max_similarity
                    })

                    # Group results by question
                    for question_index in range(1, 3):
                        # Get AI reviews specifically for this question index
                        ai_question_reviews = []
                        for col in ai_reviews_df.columns:
                            if f'_question{question_index}' in col and 'context' in col:
                                if ai_reviews_df[ai_reviews_df['imdb_id'] == movie_id][col].values:
                                    ai_question_reviews.append(ai_reviews_df[ai_reviews_df['imdb_id'] == movie_id][col].values[0])
                        
                        if not ai_question_reviews:
                            print(f"No AI reviews found for imdb_id: {movie_id}, Question: {question_index}")
                            continue
                        
                        # Filter IMDb reviews by rating
                        if question_index == 1:
                            filtered_imdb_reviews = imdb_reviews_df[(imdb_reviews_df['imdb_id'] == movie_id) & (imdb_reviews_df['Rating'] < 7)]['Review'].tolist()
                            compare_type = "negative IMDb reviews (rating < 7)"
                        elif question_index == 2:
                            filtered_imdb_reviews = imdb_reviews_df[(imdb_reviews_df['imdb_id'] == movie_id) & (imdb_reviews_df['Rating'] > 7)]['Review'].tolist()
                            compare_type = "positive IMDb reviews (rating > 7)"

                        if not filtered_imdb_reviews:
                            print(f"No {compare_type} found for imdb_id: {movie_id}")
                            continue

                        question_similarities = compute_pairwise_cosine_similarity(filtered_imdb_reviews, ai_question_reviews)
                        mean_q_similarity, median_q_similarity, max_q_similarity = aggregate_similarities(question_similarities)

                        results_by_question.append({
                            'File': file_name,
                            'imdb_id': movie_id,
                            'Question': f'Question{question_index}',
                            'ComparedWith': compare_type,
                            'MeanSimilarity': mean_q_similarity,
                            'MedianSimilarity': median_q_similarity,
                            'MaxSimilarity': max_q_similarity
                        })

        # Save results by movie to CSV
        results_by_movie_df = pd.DataFrame(results_by_movie)
        results_by_movie_df.to_csv(f"{output_file_prefix}_by_movie.csv", index=False)

        # Save results by question to CSV
        results_by_question_df = pd.DataFrame(results_by_question)
        results_by_question_df.to_csv(f"{output_file_prefix}_by_question.csv", index=False)

        print(f"Results saved to {output_file_prefix}_by_movie.csv and {output_file_prefix}_by_question.csv")

if __name__ == "__main__":
    main()