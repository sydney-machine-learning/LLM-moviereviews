import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

def read_reviews(imdb_path, ai_path):
    print(imdb_path, ai_path)
    imdb_reviews_df = pd.read_csv(imdb_path, header=None, names=['MovieID', 'Rating', 'Review'])
    ai_reviews_df = pd.read_csv(ai_path)
    #print(imdb_reviews_df.head())
    #print(ai_reviews_df.head())
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
    
    imdb_path = '../download/all_imdb_reviews.csv'
    ''' 
    ai_path = 'reviews_ai/aireviews_gemini_more_detailed_context.csv'
    
    imdb_reviews_df, ai_reviews_df = read_reviews(imdb_path, ai_path)
    
    results = []
    within_imdb_results = []
    
    for movie_id in imdb_reviews_df['MovieID'].unique():
        print(f"Processing MovieID: {movie_id}")
        
        imdb_reviews = imdb_reviews_df[imdb_reviews_df['MovieID'] == movie_id]['Review'].tolist()
        ai_reviews = ai_reviews_df[ai_reviews_df['imdb_id'] == movie_id].filter(like='context').values.flatten().tolist()
        
        if not imdb_reviews or not ai_reviews:
            print("No reviews found for this MovieID.")
            continue
        
        similarities = 1 #compute_pairwise_cosine_similarity(imdb_reviews, ai_reviews)
        
        mean_similarity, median_similarity, max_similarity = aggregate_similarities(similarities)
        print(f"Mean Similarity: {mean_similarity}, Median Similarity: {median_similarity}, Max Similarity: {max_similarity}")
        results.append({
            'MovieID': movie_id,
            'MeanSimilarity': mean_similarity,
            'MedianSimilarity': median_similarity,
            'MaxSimilarity': max_similarity
        })
        
        # Compute within IMDb similarity
        within_imdb_similarities = 1 #compute_within_imdb_similarity(imdb_reviews)
        mean_within_similarity, median_within_similarity, max_within_similarity = aggregate_similarities(within_imdb_similarities)
        print(f"Within IMDb - Mean Similarity: {mean_within_similarity}, Median Similarity: {median_within_similarity}, Max Similarity: {max_within_similarity}")
        within_imdb_results.append({
            'MovieID': movie_id,
            'MeanWithinSimilarity': mean_within_similarity,
            'MedianWithinSimilarity': median_within_similarity,
            'MaxWithinSimilarity': max_within_similarity
        })

    # Save results to CSV
    #results_df = pd.DataFrame(results)
    #results_df.to_csv('cosine_similarity_and_other_tests/cosine_similarity_results_gemini_detailed_context.csv', index=False)
    #print(results_df)
    
    #within_imdb_results_df = pd.DataFrame(within_imdb_results)
    #within_imdb_results_df.to_csv('cosine_similarity_and_other_tests/within_imdb_similarity_results.csv', index=False)
    #print(within_imdb_results_df)
    '''
    # Process reviews in screenplays and subtitles folders separately
    folders = [
        ('../reviews_ai/screenplays', 'cosine_similarity_and_other_tests/cosine_similarity_results_screenplays'),
        ('../reviews_ai/subtitles', 'cosine_similarity_and_other_tests/cosine_similarity_results_subtitles')
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

                for movie_id in imdb_reviews_df['MovieID'].unique():
                    # print(f"Processing MovieID: {movie_id}")

                    imdb_reviews = imdb_reviews_df[imdb_reviews_df['MovieID'] == movie_id]['Review'].tolist()
                    ai_reviews = ai_reviews_df[ai_reviews_df['imdb_id'] == movie_id].filter(like='context').values.flatten().tolist()

                    if not imdb_reviews or not ai_reviews:
                        print(f"No reviews found for MovieID: {movie_id}")
                        continue

                    # Compute pairwise cosine similarity
                    similarities = compute_pairwise_cosine_similarity(imdb_reviews, ai_reviews)
                    mean_similarity, median_similarity, max_similarity = aggregate_similarities(similarities)

                    # Append results by movie
                    results_by_movie.append({
                        'File': file_name,
                        'MovieID': movie_id,
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
                            print(f"No AI reviews found for MovieID: {movie_id}, Question: {question_index}")
                            continue
                        
                        # Filter IMDb reviews by rating
                        if question_index == 1:
                            filtered_imdb_reviews = imdb_reviews_df[(imdb_reviews_df['MovieID'] == movie_id) & (imdb_reviews_df['Rating'] < 7)]['Review'].tolist()
                            compare_type = "negative IMDb reviews (rating < 7)"
                        elif question_index == 2:
                            filtered_imdb_reviews = imdb_reviews_df[(imdb_reviews_df['MovieID'] == movie_id) & (imdb_reviews_df['Rating'] > 7)]['Review'].tolist()
                            compare_type = "positive IMDb reviews (rating > 7)"

                        if not filtered_imdb_reviews:
                            print(f"No {compare_type} found for MovieID: {movie_id}")
                            continue

                        question_similarities = compute_pairwise_cosine_similarity(filtered_imdb_reviews, ai_question_reviews)
                        mean_q_similarity, median_q_similarity, max_q_similarity = aggregate_similarities(question_similarities)

                        results_by_question.append({
                            'File': file_name,
                            'MovieID': movie_id,
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