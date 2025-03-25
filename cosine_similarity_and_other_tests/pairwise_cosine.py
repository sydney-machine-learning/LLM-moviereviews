import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def read_reviews(imdb_path, ai_path):
    imdb_reviews_df = pd.read_csv(imdb_path, header=None, names=['MovieID', 'Rating', 'Review'])
    ai_reviews_df = pd.read_csv(ai_path)
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
    imdb_path = 'download/all_imdb_reviews.csv'
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
        
        similarities = compute_pairwise_cosine_similarity(imdb_reviews, ai_reviews)
        
        mean_similarity, median_similarity, max_similarity = aggregate_similarities(similarities)
        print(f"Mean Similarity: {mean_similarity}, Median Similarity: {median_similarity}, Max Similarity: {max_similarity}")
        results.append({
            'MovieID': movie_id,
            'MeanSimilarity': mean_similarity,
            'MedianSimilarity': median_similarity,
            'MaxSimilarity': max_similarity
        })
        
        # Compute within IMDb similarity
        within_imdb_similarities = compute_within_imdb_similarity(imdb_reviews)
        mean_within_similarity, median_within_similarity, max_within_similarity = aggregate_similarities(within_imdb_similarities)
        print(f"Within IMDb - Mean Similarity: {mean_within_similarity}, Median Similarity: {median_within_similarity}, Max Similarity: {max_within_similarity}")
        within_imdb_results.append({
            'MovieID': movie_id,
            'MeanWithinSimilarity': mean_within_similarity,
            'MedianWithinSimilarity': median_within_similarity,
            'MaxWithinSimilarity': max_within_similarity
        })

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('cosine_similarity_and_other_tests/cosine_similarity_results_gemini_detailed_context.csv', index=False)
    print(results_df)
    
    #within_imdb_results_df = pd.DataFrame(within_imdb_results)
    #within_imdb_results_df.to_csv('cosine_similarity_and_other_tests/within_imdb_similarity_results.csv', index=False)
    #print(within_imdb_results_df)

if __name__ == "__main__":
    main()