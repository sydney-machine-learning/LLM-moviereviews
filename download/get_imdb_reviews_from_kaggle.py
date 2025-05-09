"""
IMDb Reviews Processing Pipeline

This script performs the following operations:
1. Downloads IMDb movie reviews from Kaggle, grouped by rating (1-10)
2. Processes reviews using one of two approaches:
   - Filtering for only reviews of specific movies identified in 'selected_movie_info.csv'
   - Downloading and processing all reviews in the dataset (when download_all_reviews=True)
3. Handles the non-standard CSV format using a custom CSV reader that:
   - Parses multi-line reviews correctly
   - Joins review text that spans multiple lines
   - Filters reviews based on relevant IMDb IDs when specified
4. Organizes reviews in a dictionary (imdb_reviews_dict) grouped by rating (1-10)
5. Tracks unique IMDb IDs throughout the processing
6. Combines all reviews into a single DataFrame
7. Saves the final results to 'download/all_imdb_reviews.csv'

Dependencies: pandas, kagglehub
Input files: selected_movie_info.csv (when filtering by specific movies)
Output files: download/all_imdb_reviews.csv
"""

import kagglehub
import pandas as pd

# Flag to determine whether to download all reviews or just the relevant ones
download_all_reviews = False  # Set to True to download all reviews

# Dictionary to store imdb reviews grouped by rating
imdb_reviews_dict = {}

# list of unique imdb ids
Unique_IMDB_ids = set()  

# List to store all DataFrames of reviews
all_reviews = []

if not download_all_reviews:
    # Read the selected_movie_info.csv to get the list of relevant IMDb codes
    selected_movie_info_df = pd.read_csv('selected_movie_info.csv')
    relevant_imdb_codes = set(selected_movie_info_df['imdb_id'].tolist())
    print(f"Relevant IMDb codes: {relevant_imdb_codes}")
else:
    relevant_imdb_codes = None
print(f"Relevant IMDb codes: {relevant_imdb_codes}")



path_imdb_reviews = kagglehub.dataset_download("mlopssss/imdb-movie-reviews-grouped-by-ratings")

def custom_csv_reader(file_path, relevant_codes):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        current_id = None
        current_review_number = None
        current_review = []
        for line in lines[1:]:  # Skip the first line (header)
            if line.startswith('tt'):
                if current_id is not None and (relevant_codes is None or current_id in relevant_codes):
                    data.append([current_id, current_review_number, ' '.join(current_review)])
                parts = line.split(',', 2)  # Split only on the first two commas
                current_id = parts[0]
                current_review_number = parts[1]
                current_review = [parts[2].strip()]
            else:
                current_review.append(line.strip())
        if current_id is not None and (relevant_codes is None or current_id in relevant_codes):
            data.append([current_id, current_review_number, ' '.join(current_review)])
    return pd.DataFrame(data, columns=['imdb_id', 'Rating', 'Review'])

for i in range(1, 11):
    df = custom_csv_reader(f"{path_imdb_reviews}/reviews_rating_{i}.csv", relevant_imdb_codes)
    imdb_reviews_dict[i] = df
    Unique_IMDB_ids.update(df["imdb_id"].tolist())
    all_reviews.append(df)

# Combine all reviews into a single DataFrame and save to csv
all_reviews_df = pd.concat(all_reviews, ignore_index=True)
all_reviews_df.to_csv('download/all_imdb_reviews.csv', index=False, header=True)
