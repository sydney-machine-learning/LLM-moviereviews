import kagglehub
import pandas as pd

# Dictionary to store imdb reviews grouped by rating
imdb_reviews_dict = {}

# list of unique imdb ids
Unique_IMDB_ids = set()  

# List to store all DataFrames of reviews
all_reviews = []

path_imdb_reviews = kagglehub.dataset_download("mlopssss/imdb-movie-reviews-grouped-by-ratings")

def custom_csv_reader(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        current_id = None
        current_review_number = None
        current_review = []
        for line in lines[1:]:  # Skip the first line (header)
            if line.startswith('tt'):
                if current_id is not None:
                    data.append([current_id, current_review_number, ' '.join(current_review)])
                parts = line.split(',', 2)  # Split only on the first two commas
                current_id = parts[0]
                current_review_number = parts[1]
                current_review = [parts[2].strip()]
            else:
                current_review.append(line.strip())
        if current_id is not None:
            data.append([current_id, current_review_number, ' '.join(current_review)])
    return pd.DataFrame(data, columns=['MovieID', 'ReviewNumber', 'Review'])

for i in range(1, 11):
    #df = pd.read_csv(f"{path_imdb_reviews}/reviews_rating_{i}.csv")
    df = custom_csv_reader(f"{path_imdb_reviews}/reviews_rating_{i}.csv")
    imdb_reviews_dict[i] = df
    Unique_IMDB_ids.update(df["MovieID"].tolist())
    all_reviews.append(df)

# Combine all reviews into a single DataFrame and save to csv
all_reviews_df = pd.concat(all_reviews, ignore_index=True)
all_reviews_df.to_csv('all_imdb_reviews.csv', index=False, header=False)
