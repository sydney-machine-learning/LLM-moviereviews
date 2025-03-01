
import pandas as pd
import kagglehub
import os

# Dictionary to store imdb reviews grouped by rating
imdb_reviews_dict = {}


# Download from Kaggle
path_subtitles = kagglehub.dataset_download("mlopssss/subtitles")
path_imdb_reviews = kagglehub.dataset_download("mlopssss/imdb-movie-reviews-grouped-by-ratings")

for i in range(1, 11):
    df = pd.read_csv(f"{path_imdb_reviews}/reviews_rating_{i}.csv")
    imdb_reviews_dict[i] = df
    print(df.head())

print(os.listdir(path_subtitles))

subfolder_path = os.path.join(path_subtitles, "Subtitlesforoscarandblockbusters","Blockbusters","1950")
print(os.listdir(subfolder_path)) 

#print("Path to subtitles files:", path_subtitles)

#print("Path to reviews files:", path_imdb_reviews)
