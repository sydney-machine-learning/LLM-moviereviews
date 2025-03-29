import pandas as pd
import numpy as np


np.random.seed(20)

# Load IMDb reviews
imdb_df = pd.read_csv("download/all_imdb_reviews.csv", names=["imdb_id", "rating", "review"])
imdb_df["source"] = "imdb"

selected_movie_info_df = pd.read_csv('selected_movie_info.csv')
merged_df = pd.merge(imdb_df, selected_movie_info_df, on='imdb_id')

# print(merged_df.columns)
# Index(['imdb_id', 'rating', 'review', 'movie', 'year', 'award','source'], dtype='object')

# Load AI-generated reviews from multiple sources
ai_files = ["reviews_ai/subtitles/aireviews_chatgpt.csv",
            "reviews_ai/subtitles/aireviews_deepseek.csv",
            "reviews_ai/subtitles/aireviews_gemini.csv"]
ai_dfs = {file: pd.read_csv(file) for file in ai_files}

ai_reviews_list = []

for file, df in ai_dfs.items():
    # Identify AI review columns dynamically (columns containing 'context')
    review_cols = df.filter(like="context").columns

    # Reshape:
    melted_df = df.melt(
        id_vars=["movie", "imdb_id"],  # Keep movie & imdb_id
        value_vars=review_cols,  # Select only review columns
        var_name="review_source",
        value_name="review"
    ).dropna()

    source = file.split("/")[-1].split("_")[1].split(".")[0] # This will extract 'chatgpt', 'deepseek', or 'gemini'

# Extract AI model name ('chatgpt', 'gemini', or 'deepseek')
    melted_df["source"] = source

    ai_reviews_list.append(melted_df)

# Combine all AI-generated reviews into a single dataframe
ai_reviews_long = pd.concat(ai_reviews_list, ignore_index=True)

# Randomly select one IMDb review
imdb_sample = imdb_df.sample(1)[["imdb_id", "review", "source"]]

# Randomly select one AI-generated review
ai_sample = ai_reviews_long.sample(1)[["imdb_id", "review", "source"]]

# Select 3 additional random reviews from both datasets
remaining_samples = pd.concat([imdb_df, ai_reviews_long]).sample(3)[["imdb_id", "review", "source"]]

# Combine all selected reviews and shuffle
random_selection = pd.concat([imdb_sample, ai_sample, remaining_samples]).sample(frac=1).reset_index(drop=True)

# Merge with AI & IMDb data to get the movie title
random_selection = random_selection.merge(ai_reviews_long[["imdb_id", "movie"]], on="imdb_id", how="left").drop_duplicates()

# Keep only movie, source, and review
random_selection = random_selection[["movie", "source", "review"]]

# Display selected reviews
print(random_selection)
# Save final_reviews to a CSV file
random_selection.to_csv("questionnaire_review.csv", index=False)

