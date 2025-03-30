import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# AI Model Performance for Individual Movies
# Grouped Bar Chart

merged_df = pd.read_csv('../cosine_similarity_and_other_tests/cosine_similarity_results_subtitles_by_movie.csv')

merged_df["File"] = merged_df["File"].replace({
    "aireviews_chatgpt.csv": "ChatGPT",
    "aireviews_deepseek.csv": "DeepSeek",
    "aireviews_gemini.csv": "Gemini",
    "aireviews_gemini_context_variation.csv": "Gemini (Context)"
})

merged_df.rename(columns={'File': 'AI Model'}, inplace=True)

movie_info_df = pd.read_csv('../selected_movie_info.csv')

# Merge the AI model dataframe with the movie info dataframe on 'MovieID' and 'imdb_id'
merged_df = merged_df.merge(movie_info_df[['imdb_id', 'movie']], left_on='MovieID', right_on='imdb_id', how='left')

# Drop the 'imdb_id' column and rename 'movie' to 'MovieName'
merged_df.drop(columns=['imdb_id','MovieID'], inplace=True)

# Display the result
# print(merged_df.head())

# Plot
plt.figure(figsize=(12, 8))
sns.barplot(data=merged_df, x="movie", y="MeanSimilarity", hue="AI Model",  palette="Set2")

# Labels and title
plt.xlabel("Movie")
plt.ylabel("Mean Cosine Similarity")
plt.title("Cosine Similarity Between IMDb and AI Reviews (By Movie)")
plt.xticks(rotation=45,ha='right')
plt.legend(title="AI Model",fontsize=8)

plt.tight_layout()
plt.show()
# save fig
plt.savefig("mean_similarity_comparison(by movie).png", dpi=300, bbox_inches="tight")



