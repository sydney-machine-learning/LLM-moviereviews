import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load subtitle-based results
subs_df = pd.read_csv('../cosine_similarity_and_other_tests/cosine_similarity_results_subtitles_by_movie.csv')
subs_df["File"] = subs_df["File"].replace({
    "aireviews_chatgpt.csv": "ChatGPT",
    "aireviews_deepseek.csv": "DeepSeek",
    "aireviews_gemini.csv": "Gemini",
    "aireviews_gemini_context_variation.csv": "Gemini (detailed)"
})
subs_df.rename(columns={'File': 'AI Model'}, inplace=True)
movie_info_df = pd.read_csv('../selected_movie_info.csv')
subs_df = subs_df.merge(movie_info_df[['imdb_id', 'movie']], left_on='MovieID', right_on='imdb_id', how='left')
subs_df.drop(columns=['imdb_id', 'MovieID'], inplace=True)

# Load screenplay-based results
script_df = pd.read_csv('../cosine_similarity_and_other_tests/cosine_similarity_results_screenplays_by_movie.csv')
script_df["File"] = script_df["File"].replace({
    "aireviews_chatgpt_screenplays.csv": "ChatGPT",
    "aireviews_deepseek_screenplays.csv": "DeepSeek",
    "aireviews_gemini_screenplays.csv": "Gemini",
    "aireviews_gemini_screenplays_context_variation.csv": "Gemini (detailed)"
})
script_df.rename(columns={'File': 'AI Model'}, inplace=True)
script_df = script_df.merge(movie_info_df[['imdb_id', 'movie']], left_on='MovieID', right_on='imdb_id', how='left')
script_df.drop(columns=['imdb_id', 'MovieID'], inplace=True)

# Plotting both as subplots
fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
sns.set(style="whitegrid")

# Subtitles
sns.barplot(data=subs_df, x="movie", y="MeanSimilarity", hue="AI Model", ax=axes[0], palette="Set2")
axes[0].set_title("Cosine Similarity by Movie (Subtitles)")
axes[0].set_ylabel("Mean Cosine Similarity")
axes[0].set_xlabel("")  # remove x-label
# axes[0].tick_params(axis='x', rotation=45)
# axes[0].legend(title="AI Model", loc='upper left', fontsize=9)
axes[0].get_legend().remove()

# Screenplays
sns.barplot(data=script_df, x="movie", y="MeanSimilarity", hue="AI Model", ax=axes[1], palette="Set2")
axes[1].set_title("Cosine Similarity by Movie (Screenplays)")
axes[1].set_ylabel("Mean Cosine Similarity")
axes[1].set_xlabel("Movie")
# axes[1].tick_params(axis='x', rotation=45)
# axes[1].legend(title="AI Model", loc='upper left', fontsize=9)
axes[1].get_legend().remove()

# add shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='upper center',
    ncol=4,
    title="AI Model",
    title_fontsize=11,
    fontsize=10,
    bbox_to_anchor=(0.5, 0.97)
)
# Adjust layout
plt.tight_layout()
plt.savefig("similarity_comparison_by_movie_combined.png", dpi=300, bbox_inches="tight")
plt.show()
