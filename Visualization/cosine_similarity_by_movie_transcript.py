import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# AI Model Performance for Individual Movies
# Grouped Bar Chart

merged_df = pd.read_csv('../Cosine Similarity/cosine_similarity_results_screenplays_by_movie.csv')

merged_df["File"] = merged_df["File"].replace({
    "aireviews_chatgpt_screenplays.csv": "ChatGPT",
    "aireviews_deepseek_screenplays.csv": "DeepSeek",
    "aireviews_gemini_screenplays.csv": "Gemini",
    "aireviews_gemini_screenplays_context_variation.csv": "Gemini (detailed)"
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
plt.ylim(top=0.48)  # add some headroom manually
# plt.title("Cosine Similarity Between IMDb and AI Reviews (By Movie)")
plt.xticks(rotation=45,ha='right')
# plt.legend(title="AI Model",fontsize=8)
plt.legend(title="AI Model", loc='upper left', bbox_to_anchor=(0.01, 0.99))

plt.tight_layout()
# save fig
plt.savefig("similarity_comparison_by movie(transript).png", dpi=300, bbox_inches="tight")
plt.show()

# Cosine Similarity Between IMDb and AI Reviews (By Movie) when providing movie transcript
# Observations

# 1. Overall similarity remains moderate (max ~0.42)
# Even the best models (DeepSeek) top out around 0.4 cosine similarity.
# Cosine similarity focuses on surface-level word overlap.
# Human reviews often include personal opinions, experience, cultural references, etc., things absent from screenplay-based generation.

# 2. Across all movies, DeepSeek has the highest cosine similarity with IMDb reviews.
# This may suggests that DeepSeek produced reviews are most lexically similar to human-written IMDb reviews.
# Possible reasons: DeepSeek may be more literal, closely following the choice of words, phrasing, or narrative details found in the screenplay
# This leads to higher word-level overlap, which cosine similarity rewards.

# 3.ChatGPT scores lower than all other models on every movie.
# This indicates that ChatGPT-generated reviews are less aligned with the lexical style and content of IMDb reviews.
# Possible reasons:

# 4. There is no significant difference between standard Gemini and the variation with diversified prompting/context.
# Gemini’s output is relatively stable regardless of prompting variation.
# Gemini may already produce a robust response based on the screenplay.

# 5."Crouching Tiger, Hidden Dragon" shows the most balanced scores across models.
# All models perform similarly on this film, with little variation between them.
#

# 5.Movies like Nomadland and Shawshank Redemption show lower similarity scores across all models,
# while Avatar and Titanic show higher similarity.


# Comparison Analysis: Subtitles vs. Transcripts
# compare 'Cosine similarity between IMDb reviews and AI reviews generated using subtitles' and
# 'Cosine similarity between IMDb reviews and AI reviews generated using transcript'

# Observations:
# 1. In most movies, cosine similarity scores are higher in the transcript-based plot than in the subtitle-based plot,
# especially for ChatGPT and Gemini.
# This suggests that access to full screenplay content improves AI's ability to generate reviews that resemble IMDb reviews.
# possible reason:
# Subtitles are limited to dialogue.
# Transcripts provide scene descriptions, emotional context, and structure, allowing the AI to write more nuanced reviews.

# 2. DeepSeek benefits less from transcripts than others
# DeepSeek’s performance is already high in both plots, with only small gains in the transcript version.
# Possible reason: DeepSeek is well-tuned, it has ability to extract a lot from dialogue alone,
# the extra descriptive context from transcripts doesn’t make a huge difference.

# 3. ChatGPT benefits significantly from transcripts
# ChatGPT scores are much lower in the subtitle-based plot, but improve considerably with transcripts.
# This suggests that ChatGPT is more dependent on narrative and structural information than just dialogue.

# 4.Both Gemini and Gemini (Context) perform better with transcripts, but the difference from subtitles is not dramatic.
# Gemini handles subtitle input reasonably well and captures core content, but transcripts provide a slight boost in detail and structure.

# 5. Across both plots, DeepSeek consistently ranks highest, followed by Gemini models, then ChatGPT.
