import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

# --- Load similarity summary data ---
files = {
    "ChatGPT": "../cosine_similarity_and_other_tests/cosine_similarity_results_chatgpt.csv",
    "DeepSeek": "../cosine_similarity_and_other_tests/cosine_similarity_results_deepseek.csv",
    "Gemini": "../cosine_similarity_and_other_tests/cosine_similarity_results_gemini.csv",
    "Gemini (detailed)" : "../cosine_similarity_and_other_tests/cosine_similarity_results_gemini_detailed_context.csv",
    "IMDb": "../cosine_similarity_and_other_tests/within_imdb_similarity_results.csv"
}

dfs = {name: pd.read_csv(path) for name, path in files.items()}
dfs["IMDb"].rename(columns={
    "MeanWithinSimilarity": "MeanSimilarity",
    "MedianWithinSimilarity": "MedianSimilarity",
    "MaxWithinSimilarity": "MaxSimilarity"
}, inplace=True)

for name, df in dfs.items():
    df["Source"] = name

summary_df = pd.concat([df[["MeanSimilarity", "Source"]] for df in dfs.values()], ignore_index=True)

# --- Load per-movie screenplay similarity data ---
ai_df = pd.read_csv("../cosine_similarity_and_other_tests/cosine_similarity_results_screenplays_by_movie.csv")
imdb_df = pd.read_csv("../cosine_similarity_and_other_tests/within_imdb_similarity_results.csv")

ai_df["File"] = ai_df["File"].replace({
    "aireviews_chatgpt_screenplays.csv": "ChatGPT",
    "aireviews_deepseek_screenplays.csv": "DeepSeek",
    "aireviews_gemini_screenplays.csv": "Gemini",
    "aireviews_gemini_screenplays_context_variation.csv": "Gemini (detailed)"
})
ai_df.rename(columns={"File": "Source"}, inplace=True)

imdb_df["Source"] = "IMDb"
imdb_df.rename(columns={"MeanWithinSimilarity": "MeanSimilarity"}, inplace=True)

screenplay_df = pd.concat([
    ai_df[["MeanSimilarity", "Source"]],
    imdb_df[["MeanSimilarity", "Source"]]
], ignore_index=True)

# --- Create combined boxplots ---
sources = ["ChatGPT", "DeepSeek", "Gemini", "Gemini (detailed)", "IMDb"]
colors = sns.color_palette("Set2", n_colors=len(sources))
palette_dict = dict(zip(sources, colors))

fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
palette = sns.color_palette("Set2")

# Subplot 1: Overall summary (static cosine similarity comparison)
sns.boxplot(data=summary_df, x="Source", y="MeanSimilarity", ax=axes[0], hue="Source", palette=palette_dict, legend=False)
axes[0].set_title("Cosine Similarity Distribution by Subtitles")
axes[0].set_ylabel("Mean Cosine Similarity")
axes[0].set_xlabel("")

# Subplot 2: Movie-level similarity (screenplays)
sns.boxplot(data=screenplay_df, x="Source", y="MeanSimilarity", ax=axes[1], hue="Source", palette=palette_dict, legend=False)
axes[1].set_title("Cosine Similarity Distribution by Screenplays")
axes[1].set_ylabel("Mean Cosine Similarity")
axes[1].set_xlabel("Model / IMDb")


# Shared legend
handles = [mpatches.Patch(color=palette_dict[source], label=source) for source in sources]
fig.legend(handles=handles, title="Source", loc='upper center', bbox_to_anchor=(0.5, 0.975), ncol=5,
           title_fontsize=11, fontsize=10)

# Layout
plt.tight_layout()
plt.savefig("combined_cosine_similarity_boxplots.png", dpi=300, bbox_inches="tight")
plt.show()
