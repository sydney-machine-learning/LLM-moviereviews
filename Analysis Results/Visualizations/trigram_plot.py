import pandas as pd

# Load data
imdb_df = pd.read_csv("../Trigram Analysis/all_imdb_review_trigrams.csv")
chatgpt_df = pd.read_csv("../Trigram Analysis/aireviews_chatgpt_trigrams.csv")
deepseek_df = pd.read_csv("../Trigram Analysis/aireviews_deepseek_trigrams.csv")
gemini_df = pd.read_csv("../Trigram Analysis/aireviews_gemini_trigrams.csv")
gemini_ctx_df = pd.read_csv("../Trigram Analysis/aireviews_gemini_context_variation_trigrams.csv")

def top_trigram(df, model_name):
    q2 = df[df["Question"] == "question2"]
    q2_sorted = q2.sort_values(["Movie", "Count"], ascending=[True, False])
    top4 = q2_sorted.groupby("Movie").head(4).reset_index(drop=True)
    top4 = top4.rename(columns={"Movie": "Title"})[["Title", "Trigram", "Count"]]
    top4["Model"] = model_name
    return top4

# Prepare each source
imdb_top = imdb_df.sort_values("Count", ascending=False).drop_duplicates(subset="Title")
imdb_top = imdb_top[["Title", "Trigram", "Count"]].rename(columns={"Trigram": "IMDb Trigram", "Count": "IMDb Count"})

chatgpt_top = top_trigram(chatgpt_df, "ChatGPT-4o")
deepseek_top = top_trigram(deepseek_df, "DeepSeek-V3")
gemini_top = top_trigram(gemini_df, "Gemini-2")
gemini_ctx_top = top_trigram(gemini_ctx_df, "Gemini(detailed")

print(chatgpt_top)
