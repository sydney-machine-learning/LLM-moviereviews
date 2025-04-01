import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# load IMDb file
df_imdb = pd.read_csv('../download/all_imdb_reviews.csv',
                      names = ['imdb_id','rating','review'])

# print(df_imdb.head())

# filter by IMDb ratings
# add sentiment column

def filter_rating(rating):
    if rating > 7:
        return 'positive'
    elif rating < 6:
        return 'negative'
    else: # 6 or 7
        return 'neutral'

df_imdb['sentiment'] = df_imdb['rating'].apply(filter_rating)
# print(df_imdb.columns) -> Index(['imdb_id', 'rating', 'review', 'sentiment'], dtype='object')
# mapping between imdb_id and movie
movie_info = pd.read_csv('../selected_movie_info.csv')[['imdb_id', 'movie']]
# merge with movie_info to bring in the movie title
df_imdb = df_imdb.merge(movie_info, on='imdb_id', how='left')
# add source=imdb
df_imdb['source'] = 'IMDb'
# print(df_imdb.columns) #['imdb_id', 'rating', 'review', 'sentiment', 'movie', 'source']

# Group IMDb reviews by sentiment
# df_imdb_grouped = df_imdb.groupby(["sentiment"])["review"].apply(lambda x: " ".join(map(str, x))).reset_index()
 # print(df_imdb_grouped.head())

# flatten AI reviews
# load ChatGPT ai reviews (generated based on subtitles)

def flatten_ai_reviews(filepath, ai_model_name):
    df_ai = pd.read_csv(filepath)
    df_ai['source'] = ai_model_name

    # extract the context-question columns
    # value_vars = [col for col in df_ai.columns if col.startswith("chatgpt_context")]
    value_vars = [col for col in df_ai.columns if "context" in col]

    # melt the dataframe
    df_long = df_ai.melt(
        id_vars = ['movie','imdb_id','source'],
        value_vars = value_vars,
        var_name = 'context_question',
        value_name = 'review'
    )

    # Extract context number and question from column name
    # df_long["context"] = df_long["context_question"].str.extract(r'context(\d+)_question\d+').astype(int)
    df_long["question"] = df_long["context_question"].str.extract(r'(question\d+)')

    # Drop the old column name
    df_long = df_long.drop(columns=["context_question"])
    # df_long = df_long.drop(['context'], axis=1)
    # print(df_long.head())
    # print(df_long.columns) # ['movie', 'imdb_id', 'source', 'review', 'question']
    return df_long

df_chatgpt = flatten_ai_reviews('../reviews_ai/subtitles/aireviews_chatgpt.csv', 'ChatGPT')
df_deepseek = flatten_ai_reviews('../reviews_ai/subtitles/aireviews_deepseek.csv', 'DeepSeek')
df_gemini = flatten_ai_reviews('../reviews_ai/subtitles/aireviews_gemini.csv', 'Gemini')
df_gemini_ctx = flatten_ai_reviews('../reviews_ai/subtitles/aireviews_gemini_context_variation.csv', 'Gemini(detailed)')


df_all = pd.concat([df_chatgpt, df_deepseek, df_gemini, df_gemini_ctx], ignore_index=True)

# df_all.groupby("question")["review"].apply(lambda x: " ".join(x.tolist())[:500])

# df_ai_grouped = df_all.groupby(["source", "question"])["review"].apply(lambda x: " ".join(map(str, x))).reset_index()
# print(df_ai_grouped.head())
# print(df_ai_grouped.columns)  ['source', 'question', 'review']

# --- Compute cosine similarity ---
results = []
for sentiment in ['positive', 'neutral', 'negative']:
    imdb_reviews = df_imdb[df_imdb['sentiment'] == sentiment]['review'].dropna().tolist()

    vectorizer = TfidfVectorizer()
    imdb_vectors = vectorizer.fit_transform(imdb_reviews)

    for source in df_all['source'].unique():
        for question in ['question1', 'question2', 'question3']:
            ai_reviews = df_all[(df_all['source'] == source) & (df_all['question'] == question)]['review'].dropna().tolist()
            if not ai_reviews:
                continue
            ai_vectors = vectorizer.transform(ai_reviews)

            sim_matrix = cosine_similarity(imdb_vectors, ai_vectors)
            mean_sim = sim_matrix.mean()

            results.append({
                'source': source,
                'sentiment': sentiment,
                'question': question,
                'mean_cosine_similarity': round(mean_sim, 4)
            })

# Output results
df_result = pd.DataFrame(results)
# print(df_result)

# -----------Heat Map -----------
# create a source_question column
df_result["source_question"] = df_result["source"] + "_" + df_result["question"]

# Pivot for heatmap
heatmap_df = df_result.pivot(index="sentiment", columns="source_question", values="mean_cosine_similarity")
# Reorder the index (rows)
heatmap_df = heatmap_df.reindex(["negative", "positive", "neutral"])

# Plot heatmap
# Corrected version using set_xticklabels for the secondary axis
plt.figure(figsize=(14, 6))
sns.heatmap(heatmap_df, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=0.5)

# Primary x-axis: question labels
question_labels = ["question1", "question2", "question3"] * 4
# plt.xticks(ticks=range(len(question_labels)), labels=question_labels, rotation=0)
plt.xticks(ticks=[i + 0.5 for i in range(len(question_labels))], labels=question_labels, rotation=0)

# Add a secondary x-axis below for model labels
ax = plt.gca()
ax2 = ax.secondary_xaxis('bottom')
ax2.set_ticks([1.5, 4.5, 7.5, 10.5])
ax2.set_xticklabels(["ChatGPT", "DeepSeek", "Gemini(detailed)", "Gemini"])
ax2.tick_params(axis='x', labelsize=10, pad=25)

plt.title("Cosine Similarity Heatmap: IMDb Sentiment vs AI Model by Questions")
plt.ylabel("IMDb Sentiment")
plt.xlabel("\n\nAI Models and Questions")
plt.tight_layout()
plt.savefig("similarity_comparison_by questions(subtitles).png", dpi=300, bbox_inches="tight")
plt.show()

