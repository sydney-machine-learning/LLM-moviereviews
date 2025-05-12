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

# subtitles
df_chatgpt = flatten_ai_reviews('../AI Generated Reviews/subtitles/aireviews_chatgpt.csv', 'ChatGPT')
df_deepseek = flatten_ai_reviews('../AI Generated Reviews/subtitles/aireviews_deepseek.csv', 'DeepSeek')
df_gemini = flatten_ai_reviews('../AI Generated Reviews/subtitles/aireviews_gemini.csv', 'Gemini')
df_gemini_ctx = flatten_ai_reviews('../AI Generated Reviews/subtitles/aireviews_gemini_context_variation.csv', 'Gemini(detailed)')

# screenplays
df_chatgpt_1 = flatten_ai_reviews('../AI Generated Reviews/screenplays/aireviews_chatgpt_screenplays.csv', 'ChatGPT')
df_deepseek_1 = flatten_ai_reviews('../AI Generated Reviews/screenplays/aireviews_deepseek_screenplays.csv', 'DeepSeek')
df_gemini_1 = flatten_ai_reviews('../AI Generated Reviews/screenplays/aireviews_gemini_screenplays.csv', 'Gemini')
df_gemini_ctx_1 = flatten_ai_reviews(
    '../AI Generated Reviews/screenplays/aireviews_gemini_screenplays_context_variation.csv', 'Gemini(detailed)')


df_all = pd.concat([df_chatgpt, df_deepseek, df_gemini, df_gemini_ctx], ignore_index=True)
df_all_1 = pd.concat([df_chatgpt_1, df_deepseek_1, df_gemini_1, df_gemini_ctx_1], ignore_index=True)


# --- Compute cosine similarity ---
def compute_similarity(df_ai, df_imdb):
    results = []    # subtitles

    for sentiment in ['positive', 'neutral', 'negative']:
        imdb_reviews = df_imdb[df_imdb['sentiment'] == sentiment]['review'].dropna().tolist()
        vectorizer = TfidfVectorizer()
        imdb_vectors = vectorizer.fit_transform(imdb_reviews)

        for source in df_ai['source'].unique():
            for question in ['question1', 'question2', 'question3']:
                ai_reviews = df_ai[(df_ai['source'] == source) & (df_ai['question'] == question)]['review'].dropna().tolist()
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
    return pd.DataFrame(results)


# -----------Heat Map -----------
def plot_similarity_heatmap(df_result, title, filename):
    # create a source_question column
    df_result["source_question"] = df_result["source"] + "_" + df_result["question"]
    # Pivot for heatmap
    heatmap_df = df_result.pivot(index="sentiment", columns="source_question", values="mean_cosine_similarity")
    # Reorder the index (rows)
    # heatmap_df.index = heatmap_df.index.str.capitalize()
    heatmap_df = heatmap_df.reindex(["negative", "positive", "neutral"])
    heatmap_df.index = heatmap_df.index.str.capitalize()

    # Plot heatmap
    # Corrected version using set_xticklabels for the secondary axis
    plt.figure(figsize=(14, 6))
    sns.heatmap(heatmap_df, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=0.5,
                annot_kws={"size": 18},
                cbar = False)

    # primary x-axis: question labels
    question_labels = ['Q1', 'Q2', 'Q3'] * 4
    plt.xticks(ticks=[i + 0.5 for i in range(len(question_labels))], labels=question_labels, rotation=0,
               fontsize = 16)

    plt.yticks(fontsize = 18)

    # Add a secondary x-axis below for model labels
    ax = plt.gca()

    ax2 = ax.secondary_xaxis('bottom')
    ax2.set_ticks([1.5, 4.5, 7.5, 10.5])
    ax2.set_xticklabels(["ChatGPT-4o", "DeepSeek-V3", "Gemini(detailed)", "Gemini-2"])
    ax2.tick_params(axis='x', labelsize=18, pad=25)

    # plt.title("Cosine Similarity Heatmap: IMDb Sentiment vs AI Model by Questions")
    plt.ylabel("IMDb Sentiment", fontsize = 18, labelpad=15)
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()

df_result_subtitles = compute_similarity(df_all, df_imdb)
df_result_screenplays = compute_similarity(df_all_1, df_imdb)

plot_similarity_heatmap(df_result_subtitles, "Cosine Similarity Heatmap (Subtitles)",
                        "similarity_comparison_by_questions_subtitles.png")
plot_similarity_heatmap(df_result_screenplays, "Cosine Similarity Heatmap (Screenplays)",
                        "similarity_comparison_by_questions_screenplays.png")


