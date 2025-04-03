import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# We have cosine similarity results for AI-generated reviews and human reviews,
# our goal is to visualise and analysis these results to discuss the accuracy of AI models,
# and the ability of AI model to minic human thinking

# Load the cosine similarity data
files = {
    "ChatGPT": "../cosine_similarity_and_other_tests/cosine_similarity_results_chatgpt.csv",
    "Gemini": "../cosine_similarity_and_other_tests/cosine_similarity_results_gemini.csv",
    "Gemini (detailed)" : "../cosine_similarity_and_other_tests/cosine_similarity_results_gemini_detailed_context.csv",
    "DeepSeek": "../cosine_similarity_and_other_tests/cosine_similarity_results_deepseek.csv",
    "IMDB": "../cosine_similarity_and_other_tests/within_imdb_similarity_results.csv",
}

# Read all files into a dictionary of DataFrames
dfs = {name: pd.read_csv(file) for name, file in files.items()}

dfs["IMDB"].rename(columns={
    "MeanWithinSimilarity": "MeanSimilarity",
    "MedianWithinSimilarity": "MedianSimilarity",
    "MaxWithinSimilarity": "MaxSimilarity"
}, inplace=True)

# check
# print(dfs["IMDB"].head())

# Add a "Source" column to each DataFrame for comparison
for name, df in dfs.items():
    df["Source"] = name

# Combine all DataFrames into a single DataFrame
df_all = pd.concat(dfs.values(), ignore_index=True)

# Display first few rows
# print(df_all.head())

# We want to analyze
# 1. Overall similarity distribution: compare Mean, Median and Max Similarity for AI models vs. IMDb
# 2. Boxplot: Show variance and distribution of cosine similarity scores across models.
# 3. Heatmap: Show correlation between different models.

plt.figure(figsize=(12, 8))
sns.boxplot(data=df_all, x="Source", y="MeanSimilarity", palette="Set2")
# plt.title("Comparison of Mean Similarity Across AI Models and IMDB")
plt.ylabel("Mean Cosine Similarity")
plt.xlabel("\n\nAI Models")
#plt.xticks(rotation=10)
# save fig
plt.savefig("mean_similarity_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# Cosine Similarity is a robust and effective measure for assessing text similarity. (https://www.irjmets.com/uploadedfiles/paper//issue_4_april_2024/52264/final/fin_irjmets1712589489.pdf)
# When using cosine similarity to compare reviews, we are looking at how similar the reviews are in terms of their
# texture representation.  Lexical similarity (word choice and vocabulary): frequency of words or phrases they contain

# Cosine similarity however still can’t handle semantic meaning of the text perfectly. (https://www.researchgate.net/profile/Faisal-Rahutomo/publication/262525676_Semantic_Cosine_Similarity/links/0a85e537ee3b675c1e000000/Semantic-Cosine-Similarity.pdf)

# Interpretation of the mean similarity box plot:
# This boxplot compares the mean cosine similarity across AI-generated reviews (ChatGPT, Gemini, DeepSeek)
# and IMDB human reviews

# Observations:
# 1. AI-generated reviews are more similar to IMDb reviews than IMDb reviews to each other
# This indicates that AI-generated reviews are closer in common words to human reviews than humans to each other.
# This may because AI models can replicate human language patterns well, like word choice,
# but AI models does not fully capture human thought diversity becasue human reviewrs have unique opinions.

# Also, AI reviews are still not identical to human reviews, as the similarity values are not close to 1.

# 2. DeepSeek Reviews Are Most Similar to IMDB Reviews (highest mean cosine similarity)
# DeepSeek has the highest mean similarity to IMDB, followed by Gemini, and then ChatGPT.
# This suggests that DeepSeek’s writing style is the most aligned with human reviews, possibly due to more natural phrasing or diverse sentence structures.

# 3. Human Reviews Are the Most Diverse (Lowest Mean Similarity Within IMDB)
# The IMDB self-similarity is the lowest, meaning human-written reviews vary the most.
# This may because human reviews come from different reviewers with unique writing styles and opinions.

# Now that we have both polarity scores and cosine similarity, we can analyze their relationship.

