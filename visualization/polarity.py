import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
ai_models = pd.read_csv('../polarity_scores_output/average_polarity_scores_subtitles.csv')
imdb_above7 = pd.read_csv('../average_polarity_scores_imdb_above_7.csv')
imdb_below6 = pd.read_csv('../average_polarity_scores_imdb_below_6.csv')
imdb_6_and_7 = pd.read_csv('../average_polarity_scores_imdb_between_6_and_7.csv')

ai_models['File'] = ai_models['File'].replace({
    'aireviews_chatgpt.csv': 'chatgpt',
    'aireviews_deepseek.csv': 'deepseek',
    'aireviews_gemini.csv': 'gemini',
    'aireviews_gemini_context_variation.csv' : 'gemini (detailed context)'
})

# Rename the 'File' column to 'Source'
ai_models = ai_models.rename(columns={'File': 'Model'})

# Filter out AI models for Question 2 (good reviews)
ai_good_reviews = ai_models[ai_models['Question'] == 'question2'].copy()
ai_good_reviews['Source'] = ai_good_reviews['Model']  # Label the AI source as per the model

ai_bad_reviews = ai_models[ai_models['Question'] == 'question1'].copy()
ai_neutral_reviews = ai_models[ai_models['Question'] == 'question3'].copy()

ai_bad_reviews['Source'] = ai_bad_reviews['Model']  # Label the AI source for Q1 (bad reviews)
ai_neutral_reviews['Source'] = ai_neutral_reviews['Model']  # Label the AI source for Q3 (neutral reviews)

# Add the IMDb data to the plot with the 'Source' as 'IMDb'
imdb_above7['Source'] = 'IMDb'
imdb_below6['Source'] = 'IMDb Below 6'
imdb_6_and_7['Source'] = 'IMDb 6 and 7'

# Select the relevant columns for polarity scores
ai_good_reviews = ai_good_reviews[['Movie', 'Source', 'Average Negative', 'Average Neutral', 'Average Positive']]
imdb_above7 = imdb_above7[['Movie', 'Source', 'Average Negative', 'Average Neutral', 'Average Positive']]
ai_bad_reviews = ai_bad_reviews[['Movie', 'Source', 'Average Negative', 'Average Neutral', 'Average Positive']]
ai_neutral_reviews = ai_neutral_reviews[['Movie', 'Source', 'Average Negative', 'Average Neutral', 'Average Positive']]
imdb_below6 = imdb_below6[['Movie', 'Source', 'Average Negative', 'Average Neutral', 'Average Positive']]
imdb_6_and_7 = imdb_6_and_7[['Movie', 'Source', 'Average Negative', 'Average Neutral', 'Average Positive']]


# Combine IMDb and AI data into one dataset
combined_data = pd.concat([ai_good_reviews, imdb_above7], ignore_index=True)
combined_bad_reviews = pd.concat([ai_bad_reviews, imdb_below6], ignore_index=True)
combined_neutral_reviews = pd.concat([ai_neutral_reviews, imdb_6_and_7], ignore_index=True)


# Melt the data for good reviews
melted_data = pd.melt(combined_data, id_vars=['Movie', 'Source'],
                      value_vars=['Average Negative', 'Average Neutral', 'Average Positive'],
                      var_name='Polarity Type', value_name='Polarity Score')

# Melt the data for bad reviews (Q1 vs IMDb Below 6)
melted_bad_reviews = pd.melt(combined_bad_reviews, id_vars=['Movie', 'Source'],
                             value_vars=['Average Negative', 'Average Neutral', 'Average Positive'],
                             var_name='Polarity Type', value_name='Polarity Score')

# Melt the data for neutral reviews (Q3 vs IMDb Between 6 and 7)
melted_neutral_reviews = pd.melt(combined_neutral_reviews, id_vars=['Movie', 'Source'],
                                 value_vars=['Average Negative', 'Average Neutral', 'Average Positive'],
                                 var_name='Polarity Type', value_name='Polarity Score')

### Box Plot

# Create a boxplot to compare IMDb and AI models (Question 2)
plt.figure(figsize=(12, 6))
sns.boxplot(x='Source', y='Polarity Score', hue='Polarity Type', data=melted_data, palette='Set2')

# Add titles and labels
plt.title("Comparison of Polarity Scores for IMDb (Rating > 7) vs AI (Question 2)", fontsize=14)
plt.xlabel("Source (IMDb vs AI)", fontsize=12)
plt.ylabel("Polarity Score", fontsize=12)
plt.xticks(rotation=45)

# Display the plot
plt.tight_layout()
plt.show()

# Create a boxplot for bad reviews (Q1 vs IMDb Below 6)
plt.figure(figsize=(12, 6))
sns.boxplot(x='Source', y='Polarity Score', hue='Polarity Type', data=melted_bad_reviews, palette='Set2')

# Add titles and labels for bad reviews plot
plt.title("Comparison of Polarity Scores for AI (Question 1) vs IMDb (Rating < 6)", fontsize=14)
plt.xlabel("Source (IMDb Below 6 vs AI - Q1)", fontsize=12)
plt.ylabel("Polarity Score", fontsize=12)
plt.xticks(rotation=45)

# Display the plot
plt.tight_layout()
plt.show()

# Create a boxplot for neutral reviews (Q3 vs IMDb Between 6 and 7)
plt.figure(figsize=(12, 6))
sns.boxplot(x='Source', y='Polarity Score', hue='Polarity Type', data=melted_neutral_reviews, palette='Set2')

# Add titles and labels for neutral reviews plot
plt.title("Comparison of Polarity Scores for AI (Question 3) vs IMDb (Rating 6-7)", fontsize=14)
plt.xlabel("Source (IMDb 6-7 vs AI - Q3)", fontsize=12)
plt.ylabel("Polarity Score", fontsize=12)
plt.xticks(rotation=45)

# Display the plot
plt.tight_layout()
plt.show()

### Bar Plot
# Create a bar plot for the melted data (Good Reviews)
plt.figure(figsize=(12, 6))
sns.barplot(x='Source', y='Polarity Score', hue='Polarity Type', data=melted_data, palette='Set2')

# Add titles and labels
plt.title("Comparison of Polarity Scores for IMDb vs AI (Good Reviews)", fontsize=14)
plt.xlabel("Source (IMDb vs AI)", fontsize=12)
plt.ylabel("Polarity Score", fontsize=12)
plt.xticks(rotation=45)

# Display the plot
plt.tight_layout()
plt.show()
# save fig
plt.savefig("polarity_score_comparison(good).png", dpi=300, bbox_inches="tight")

# Create a bar plot for the melted data (Bad Reviews)
plt.figure(figsize=(12, 6))
sns.barplot(x='Source', y='Polarity Score', hue='Polarity Type', data=melted_data, palette='Set2')

# Add titles and labels
# Comparison of Polarity Scores for AI (Question 1) vs IMDb (Rating < 6)
plt.title("Comparison of Polarity Scores for AI vs IMDb (Bad Reviews)", fontsize=14)
plt.xlabel("Source (IMDb Below 6 vs AI - Q1)", fontsize=12)
plt.ylabel("Polarity Score", fontsize=12)
plt.xticks(rotation=45)

# Display the plot
plt.tight_layout()
plt.show()
# save fig
plt.savefig("polarity_score_comparison(bad).png", dpi=300, bbox_inches="tight")

# Create a bar plot for the melted data (Neutral Reviews)
plt.figure(figsize=(12, 6))
sns.barplot(x='Source', y='Polarity Score', hue='Polarity Type', data=melted_data, palette='Set2')

# Add titles and labels
# Comparison of Polarity Scores for AI (Question 3) vs IMDb (Rating 6-7)
plt.title("Comparison of Polarity Scores for AI vs IMDb (Neutral Reviews)", fontsize=14)
plt.xlabel("Source (IMDb 6-7 vs AI - Q3)", fontsize=12)
plt.ylabel("Polarity Score", fontsize=12)
plt.xticks(rotation=45)

# Display the plot
plt.tight_layout()
plt.show()
# save fig
plt.savefig("polarity_score_comparison(neutral).png", dpi=300, bbox_inches="tight")
