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

# Melt the data for neutral reviews (Q3 vs IMDb Between 6 and 7)
melted_good_reviews = pd.melt(combined_data, id_vars=['Movie', 'Source'],
                                 value_vars=['Average Negative', 'Average Neutral', 'Average Positive'],
                                 var_name='Polarity Type', value_name='Polarity Score')

### Bar Plot
# Create a bar plot for the melted data (Good Reviews)
# plt.figure(figsize=(12, 6))
# sns.barplot(x='Source', y='Polarity Score', hue='Polarity Type', data=melted_data, palette='Set2')
#
# # Add titles and labels
# plt.title("Comparison of Polarity Scores for IMDb vs AI (Good Reviews)", fontsize=14)
# plt.xlabel("Source (IMDb vs AI)", fontsize=12)
# plt.ylabel("Polarity Score", fontsize=12)
# plt.xticks(rotation=45)
#
# # Display the plot
# plt.tight_layout()
# # save fig
# plt.savefig("polarity_score_comparison(good).png", dpi=300, bbox_inches="tight")
# plt.show()
#
# # Create a bar plot for the melted data (Bad Reviews)
# plt.figure(figsize=(12, 6))
# sns.barplot(x='Source', y='Polarity Score', hue='Polarity Type', data=melted_data, palette='Set2')
#
# # Add titles and labels
# # Comparison of Polarity Scores for AI (Question 1) vs IMDb (Rating < 6)
# plt.title("Comparison of Polarity Scores for AI vs IMDb (Bad Reviews)", fontsize=14)
# plt.xlabel("Source (IMDb Below 6 vs AI - Q1)", fontsize=12)
# plt.ylabel("Polarity Score", fontsize=12)
# plt.xticks(rotation=45)
#
# # Display the plot
# plt.tight_layout()
# # save fig
# plt.savefig("polarity_score_comparison(bad).png", dpi=300, bbox_inches="tight")
# plt.show()
#
# # Create a bar plot for the melted data (Neutral Reviews)
# plt.figure(figsize=(12, 6))
# sns.barplot(x='Source', y='Polarity Score', hue='Polarity Type', data=melted_data, palette='Set2')
#
# # Add titles and labels
# # Comparison of Polarity Scores for AI (Question 3) vs IMDb (Rating 6-7)
# plt.title("Comparison of Polarity Scores for AI vs IMDb (Neutral Reviews)", fontsize=14)
# plt.xlabel("Source (IMDb 6-7 vs AI - Q3)", fontsize=12)
# plt.ylabel("Polarity Score", fontsize=12)
# plt.xticks(rotation=45)
#
# # Display the plot
# plt.tight_layout()
# # save fig
# plt.savefig("polarity_score_comparison(neutral).png", dpi=300, bbox_inches="tight")
# plt.show()

fig, axes = plt.subplots(3, 1, figsize=(12,9.5), sharey=True)
sns.set(style="whitegrid")
palette = sns.color_palette("Set2", 3)

# Plot bad reviews
sns.boxplot(x='Source', y='Polarity Score', hue='Polarity Type',
            data=melted_bad_reviews, ax=axes[0], palette=palette,
            showfliers=False) # no outliers
axes[0].set_title("IMDb < 6 vs AI (Q1 - Bad Reviews)")
# axes[0].set_xlabel("Source")
axes[0].set_xlabel("")
axes[0].set_ylabel("Polarity Score")
axes[0].tick_params(axis='x')
axes[0].get_legend().remove()

# Plot good reviews
sns.boxplot(x='Source', y='Polarity Score', hue='Polarity Type',
            data=melted_good_reviews, ax=axes[1], palette=palette,
            showfliers=False)
axes[1].set_title("IMDb > 7 vs AI (Q2 - Good Reviews)")
# axes[1].set_xlabel("Source")
axes[1].set_xlabel("")
axes[1].tick_params(axis='x')
axes[1].get_legend().remove()

# Plot neutral reviews
sns.boxplot(x='Source', y='Polarity Score', hue='Polarity Type',
            data=melted_neutral_reviews, ax=axes[2], palette=palette,
            showfliers=False)
axes[2].set_title("IMDb 6–7 vs AI (Q3 - Neutral Reviews)")
# axes[2].set_xlabel("Source")
axes[2].set_xlabel("")
axes[2].tick_params(axis='x')
axes[2].get_legend().remove()

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left', ncol=1,
           fontsize=10, title_fontsize=11)

# Add bottom label for the whole figure
# fig.text(0.5, 0.04, 'Source', ha='center', fontsize=12)

# layout adjustment
fig.subplots_adjust(hspace=1)
plt.tight_layout(rect=(0, 0, 1, 0.96))

# plt.savefig("combined_polarity_score_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# neglect Box Plot for now


# Obeservation of Bar Plot
# The Bar Plot presents the polarity score distribution across different AI models and IMDb for good reviews (i.e. Question 2).

# Observations:
# When the prompt question is 'generate a good reciew':
# 1. AI Models Show Distinctive Sentiment Distributions:
# 1.1 ChatGPT has the highest average positive score, close to 1, with almost no neutral or negative scores
# 1.2 DeepSeek has relatively high average score, slightly lower than ChatGPT's, which is around 0.7,
# but also has a noticeable neutral component and a small negative score.
# 1.3 Gemini's barplot is more balanced than ChatGPT and DeepSeek,
# it has with significant neutral and positive sentiment, but also some negative sentiment.
# 1.4 Gemini (detailed prompt): The prompt question is 'generate a good reciew', while when we provided more detailed context,
# the average negative score is slightly higher than both average neutral and average positive,
# though the distribution of polarity score is more balance than others.
# It is noticable that average positive score is the lowest one in this barplot even when the question is 'generate a good reciew'

# 2. IMDb Reviews Are More Balanced
# IMDb reviews show a mix of positive, neutral, and negative sentiment, but positive sentiment still dominate

# 3. AI Models vs. IMDb: Sentiment Bias
# 3.1 DeepSeek and IMDb have the most similar sentiment distributions in terms of negative, neutral, and positive polarity scores
# 3.2 In all distributions except Gemini (detailed prompt), the polarity scores follow this general pattern:
# Positive > Neutral > Negative.
# However, Gemini (detailed prompt) is the only model where this order is reversed: Negative > Neutral > Positive


# Analysis:
# 1. ChatGPT might be over-optimistic, possibly due to training biases in favor of more positive language.
# 2. DeepSeek may generate more realistic or nuanced responses, closer to human-written IMDb reviews
# 3. Gemini is more balanced, which differs from other AI models and IMDb, indicating it might be designed to mimic human sentiment more closely.
# 4. Gemini (detailed prompt) stands out, this suggests that adding more context leads to a more critical or neutral stance,
# possibly because it considers a broader range of perspectives or prompt itself could have sentiment bias.


# polarity scores by ai_models (x-axis represents 3 scores, legend represents ai models)
def generate_polarity_box_plots(ai_df, source_name='ai', ax=None):
    ai_df = ai_df.copy()
    ai_df.rename(columns={'File': 'AI Model'}, inplace=True)

    # change column names
    if source_name == 'Subtitle':
        ai_df["AI Model"] = ai_df["AI Model"].replace({
            "aireviews_chatgpt.csv": "ChatGPT",
            "aireviews_deepseek.csv": "DeepSeek",
            "aireviews_gemini.csv": "Gemini",
            "aireviews_gemini_context_variation.csv": "Gemini_Context"
        })
    elif source_name == 'Screenplay':
        ai_df["AI Model"] = ai_df["AI Model"].replace({
            "aireviews_chatgpt_screenplays.csv": "ChatGPT",
            "aireviews_deepseek_screenplays.csv": "DeepSeek",
            "aireviews_gemini_screenplays.csv": "Gemini",
            "aireviews_gemini_screenplays_context_variation.csv": "Gemini_Context"
        })
    # replace NA with 0
    ai_df = ai_df.fillna(0)

    # change wide-format to long-format
    ai_melted = ai_df.melt(
        id_vars=['AI Model', 'Movie'],
        value_vars=['Average Negative', 'Average Neutral', 'Average Positive'],
        var_name='Polarity',
        value_name='Score'
    )

    # display boxplot
    sns.boxplot(data=ai_melted, x='Polarity', y='Score', hue='AI Model', palette="Set2", showfliers=False, ax=ax)

    # ddding title and labels
    ax.set_title(f'{source_name}', fontsize=16)
    ax.set_xlabel('Polarity', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)

    ax.grid(False)

    ax.get_legend().remove()
    if ax == axes[0]:
        ax.set_xlabel('')

# load the subtitle and screenplay data
subtitles_df = pd.read_csv('../polarity_scores_output/average_polarity_scores_subtitles.csv')
screenplays_df = pd.read_csv('../polarity_scores_output/average_polarity_scores_screenplays.csv')

# display subplot
fig, axes = plt.subplots(2, 1, figsize=(16, 8))

# subplot 1
generate_polarity_box_plots(subtitles_df, source_name='Subtitle', ax=axes[0])

# subplot
generate_polarity_box_plots(screenplays_df, source_name='Screenplay', ax=axes[1])

# add shared legend
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.13, 1.05), ncol=1)

plt.savefig('polarity_scores(subtitles + screenplays).png',dpi=300, bbox_inches="tight")
plt.tight_layout()
plt.show()

