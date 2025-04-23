import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# load the data
subtitles_df = pd.read_csv('../polarity_scores_output/average_polarity_scores_subtitles.csv')
screenplays_df = pd.read_csv('../polarity_scores_output/average_polarity_scores_screenplays.csv')
imdb_above7 = pd.read_csv('../average_polarity_scores_imdb_above_7.csv')
imdb_below6 = pd.read_csv('../average_polarity_scores_imdb_below_6.csv')
imdb_6_and_7 = pd.read_csv('../average_polarity_scores_imdb_between_6_and_7.csv')

# replace file name with AI model name
def replace_model_name(file_name):
    # Check for 'gemini' with context variation first
    if 'gemini_screenplays_context_variation' in file_name.lower() or 'gemini_context_variation' in file_name.lower():
        return 'Gemini (detailed context)'
    elif 'gemini_screenplays' in file_name.lower():
        return 'Gemini 2'
    elif 'chatgpt' in file_name.lower():
        return 'ChatGPT-4o'
    elif 'deepseek' in file_name.lower():
        return 'DeepSeek'
    elif 'gemini' in file_name.lower():
        return 'Gemini 2'
    else:
        return 'unknown'

def prepare_data(df,question=None):
    # replace file name with AI model name
    df['Source'] = df['File'].apply(replace_model_name)
    print(df['Source'])

    # split to good, bad, neutral review based on question column
    if question:
        df = df[df['Question'] == question]
    return df[['Movie', 'Source', 'Average Negative', 'Average Neutral', 'Average Positive']]

# label the IMDb data with rating
imdb_above7['Source'] = 'IMDb Above 7'
imdb_below6['Source'] = 'IMDb Below 6'
imdb_6_and_7['Source'] = 'IMDb 6 and 7'

# change wide-format to long format
def melt_data(df):
    return pd.melt(df, id_vars = ['Movie', 'Source'],
                   value_vars=['Average Negative', 'Average Neutral', 'Average Positive'],
                   var_name='Polarity Type', value_name='Polarity Score')


def generate_polarity_barplot(source_name):
    if source_name == 'Subtitle':
        df = subtitles_df
    else:
        df = screenplays_df

    ai_good_reviews = prepare_data(df, 'question2')
    ai_bad_reviews = prepare_data(df, 'question1')
    ai_neutral_reviews = prepare_data(df, 'question3')

    # conmine IMDb reviews and correspoding AI reviews
    combined_data = pd.concat([ai_good_reviews, imdb_above7], ignore_index=True)
    combined_bad_reviews = pd.concat([ai_bad_reviews, imdb_below6], ignore_index=True)
    combined_neutral_reviews = pd.concat([ai_neutral_reviews, imdb_6_and_7], ignore_index=True)

    melted_good_reviews = melt_data(combined_data)
    melted_bad_reviews = melt_data(combined_bad_reviews)
    melted_neutral_reviews = melt_data(combined_neutral_reviews)


    # plot
    fig, axes = plt.subplots(3, 1, figsize=(12,9.5), sharey=True)
    sns.set(style="whitegrid")
    palette = sns.color_palette("Set2", 3)

    plot_data = [
        ('IMDb < 6 vs LLM (Q1 - Negative Reviews)', melted_bad_reviews, axes[0]),
        ('IMDb > 7 vs LLM (Q2 - Positve Reviews)', melted_good_reviews, axes[1]),
        ('IMDb 6–7 vs LLM (Q3 - Neutral Reviews)', melted_neutral_reviews, axes[2])
    ]

    for title, df, ax in plot_data:
        sns.barplot(x='Source', y='Polarity Score', hue='Polarity Type',
                    data=df, ax=ax, palette=palette,
                    errorbar=None)

        ax.set_title(title)
        ax.set_xlabel('')
        ax.set_ylabel('Polarity Score')
        ax.tick_params(axis='x', labelsize=11)
        ax.get_legend().remove()
        ax.grid(False)

    # shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', ncol=1,
               bbox_to_anchor=(0.1, 1.0),
               fontsize=10, title_fontsize=11)


    # layout adjustment
    # plt.tight_layout(rect=(0, 0, 1, 0.94))
    fig.subplots_adjust(hspace=0.6, top=0.92)
    plt.grid(False)
    plt.savefig(f"combined_polarity_score_comparison_{source_name}", dpi=300, bbox_inches="tight")
    plt.show()

generate_polarity_barplot("Subtitle")
generate_polarity_barplot("Screenplays")


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
            "aireviews_chatgpt.csv": "ChatGPT-4o",
            "aireviews_deepseek.csv": "DeepSeek",
            "aireviews_gemini.csv": "Gemini 2",
            "aireviews_gemini_context_variation.csv": "Gemini(detailed)"
        })
        # plot_title = "(a) Sentiment Polarity Distributioin by Subtitles"
    elif source_name == 'Screenplay':
        ai_df["AI Model"] = ai_df["AI Model"].replace({
            "aireviews_chatgpt_screenplays.csv": "ChatGPT-4o",
            "aireviews_deepseek_screenplays.csv": "DeepSeek",
            "aireviews_gemini_screenplays.csv": "Gemini 2" ,
            "aireviews_gemini_screenplays_context_variation.csv": "Gemini(detailed)"
        })
        # plot_title = "(b) Sentiment Polarity Distributioin by Screenplays"
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
    # ax.set_title(f'{source_name}', fontsize=16)

    # adjust subplot title location
    # ax.text(0.5, -0.2, plot_title, fontsize=18, ha='center', transform=ax.transAxes)
    if ax == axes[0]:
        ax.text(0.5, -0.2, "(a) Sentiment Polarity Distributioin by Subtitles", fontsize=16, ha='center', transform=ax.transAxes)
    else:
        ax.text(0.5, -0.2, "(b) Sentiment Polarity Distributioin by Screenplays", fontsize=16, ha='center', transform=ax.transAxes)

    ax.set_xlabel('', fontsize=18)
    ax.set_ylabel('Score', fontsize=18)

    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=14)
    ax.grid(False)

    ax.get_legend().remove()
    if ax == axes[0]:
        ax.set_xlabel('')

# load the subtitle and screenplay data

# display subplot
fig, axes = plt.subplots(2, 1, figsize=(16, 12))


# subplot 1
generate_polarity_box_plots(subtitles_df, source_name='Subtitle', ax=axes[0])

# subplot
generate_polarity_box_plots(screenplays_df, source_name='Screenplay', ax=axes[1])

# add shared legend
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.08, 1.02), ncol=1,fontsize=14)

# fig.subplots_adjust(hspace=0.6, top=0.92)
plt.tight_layout(pad=4.0, rect=(0, 0, 1, 0.96))

plt.savefig('polarity_scores(subtitles + screenplays).png',dpi=300, bbox_inches="tight")
# plt.tight_layout()
plt.show()

