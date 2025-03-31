import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load IMDb average emotion scores
imdb_df = pd.read_csv('../emotions_output/average_emotion_scores_imdb.csv')

# Melt for visualization
imdb_melted = imdb_df.melt(id_vars='Movie', var_name='Emotion', value_name='IMDb Score')
# imdb_melted['Source'] = 'IMDb'

# Load AI average emotion scores
ai_df = pd.read_csv('../emotions_output/average_emotion_scores_subtitles.csv')
ai_df.rename(columns={'File': 'AI Model'}, inplace=True)

ai_df["AI Model"] = ai_df["AI Model"].replace({
    "aireviews_chatgpt.csv": "ChatGPT",
    "aireviews_deepseek.csv": "DeepSeek",
    "aireviews_gemini.csv": "Gemini",
    "aireviews_gemini_context_variation.csv": "Gemini (Context)"
})

ai_df = ai_df.fillna(0)

def plot_emotion_by_question(ai_df, question_label):
    question_df = ai_df[ai_df['Question'] == question_label]

    # Melt emotion columns
    melted = question_df.melt(
        id_vars=['AI Model'],
        value_vars=['surprise', 'anger', 'neutral', 'disgust', 'sadness', 'fear', 'joy'],
        var_name='Emotion',
        value_name='Score'
    )

    # Compute average emotion score per model
    avg_scores = melted.groupby(['AI Model', 'Emotion'])['Score'].mean().reset_index()

    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(data=avg_scores, x='Emotion', y='Score', hue='AI Model', palette="Set2")
    plt.title(f'Emotion Scores for {question_label.capitalize()} by AI Model')
    plt.xticks(rotation=30)
    plt.tight_layout()
    # save fig
    plt.savefig(f"emotion_scores_{question_label}.png", dpi=300, bbox_inches="tight")
    plt.show()

# plot_emotion_by_question(ai_df, 'question1')
# plot_emotion_by_question(ai_df, 'question2')
# plot_emotion_by_question(ai_df, 'question3')

def plot_emotion_pies_by_question(ai_df, question_label):
    # filter for question
    ai_subset = ai_df[(ai_df['Question'] == question_label)]
    ai_sub_melted = ai_subset.melt(
        id_vars=['AI Model', 'Movie'],       # Columns to keep
        value_vars=['fear', 'joy', 'sadness', 'anger', 'disgust', 'neutral', 'surprise'],  # Emotion columns
        var_name='Emotion',                  # New column name for melted column names
        value_name='Score'                   # New column name for the emotion values
    )

    # print(ai_sub_melted.head())

    # calculate average score across emotion
    avg_emotions_df = ai_sub_melted.groupby(['AI Model','Emotion'])['Score'].mean().reset_index()
    avg_emotions_df.columns = ['AI Model','Emotion','Average Score']

    models = avg_emotions_df['AI Model'].unique()

    # Set up subplots (one per model)
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 6))
    colors = sns.color_palette("Set2", n_colors=7)
    # Generate pie chart for each model
    legend_labels = None
    for ax, model in zip(axes, models):
        model_data = avg_emotions_df[avg_emotions_df['AI Model'] == model].set_index('Emotion')
        ax.pie(model_data['Average Score'],
               # labels=model_data.index,
               labels = None,
               autopct='%1.1f%%',
               startangle=140,
               colors = colors,
               pctdistance=1.1,
               textprops={'fontsize': 8})

        ax.set_title(f"{model}")
        if legend_labels is None:
            legend_labels = model_data.index

    fig.legend(legend_labels, loc='upper left', title='Emotion')

    # Add a main title
    fig.suptitle(f"Emotion Distribution by AI Model ({question_label.capitalize()})", fontsize=16, y=0.9)
    plt.tight_layout()
    # save fig
    plt.savefig(f"emotion_scores_{question_label}.png", dpi=300, bbox_inches="tight")
    plt.show()

plot_emotion_pies_by_question(ai_df, 'question1')
plot_emotion_pies_by_question(ai_df, 'question2')
plot_emotion_pies_by_question(ai_df, 'question3')