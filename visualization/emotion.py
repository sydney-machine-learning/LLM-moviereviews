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

def plot_emotion_by_question(ai_df, question_label, ):
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

plot_emotion_by_question(ai_df, 'question1')
plot_emotion_by_question(ai_df, 'question2')
plot_emotion_by_question(ai_df, 'question3')
