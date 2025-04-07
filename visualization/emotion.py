import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

# Load AI data
ai_df = pd.read_csv('../emotions_output/average_emotion_scores_subtitles.csv')
ai_df.rename(columns={'File': 'AI Model'}, inplace=True)

ai_df["AI Model"] = ai_df["AI Model"].replace({
    "aireviews_chatgpt.csv": "ChatGPT",
    "aireviews_deepseek.csv": "DeepSeek",
    "aireviews_gemini.csv": "Gemini",
    "aireviews_gemini_context_variation.csv": "Gemini (detailed)"
})
ai_df = ai_df.fillna(0)


# Load IMDb average emotion scores
imdb_df = pd.read_csv('../emotions_output/average_emotion_scores_imdb.csv')
# Melt for visualization
imdb_melted = imdb_df.melt(id_vars='Movie', var_name='Emotion', value_name='Score')
imdb_melted = imdb_melted.fillna(0)
#print(imdb_melted)

# calculate average emotion scores
imdb_avg_emotions = imdb_melted.groupby('Emotion')['Score'].mean().reset_index()
imdb_avg_emotions.columns = ['Emotion', 'Average Score']
#print(imdb_avg_emotions)

# pie chart for imdb average emotion scores
colors = sns.color_palette("Set2", n_colors=7)
plt.figure(figsize=(8, 6))
plt.pie(imdb_avg_emotions['Average Score'],
        labels= None,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors
        )

plt.legend(imdb_avg_emotions['Emotion'],
           loc='lower center',
           title='Emotion',
           bbox_to_anchor=(0.5, -0.08),
           ncol=7)

plt.title('IMDb')
plt.axis('equal')
# plt.savefig('emotion_scores_imdb.png',dpi=300, bbox_inches="tight")
# plt.show()

def generate_emotion_pie_charts(ai_df, source_name='ai'):
    ai_df = ai_df.copy()
    ai_df.rename(columns={'File': 'AI Model'}, inplace=True)
    # mapping
    if source_name == 'subtitle':
        ai_df["AI Model"] = ai_df["AI Model"].replace({
            "aireviews_chatgpt.csv": "ChatGPT",
            "aireviews_deepseek.csv": "DeepSeek",
            "aireviews_gemini.csv": "Gemini",
            "aireviews_gemini_context_variation.csv": "Gemini (Context)"
        })
    elif source_name == 'screenplay':
        ai_df["AI Model"] = ai_df["AI Model"].replace({
            "aireviews_chatgpt_screenplays.csv": "ChatGPT",
            "aireviews_deepseek_screenplays.csv": "DeepSeek",
            "aireviews_gemini_screenplays.csv": "Gemini",
            "aireviews_gemini_screenplays_context_variation.csv": "Gemini (Context)"
        })

    # some emotions are not detected, replace NA with 0
    ai_df = ai_df.fillna(0)
    # Loop over questions
    for question_label in ['question1', 'question2', 'question3']:
        # Filter for this question
        ai_subset = ai_df[ai_df['Question'] == question_label]

        # Melt the emotion columns
        ai_sub_melted = ai_subset.melt(
            id_vars=['AI Model', 'Movie'],
            value_vars=['fear', 'joy', 'sadness', 'anger', 'disgust', 'neutral', 'surprise'],
            var_name='Emotion',
            value_name='Score'
        )
        # Calculate average score
        avg_emotions_df = ai_sub_melted.groupby(['AI Model', 'Emotion'])['Score'].mean().reset_index()
        avg_emotions_df.columns = ['AI Model', 'Emotion', 'Average Score']
        models = avg_emotions_df['AI Model'].unique()
        # Set up subplots (one per model)
        fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 6))
        colors = sns.color_palette("Set2", n_colors=7)
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
        fig.suptitle(f"Emotion Distribution by AI Model ({question_label.capitalize()})", fontsize=16, y=0.95)

        # add source at the bottom (reivews generated based on subtitles or screenplays)
        fig.text(0.5, 0.01,
                 f"Emotion analysis based on AI reviews generated from {source_name}",
                 fontsize=10, ha='center')

        plt.tight_layout()
        question_short = f"q{question_label[-1]}"
        # save fig
        plt.savefig(f"emotion_scores_{source_name}_{question_short}.png", dpi=300, bbox_inches="tight")
        # plt.show()

# bar plot by question
def plot_emotion_by_question_subplots(ai_df):
    question_labels = ['question1', 'question2', 'question3']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, question_label in enumerate(question_labels):
        question_df = ai_df[ai_df['Question'] == question_label]
        # melt emotion columns
        melted = question_df.melt(
            id_vars=['AI Model'],
            value_vars=['surprise', 'anger', 'neutral', 'disgust', 'sadness', 'fear', 'joy'],
            var_name='Emotion',
            value_name='Score'
        )

        # calculate average emotion score per model
        avg_scores = melted.groupby(['AI Model', 'Emotion'])['Score'].mean().reset_index()

        # plot the barplot for each question in its respective subplot
        sns.barplot(data=avg_scores, x='Emotion', y='Score', hue='AI Model', palette="Set2",ax=axes[idx])
        # delete subplot legend
        axes[idx].set_title(f'{question_label.capitalize()}')
        # axes[idx].set_title(f'Emotion Scores for {question_label.capitalize()} by AI Model')
        axes[idx].tick_params(axis='x')
        axes[idx].legend().set_visible(False)
        axes[idx].set_xlabel('')

    plt.tight_layout()
    # add shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', ncol=4, bbox_to_anchor=(0.5, 0.01))
    plt.savefig('emotion_scores_by_questions.png',dpi=300, bbox_inches="tight")
    plt.show()

plot_emotion_by_question_subplots(ai_df)

# bar plot for average emotion scores of each movie
emotions = ai_df.columns[3:]
ai_models = ai_df['AI Model'].unique()
movies = ai_df['Movie'].unique()

# store the results
average_emotions_list = []

for ai_model in ai_models:
    for movie in movies:
        # filter each movie and ai model
        filtered_df = ai_df[(ai_df['Movie'] == movie) & (ai_df['AI Model'] == ai_model)]

        # compute average emotion scores
        average_emotion_df = filtered_df[emotions].mean().to_frame().T

        average_emotion_df['AI Model'] = ai_model
        average_emotion_df['Movie'] = movie

        average_emotions_list.append(average_emotion_df)

# dataframe
average_emotions_df = pd.concat(average_emotions_list)
average_emotions_df.reset_index(drop=True, inplace=True)

# print(average_emotions_df)
# re order the columns
new_order = ['AI Model', 'Movie'] + list(emotions)
average_emotions_df = average_emotions_df[new_order]
print(average_emotions_df)

# melt the dataframe, from wide format to long format
melted_df = average_emotions_df.melt(id_vars=['AI Model', 'Movie'],
                       value_vars=emotions,
                       var_name='Emotion',
                       value_name='Score')

plt.figure(figsize=(12, 8))
sns.barplot(data=melted_df, x='Emotion', y='Score', hue='Movie',
            palette="Set2", errorbar=None)

# title and labels
# plt.title('Average Emotion Scores by Movie', fontsize=16)
plt.xlabel('Emotion', fontsize=12)
plt.ylabel('Average Score', fontsize=12)

plt.legend(loc='upper left')
# show the plot
plt.savefig('emotion_scores_by_moive.png',dpi=300, bbox_inches="tight")
plt.tight_layout()
plt.show()


# For subtitles
# ai_subtitle = pd.read_csv('../emotions_output/average_emotion_scores_subtitles.csv')
# generate_emotion_pie_charts(ai_subtitle, source_name="subtitle")
#
# # For screenplays
# ai_screenplay = pd.read_csv('../emotions_output/average_emotion_scores_screenplays.csv')
# generate_emotion_pie_charts(ai_screenplay, source_name="screenplay")

# pie chart for AI models average emotion scores

ai_df = pd.read_csv('../emotions_output/average_emotion_scores_subtitles.csv')
colors = sns.color_palette("Set2", n_colors=7)
models = ['chatgpt', 'deepseek', 'gemini', 'gemini_context']
numeric_columns = ai_df.select_dtypes(include=['float64', 'int64']).columns

fig, axes = plt.subplots(1, len(models), figsize=(18, 6))
# store legend names
model_data_list = []
for idx, model in enumerate(models):
    model_data = ai_df[ai_df['File'].str.contains(model)][numeric_columns].mean(axis=0)
    model_data_list.append(model_data)
    axes[idx].pie(model_data,
                  autopct='%1.1f%%',
                  startangle=90,
                  textprops={'fontsize':8},
                  pctdistance=0.8,
                  colors=colors)
    axes[idx].axis('equal')
    axes[idx].set_title(f'{model}', y = 1.0)

# legends
legend_labels = model_data_list[0].index

# plt.suptitle('Emotion Distribution of AI Models')
plt.tight_layout()
fig.legend(legend_labels, loc='center', ncol=7, bbox_to_anchor=(0.5, 0.05))
plt.savefig('emotion_scores_ai_models.png',dpi=300, bbox_inches="tight")
plt.show()

# table for average emotion scores (AI models and IMDb)
model_avg_emotions = {}
for model in models:
    model_data = ai_df[ai_df['File'].str.contains(model)][numeric_columns].mean(axis=0)
    model_avg_emotions[model] = model_data
avg_emotions_df = pd.DataFrame(model_avg_emotions).T

# convert to df
imdb_avg_emotions_df = imdb_avg_emotions.set_index('Emotion').T
imdb_avg_emotions_df.index = ['IMDb']
imdb_avg_emotions_df.columns.name = None

# change column order:
change_column = avg_emotions_df.columns
imdb_avg_emotions_df = imdb_avg_emotions_df.reindex(columns=change_column)
# print(imdb_avg_emotions_df)

combined_df = avg_emotions_df.copy()
combined_df.loc['IMDb'] = imdb_avg_emotions_df.values[0]
combined_df = combined_df.round(3)
# print(combined_df)

plt.figure(figsize=(12, 6))
plt.table(cellText=combined_df.values,
          colLabels=combined_df.columns,
          rowLabels=combined_df.index,
          loc='center',
          cellLoc='center',
          bbox=[0.0, 0.0, 1.0, 1.0])  # Adjust position of the table
plt.axis('off')
# plt.title('Combined Emotion Scores Table (AI Models + IMDb as 6th row)')
plt.savefig('emotion_scores_table.png',dpi=300, bbox_inches="tight")
plt.show()



# quantitative analysis
# perform one-way ANOVA for each model, to test if there are any significant differences
# in the emotion scores between the AI models and IMDb for each emotion

# store ANOVA test results
anova_results = {}
unrounded_combined_df = avg_emotions_df.copy()
unrounded_combined_df.loc['IMDb'] = imdb_avg_emotions_df.values[0]
for emotion in unrounded_combined_df.columns:
    models_data = [unrounded_combined_df[emotion].values for model in unrounded_combined_df.index]
    # perform ANOVA
    f_stat, p_value = f_oneway(*models_data)
    anova_results[emotion] = {'F-statistic': f_stat, 'p-value': p_value}

anova_df = pd.DataFrame(anova_results).T
# print(anova_df)