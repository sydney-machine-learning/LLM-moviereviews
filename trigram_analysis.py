# This script will perform several sentiment analysis tests on movie reviews output from various AI APIs versus human-generated reviews
# from IMDb. The goal is to determine the accuracy of the AI-generated reviews and compare
# them to human-generated reviews. The script will output the results of the sentiment analysis.

import pandas as pd
import nltk
from nltk import trigrams
from collections import Counter
import string
from nltk.corpus import stopwords

# Download necessary NLTK data
# nltk.download('punkt')
# nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Remove punctuation and stop words
    tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
    return tokens

def generate_trigrams(text):
    tokens = preprocess_text(text)
    trigrams_list = list(trigrams(tokens))
    return trigrams_list

def trigram_analysis(reviews):
    trigram_counts = Counter()
    for review in reviews:
        trigrams_list = generate_trigrams(review)
        trigram_counts.update(trigrams_list)
    return trigram_counts.most_common(3)

def process_reviews(df, ai_client):
    results = {}
    for _, row in df.iterrows():
        movie = row['movie']
        results[movie] = {}
        for question_index in range(1, 4):
            question_reviews = row.filter(like=f'{ai_client}_context').filter(like=f'_question{question_index}').tolist()
            trigrams = trigram_analysis(question_reviews)
            results[movie][f'question{question_index}'] = trigrams
    return results

def results_to_dataframe(results):
    data = []
    for movie, questions in results.items():
        for question, trigrams in questions.items():
            trigram_list = [' '.join(trigram) for trigram, count in trigrams]
            data.append([movie, question, trigram_list])
    return pd.DataFrame(data, columns=['Movie', 'Question', 'Trigrams'])

# Read the CSV files into separate DataFrames
chatgpt_reviews_df = pd.read_csv('aireviews_chatgpt.csv', header=0)
gemini_reviews_df = pd.read_csv('aireviews_gemini.csv', header=0)
deepseek_reviews_df = pd.read_csv('aireviews_deepseek.csv', header=0)

# Process reviews and generate trigrams
chatgpt_results = process_reviews(chatgpt_reviews_df, 'chatgpt')
gemini_results = process_reviews(gemini_reviews_df, 'gemini')
deepseek_results = process_reviews(deepseek_reviews_df, 'deepseek')

# Convert results to DataFrames
chatgpt_df = results_to_dataframe(chatgpt_results)
gemini_df = results_to_dataframe(gemini_results)
deepseek_df = results_to_dataframe(deepseek_results)

# Save DataFrames to CSV files
chatgpt_df.to_csv('chatgpt_trigrams.csv', index=False)
gemini_df.to_csv('gemini_trigrams.csv', index=False)
deepseek_df.to_csv('deepseek_trigrams.csv', index=False)


