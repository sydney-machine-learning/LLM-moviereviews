# This script will perform several sentiment analysis tests on movie reviews output from various AI APIs versus human-generated reviews
# from IMDb. The goal is to determine the accuracy of the AI-generated reviews and compare
# them to human-generated reviews. The script will output the results of the sentiment analysis.

import pandas as pd
import nltk
from nltk import trigrams
from collections import Counter

# Download necessary NLTK data
#nltk.download('punkt_tab')

#punkt_path = nltk.data.find('tokenizers/punkt')
#print(f"'punkt' data is downloaded to: {punkt_path}")

def generate_trigrams(text):
    tokens = nltk.word_tokenize(text)
    trigrams_list = list(trigrams(tokens))
    return trigrams_list

def trigram_analysis(reviews):
    trigram_counts = Counter()
    for review in reviews:
        trigrams_list = generate_trigrams(review)
        trigram_counts.update(trigrams_list)
    return trigram_counts.most_common(10)

# Read the CSV file
df = pd.read_csv('movies_with_subtitles_and_aireviews.csv')

# Drop the column that contains the subtitles
df = df.drop(columns=['bodyContent'])

# Extract reviews for trigram analysis
gemini_reviews = df['gemini_a'].tolist() #+ df['gemini_good'].tolist()
chatgpt_reviews = df['chatgpt_a'].tolist() #+ df['chatgpt_good'].tolist()
#deepseek_reviews = df['deepseek_bad'].tolist() + df['deepseek_good'].tolist()

# Perform trigram analysis
gemini_trigrams = trigram_analysis(gemini_reviews)
chatgpt_trigrams = trigram_analysis(chatgpt_reviews)
#deepseek_trigrams = trigram_analysis(deepseek_reviews)

# Output the results
print("Top 10 trigrams for Gemini reviews:")
print(gemini_trigrams)

print("Top 10 trigrams for ChatGPT reviews:")
print(chatgpt_trigrams)


