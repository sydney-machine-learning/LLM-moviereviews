import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline
from scipy.special import softmax

def get_polarity_scores(review):
    # Load the RoBERTa tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
    model = RobertaForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')

    # Tokenize the review
    inputs = tokenizer(review, return_tensors='pt')

    # Get the model's output
    with torch.no_grad():
        outputs = model(**inputs)

    # Apply softmax to get probabilities
    scores = outputs.logits[0].numpy()
    scores = softmax(scores)

    # Create a dictionary with the polarity scores
    polarity_scores = {
        'negative': scores[0],
        'neutral': scores[1],
        'positive': scores[2]
    }

    return polarity_scores

def get_emotion_scores(review):
    # Load the emotion analysis pipeline
    emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

    # Get the emotion scores
    emotion_scores = emotion_pipeline(review)

    # Convert the list of dictionaries to a single dictionary
    emotion_dict = {item['label']: item['score'] for item in emotion_scores[0]}

    return emotion_dict

def analyze_reviews():
    # Read the CSV file
    df = pd.read_csv('movies_with_subtitles_and_aireviews.csv')

    # Filter reviews for "The Shawshank Redemption"
    shawshank_reviews = df[df['movie'] == 'The Shawshank Redemption']

    # Extract AI-generated reviews
    gemini_reviews = shawshank_reviews['gemini_a'].tolist()
    chatgpt_reviews = shawshank_reviews['chatgpt_a'].tolist()

    # Analyze polarity scores and emotion scores for each review
    all_reviews = gemini_reviews + chatgpt_reviews
    for review in all_reviews:
        polarity_scores = get_polarity_scores(review)
        emotion_scores = get_emotion_scores(review)
        print(f"Review: {review}")
        print(f"Polarity Scores: {polarity_scores}")
        print(f"Emotion Scores: {emotion_scores}")
        print()

if __name__ == "__main__":
    analyze_reviews()