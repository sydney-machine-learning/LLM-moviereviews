"""
Movie Review Sentiment and Emotion Analysis

This script conducts comprehensive sentiment and emotion analysis on movie reviews using transformer models:
- Sentiment analysis: RoBERTa model (cardiffnlp/twitter-roberta-base-sentiment) for polarity scoring
- Emotion analysis: DistilRoBERTa model (j-hartmann/emotion-english-distilroberta-base) for emotion detection

Data sources processed:
1. AI-generated reviews (from both screenplay and subtitles sources)
2. IMDb user reviews (filtered by various rating thresholds)


- Handles long text by chunking and averaging scores
- Processes both positive and negative review sentiments
- Identifies multiple emotions (anger, joy, sadness, fear, surprise, disgust, neutral)


Output files:
- Polarity scores: negative/neutral/positive sentiment distributions
- Emotion scores: distribution of emotional content across reviews
- Results saved to CSV files in polarity_scores_output/ and emotions_output/ directories
"""



import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline
from scipy.special import softmax
import os

def get_polarity_scores(review):
    # Load the RoBERTa tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
    model = RobertaForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')

    # Tokenize the review with truncation
    inputs = tokenizer(review, return_tensors='pt', max_length=512, truncation=True)

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

def get_polarity_scores_for_long_text(review):
    # Load the RoBERTa tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
    model = RobertaForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')

    # Tokenize the review into chunks of 512 tokens
    tokens = tokenizer.tokenize(review)
    chunks = [tokens[i:i + 512] for i in range(0, len(tokens), 512)]

    # Initialize cumulative scores
    cumulative_scores = {'negative': 0, 'neutral': 0, 'positive': 0}

    # Process each chunk
    for chunk in chunks:
        inputs = tokenizer.convert_tokens_to_ids(chunk)
        inputs = torch.tensor([inputs])

        # Get the model's output
        with torch.no_grad():
            outputs = model(inputs)

        # Apply softmax to get probabilities
        scores = outputs.logits[0].numpy()
        scores = softmax(scores)

        # Accumulate the scores
        cumulative_scores['negative'] += scores[0]
        cumulative_scores['neutral'] += scores[1]
        cumulative_scores['positive'] += scores[2]

    # Average the scores
    num_chunks = len(chunks)
    average_scores = {key: value / num_chunks for key, value in cumulative_scores.items()}

    return average_scores

def get_average_polarity_scores():
    # List of folders and corresponding output files
    folders = [
        ('LLM Generated Reviews/screenplays', 'Polarity Analysis/average_polarity_scores_screenplays.csv'),
        ('LLM Generated Reviews/subtitles', 'Polarity Analysis/average_polarity_scores_subtitles.csv')
    ]

    for folder, output_file in folders:
        all_results = {}

        for file in os.listdir(folder):
            if file.endswith('.csv'):
                file_path = os.path.join(folder, file)
                df = pd.read_csv(file_path)

                results = {}
                for _, row in df.iterrows():
                    movie = row['movie']
                    print(movie)
                    if movie not in results:
                        results[movie] = {'question1': [], 'question2': [], 'question3': []}

                    for question_index in range(1, 4):
                        question_key = f'question{question_index}'
                        question_reviews = row.filter(like=f'_context').filter(like=f'_question{question_index}').tolist()

                        # Calculate polarity scores for each review
                        for review in question_reviews:
                            polarity_scores = get_polarity_scores_for_long_text(review)
                            print(movie, polarity_scores)
                            results[movie][question_key].append(polarity_scores)

                # Calculate average polarity scores
                average_results = {}
                for movie, questions in results.items():
                    average_results[movie] = {}
                    for question, scores in questions.items():
                        avg_negative = sum(score['negative'] for score in scores) / len(scores)
                        avg_neutral = sum(score['neutral'] for score in scores) / len(scores)
                        avg_positive = sum(score['positive'] for score in scores) / len(scores)
                        average_results[movie][question] = {
                            'average_negative': avg_negative,
                            'average_neutral': avg_neutral,
                            'average_positive': avg_positive
                        }

                all_results[file] = average_results

        # Convert the nested dictionary to a DataFrame
        rows = []
        for file, movies in all_results.items():
            for movie, questions in movies.items():
                for question, scores in questions.items():
                    row = {
                        'File': file,
                        'Movie': movie,
                        'Question': question,
                        'Average Negative': scores['average_negative'],
                        'Average Neutral': scores['average_neutral'],
                        'Average Positive': scores['average_positive']
                    }
                    rows.append(row)

        df = pd.DataFrame(rows)

        # Save the DataFrame to a CSV file
        df.to_csv(output_file, index=False)
        print(f"Polarity scores saved to {output_file}")

def save_results_to_csv(all_results, filename):
    # Convert the nested dictionary to a DataFrame
    rows = []
    for model, movies in all_results.items():
        for movie, questions in movies.items():
            for question, scores in questions.items():
                row = {
                    'Model': model,
                    'Movie': movie,
                    'Question': question,
                    'Average Negative': scores['average_negative'],
                    'Average Neutral': scores['average_neutral'],
                    'Average Positive': scores['average_positive']
                }
                rows.append(row)

    df = pd.DataFrame(rows)

    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)

def get_emotion_scores(review):
    # Load the emotion analysis pipeline
    emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

    # Tokenize the review into chunks of 512 tokens
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    tokens = tokenizer.tokenize(review)
    chunks = [tokens[i:i + 510] for i in range(0, len(tokens), 510)]  # Use 510 to account for special tokens

    # Initialize cumulative scores with all emotions set to 0
    cumulative_scores = {"anger": 0, "joy": 0, "sadness": 0, "fear": 0, "surprise": 0, "disgust": 0, "neutral": 0}

    # Process each chunk
    for chunk in chunks:
        chunk_text = tokenizer.convert_tokens_to_string(chunk)
        emotion_scores = emotion_pipeline(chunk_text)

        # Convert the list of dictionaries to a single dictionary
        emotion_dict = {item['label']: item['score'] for item in emotion_scores[0]}

        # Accumulate the scores, ensuring all keys are present
        for emotion in cumulative_scores.keys():
            cumulative_scores[emotion] += emotion_dict.get(emotion, 0)

    # Average the scores
    num_chunks = len(chunks)
    average_scores = {key: value / num_chunks for key, value in cumulative_scores.items()}

    # Sort the emotions by score and get the top 5
    top_emotions = dict(sorted(average_scores.items(), key=lambda item: item[1], reverse=True)[:5])

    return top_emotions

def emotions_ai_reviews():
    # List of folders and corresponding output files
    folders = [
        ('LLM Generated Reviews/screenplays', 'Emotion Analysis/average_emotion_scores_screenplays.csv'),
        ('LLM Generated Reviews/subtitles', 'Emotion Analysis/average_emotion_scores_subtitles.csv')
    ]

    for folder, output_file in folders:
        all_results = {}

        for file in os.listdir(folder):
            if file.endswith('.csv'):
                file_path = os.path.join(folder, file)
                df = pd.read_csv(file_path)

                results = {}

                for _, row in df.iterrows():
                    movie = row['movie']
                    print(movie)
                    if movie not in results:
                        results[movie] = {'question1': [], 'question2': [], 'question3': []}

                    for question_index in range(1, 4):
                        question_key = f'question{question_index}'
                        question_reviews = row.filter(like=f'_context').filter(like=f'_question{question_index}').tolist()

                        # Calculate emotion scores for each review
                        for review in question_reviews:
                            emotion_scores = get_emotion_scores(review)
                            results[movie][question_key].append(emotion_scores)

                # Calculate average emotion scores
                average_results = {}
                for movie, questions in results.items():
                    average_results[movie] = {}
                    for question, scores in questions.items():
                        # Ensure all keys are initialized to 0 to avoid KeyError
                        all_keys = set(key for score in scores for key in score.keys())
                        avg_scores = {key: sum(score.get(key, 0) for score in scores) / len(scores) for key in all_keys}
                        average_results[movie][question] = avg_scores

                all_results[file] = average_results

        # Convert the nested dictionary to a DataFrame
        rows = []
        for file, movies in all_results.items():
            for movie, questions in movies.items():
                for question, scores in questions.items():
                    row = {
                        'File': file,
                        'Movie': movie,
                        'Question': question,
                        **scores
                    }
                    rows.append(row)

        df = pd.DataFrame(rows)

        # Save the DataFrame to a CSV file
        df.to_csv(output_file, index=False)
        print(f"Emotion scores saved to {output_file}")

def get_average_polarity_scores_imdb():
    # Define base directory relative to the script's location
    script_path = os.path.abspath(__file__)
    utils_dir = os.path.dirname(script_path)
    base_dir = os.path.dirname(utils_dir) # This is the project root

    # Read the CSV files
    imdb_reviews_path = os.path.join(base_dir, 'IMDb Reviews', 'all_imdb_reviews.csv')
    selected_movie_info_path = os.path.join(base_dir, 'selected_movie_info.csv')
    
    imdb_reviews_df = pd.read_csv(imdb_reviews_path)
    selected_movie_info_df = pd.read_csv(selected_movie_info_path)

    # Filter reviews with a rating below 6
    filtered_reviews_df = imdb_reviews_df[imdb_reviews_df['Rating'] < 6]

    

    # Merge with selected_movie_info_df to get movie titles
    merged_df = pd.merge(filtered_reviews_df, selected_movie_info_df, on='imdb_id')

    print(merged_df.head())

    results = {}

    for _, row in merged_df.iterrows():
        movie = row['movie']
        if movie not in results:
            results[movie] = []

        review = row['Review']
        polarity_scores = get_polarity_scores_for_long_text(review)
        print(polarity_scores, movie)
        results[movie].append(polarity_scores)

    # Calculate average polarity scores
    average_results = []
    for movie, scores in results.items():
        avg_negative = sum(score['negative'] for score in scores) / len(scores)
        avg_neutral = sum(score['neutral'] for score in scores) / len(scores)
        avg_positive = sum(score['positive'] for score in scores) / len(scores)
        average_results.append({
            'Movie': movie,
            'Average Negative': avg_negative,
            'Average Neutral': avg_neutral,
            'Average Positive': avg_positive
        })

    # Convert the results to a DataFrame
    df = pd.DataFrame(average_results)

    # Save the DataFrame to a CSV file
    output_file_name = 'average_polarity_scores_imdb.csv'
    output_dir_path = os.path.join(base_dir, 'Polarity Analysis')
    os.makedirs(output_dir_path, exist_ok=True) # Ensure the directory exists
    output_path = os.path.join(output_dir_path, output_file_name)
    df.to_csv(output_path, index=False)


def get_average_emotion_scores_imdb():
    # Define base directory relative to the script's location
    script_path = os.path.abspath(__file__)
    utils_dir = os.path.dirname(script_path)
    base_dir = os.path.dirname(utils_dir) # This is the project root

    # Read the IMDb reviews and movie information
    imdb_reviews_path = os.path.join(base_dir, 'IMDb Reviews', 'all_imdb_reviews.csv')
    selected_movie_info_path = os.path.join(base_dir, 'selected_movie_info.csv')

    imdb_reviews_df = pd.read_csv(imdb_reviews_path)
    selected_movie_info_df = pd.read_csv(selected_movie_info_path)

    # Merge IMDb reviews with movie information to get movie titles
    merged_df = pd.merge(imdb_reviews_df, selected_movie_info_df, on='imdb_id')

    results = {}

    for _, row in merged_df.iterrows():
        movie = row['movie']
        if movie not in results:
            results[movie] = []

        review = row['Review']
        emotion_scores = get_emotion_scores(review)
        results[movie].append(emotion_scores)

    # Calculate average emotion scores per movie
    average_results = []
    for movie, scores in results.items():
        avg_scores = {key: sum(score.get(key, 0) for score in scores) / len(scores) for key in scores[0].keys()}
        average_results.append({
            'Movie': movie,
            **avg_scores
        })

    # Convert the results to a DataFrame
    df = pd.DataFrame(average_results)

    # Save the DataFrame to a CSV file
    output_file_name = 'average_emotion_scores_imdb.csv'
    output_dir_path = os.path.join(base_dir, 'Emotion Analysis')
    os.makedirs(output_dir_path, exist_ok=True) # Ensure the directory exists
    output_path = os.path.join(output_dir_path, output_file_name)
    df.to_csv(output_path, index=False)
    #print(f"Average emotion scores per movie saved to {output_path}")

if __name__ == "__main__":
    #get_average_polarity_scores()
    #save_results_to_csv(ai_average_polarity_scores, 'average_polarity_scores_ai.csv')
    get_average_polarity_scores_imdb()    
    #emotions_ai_reviews()
    #get_average_emotion_scores_imdb()   # working
