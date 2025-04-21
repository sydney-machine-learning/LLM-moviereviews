# LLM-MovieReviews: Evaluating AI-Generated Movie Reviews

This repository contains the code and output results for Group 8's research project on evaluating Large Language Models (LLMs) for generating movie reviews. The project systematically compares AI-generated reviews from various models (ChatGPT, Gemini, DeepSeek) with human reviews from the IMDb database using multiple analysis methods.

## Project Overview

This research evaluates how well different LLMs can generate movie reviews by comparing them to human reviews. The analysis includes:

- **Sentiment analysis**: Using RoBERTa models to evaluate emotional polarity (positive/negative/neutral)
- **Emotion analysis**: Analyzing emotional content (joy, anger, sadness, etc.) using DistilRoBERTa
- **Semantic similarity**: Calculating cosine similarity between AI and human reviews
- **Trigram analysis**: Examining trigrams of ai reviews and imdb reviews

## Repository Structure

```
LLM-moviereviews/
├── cosine_similarity_and_other_tests/  # Cosine similarity analysis between reviews
├── download/                           # Scripts for downloading IMDb data
├── emotions_output/                    # Emotion analysis results
├── polarity_scores_output/             # Sentiment polarity results
├── reviews_ai/                         # AI-generated review scripts and data
│   ├── screenplays/                    # Reviews based on movie screenplays
│   └── subtitles/                      # Reviews based on movie subtitles
├── screenplay_pdfs/                    # Movie screenplay source files
├── tables/                             # Processed data tables for analysis
├── trigrams_output/                    # N-gram analysis results
└── visualization/                      # Data visualization scripts and outputs
```

## Key Scripts

- **`Roberta.py`**: Performs sentiment and emotion analysis on reviews
- **`pairwise_cosine.py`**: Calculates semantic similarity between reviews
- **`generate_ai_reviews.py`**: Generates reviews using different LLM APIs
- **`trigram_analysis.py`**: Generates top trigrams for ai and imbd reviews.


## Setup and Installation

### Prerequisites
- API keys for: OpenAI, Google Gemini, DeepSeek, OMDB

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/LLM-moviereviews.git
   cd LLM-moviereviews
   ```

2. Install required dependencies:
   ```bash
   pip install pandas numpy scikit-learn torch transformers openai google-generativeai kagglehub python-dotenv
   ```

3. Create a `.env` file in the root directory with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   GEMINI_API_KEY=your_gemini_api_key
   DEEP_AI_API_KEY=your_deepseek_api_key
   ```

## Usage

### Generating AI Reviews

To generate movie reviews using different LLM models:


python reviews_ai/generate_ai_reviews.py


### Running Sentiment and Emotion Analysis

To analyze sentiment and emotions in reviews:


python Roberta.py


### Calculating Similarity

To calculate cosine similarity between different review types:


python cosine_similarity_and_other_tests/pairwise_cosine.py


## Results

Results from various analyses are stored in dedicated directories:
- Sentiment analysis: `polarity_scores_output/`
- Emotion analysis: `emotions_output/`
- Semantic similarity: `cosine_similarity_and_other_tests/`


## License

This project is intended for research purposes. Please cite this repository if using the code or results in academic work.
