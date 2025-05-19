# LLM-MovieReviews: Evaluating AI-Generated Movie Reviews

This repository contains the code and output results for Group 8's research project on evaluating Large Language Models (LLMs) for generating movie reviews. The project systematically compares AI-generated reviews from various models (gpt4o, Gemini 2.0, DeepSeek V3) with human reviews from the IMDb database using multiple analysis methods.

## Project Overview

This research evaluates how well different LLMs can generate movie reviews by comparing them to human reviews. The analysis includes:

- **Sentiment analysis**: Using RoBERTa models to evaluate emotional polarity (positive/negative/neutral)
- **Emotion analysis**: Analyzing emotional content (joy, anger, sadness, etc.) using DistilRoBERTa
- **Semantic similarity**: Calculating cosine similarity between AI and human reviews
- **Trigram analysis**: Examining trigrams of AI reviews and IMDb reviews

## Repository Structure

```
LLM-moviereviews/
├── README.md
├── selected_movie_info.csv
├── titles with awards and categories.txt
├── Cosine Similarity/
│   ├── cosine_similarity_results_screenplays_by_movie.csv
│   ├── cosine_similarity_results_screenplays_by_question.csv
│   ├── cosine_similarity_results_subtitles_by_movie.csv
│   ├── cosine_similarity_results_subtitles_by_question.csv
│   ├── pairwise_cosine.py
│   ├── README.md
│   └── within_imdb_similarity_results.csv
├── Data/
│   ├── cleaned_screenplays.csv
│   ├── cleaned_subtitles.csv
│   └── screenplay files/           # Movie screenplay source PDF files (contains .pdf files)
├── Emotion Analysis/               # Emotion analysis results (contains .csv files)
│   ├── average_emotion_scores_imdb.csv
│   ├── average_emotion_scores_screenplays.csv
│   └── average_emotion_scores_subtitles.csv
├── IMDb Reviews/                   # IMDb review data
│   └── all_imdb_reviews.csv
├── LLM Generated Reviews/          # AI-generated review scripts and data
│   ├── generate_ai_reviews.py
│   ├── screenplays/                # Reviews based on movie screenplays (contains .csv files)
│   └── subtitles/                  # Reviews based on movie subtitles (contains .csv files)
├── Polarity Analysis/              # Sentiment polarity results (contains .csv files)
│   ├── average_polarity_scores_imdb_between_6_and_7.csv
│   ├── average_polarity_scores_imdb_rated_above7.csv
│   ├── average_polarity_scores_imdb_rated_below6.csv
│   ├── average_polarity_scores_imdb.csv
│   ├── average_polarity_scores_screenplays.csv
│   └── average_polarity_scores_subtitles.csv
├── Questionaire/                   # Questionnaire data and processing script
│   ├── questionnaire_review.csv
│   └── questionnaire.py
├── tables/                         # Processed data tables for analysis (contains .csv files)
│   ├── Emotion_Score_Comparison__Percentage_Screenplays.csv
│   └── Emotion_Score_Comparison__Percentage_Subtitles.csv
├── Trigram Analysis/               # N-gram analysis results and script
│   ├── aireviews_chatgpt_screenplays_trigrams.csv
│   ├── aireviews_chatgpt_trigrams.csv
│   ├── aireviews_deepseek_screenplays_trigrams.csv
│   ├── aireviews_deepseek_trigrams.csv
│   ├── aireviews_gemini_screenplays_trigrams.csv
│   ├── aireviews_gemini_screenplays_v2_trigrams.csv
│   ├── aireviews_gemini_trigrams.csv
│   ├── aireviews_gemini_v2_trigrams.csv
│   ├── all_imdb_review_trigrams.csv
│   ├── shawshank_imdb_10_trigrams.csv
│   └── trigram_analysis.py
├── utils/                          # Utility scripts (data extraction, processing, RoBERTa model)
│   ├── consolidate_csv_files.py
│   ├── dict_to_dataframe.py
│   ├── extract_info.py
│   ├── extract_text_from_pdf_v1.py
│   ├── extract_text_from_pdf.py
│   ├── get_imdb_reviews_from_kaggle.py
│   ├── get_subtitles_from_kaggle.py
│   └── Roberta.py
└── visualization/                  # Data visualization scripts and outputs (contains .png and .py files)
    ├── combined_cosine_similarity_boxplots.png
    ├── combined_polarity_score_comparison_Screenplays.png
    ├── combined_polarity_score_comparison_Subtitle.png
    ├── combined_polarity_score_comparison.png
    └── cosine_similarity_boxplot.py 
    # (other .py and .png files)
```

## Key Scripts

- **`utils/Roberta.py`**: Performs sentiment and emotion analysis on reviews.
- **`Cosine Similarity/pairwise_cosine.py`**: Calculates semantic similarity between reviews.
- **`LLM Generated Reviews/generate_ai_reviews.py`**: Generates reviews using different LLM APIs.
- **`Trigram Analysis/trigram_analysis.py`**: Generates top trigrams for AI and IMDb reviews.
- **`utils/extract_text_from_pdf.py`**: Extracts text from PDF screenplay files.
- **`utils/get_imdb_reviews_from_kaggle.py`**: Downloads IMDb reviews from Kaggle.
- **`utils/get_subtitles_from_kaggle.py`**: Downloads subtitles from Kaggle.

## Setup and Installation

### Prerequisites
- API keys for: OpenAI, Google Gemini, DeepSeek, OMDB (Note: OMDB might be used by one of the utility scripts, ensure it's available if needed).
- Kaggle API token (`kaggle.json`) configured for downloading datasets.

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/LLM-moviereviews.git
   cd LLM-moviereviews
   ```

2. Install required dependencies:
   ```bash
   pip install pandas numpy scikit-learn torch transformers openai google-generativeai kaggle python-dotenv
   ```
   (Note: `kagglehub` was in the original list, `kaggle` is usually the one for the API. If `kagglehub` is specifically used, please add it back.)

3. Create a `.env` file in the root directory with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   GEMINI_API_KEY=your_gemini_api_key
   DEEPSEEK_API_KEY=your_deepseek_api_key
   # OMDB_API_KEY=your_omdb_api_key (if used)
   ```
4. Place your `kaggle.json` file in the appropriate location (e.g., `~/.kaggle/kaggle.json` or `C:\\Users\\<YourUsername>\\.kaggle\\kaggle.json`) for the Kaggle API to work.

## Usage

Ensure your `.env` file is correctly set up and `kaggle.json` is in place before running scripts that require them.

### Downloading Data (if not already present)
```bash
python utils/get_imdb_reviews_from_kaggle.py
python utils/get_subtitles_from_kaggle.py
```

### Generating AI Reviews

To generate movie reviews using different LLM models:
```bash
python "LLM Generated Reviews/generate_ai_reviews.py"
```

### Running Sentiment and Emotion Analysis

To analyze sentiment and emotions in reviews (ensure `all_imdb_reviews.csv` and `selected_movie_info.csv` are present):
```bash
python utils/Roberta.py
```

### Calculating Similarity

To calculate cosine similarity between different review types:
```bash
python "Cosine Similarity/pairwise_cosine.py"
```

### Running Trigram Analysis
```bash
python "Trigram Analysis/trigram_analysis.py"
```

## Results

Results from various analyses are stored in dedicated directories:
- Sentiment analysis: `Polarity Analysis/`
- Emotion analysis: `Emotion Analysis/`
- Semantic similarity: `Cosine Similarity/`
- Trigram analysis: `Trigram Analysis/`

## License

This project is intended for research purposes. Please cite this repository if using the code or results in academic work.
