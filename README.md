# LLM-MovieReviews: Evaluating AI-Generated Movie Reviews

This repository contains the code and output results for Group 8's research project on evaluating Large Language Models (LLMs) for generating movie reviews. The project compares AI-generated reviews from various models (gpt4o, Gemini 2.0, DeepSeek V3) with human reviews from the IMDb database using multiple analysis methods.

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
├── .gitignore
├── Analysis Results/               # All analysis output results
│   ├── Cosine Similarity/
│   │   ├── cosine_similarity_results_screenplays_by_movie.csv
│   │   ├── cosine_similarity_results_screenplays_by_question.csv
│   │   ├── cosine_similarity_results_subtitles_by_movie.csv
│   │   ├── cosine_similarity_results_subtitles_by_question.csv
│   │   ├── README.md
│   │   └── within_imdb_similarity_results.csv
│   ├── Emotion Analysis/
│   │   ├── average_emotion_scores_imdb.csv
│   │   ├── average_emotion_scores_screenplays.csv
│   │   ├── average_emotion_scores_subtitles.csv
│   │   ├── Emotion_Score_Comparison__Percentage_Screenplays.csv
│   │   └── Emotion_Score_Comparison__Percentage_Subtitles.csv
│   ├── Polarity Analysis/
│   │   ├── average_polarity_scores_imdb_between_6_and_7.csv
│   │   ├── average_polarity_scores_imdb_rated_above7.csv
│   │   ├── average_polarity_scores_imdb_rated_below6.csv
│   │   ├── average_polarity_scores_imdb.csv
│   │   ├── average_polarity_scores_screenplays.csv
│   │   └── average_polarity_scores_subtitles.csv
│   ├── Questionaire/
│   │   └── questionnaire_review.csv
│   ├── Trigram Analysis/
│   │   ├── aireviews_chatgpt_screenplays_trigrams.csv
│   │   ├── aireviews_chatgpt_trigrams.csv
│   │   ├── aireviews_deepseek_screenplays_trigrams.csv
│   │   ├── aireviews_deepseek_trigrams.csv
│   │   ├── aireviews_gemini_screenplays_trigrams.csv
│   │   ├── aireviews_gemini_screenplays_v2_trigrams.csv
│   │   ├── aireviews_gemini_trigrams.csv
│   │   ├── aireviews_gemini_v2_trigrams.csv
│   │   ├── all_imdb_review_trigrams.csv
│   │   └── shawshank_imdb_10_trigrams.csv
│   └── Visualizations/             # Data visualization outputs
│       ├── combined_cosine_similarity_boxplots.png
│       ├── combined_polarity_score_comparison_Screenplays.png
│       ├── combined_polarity_score_comparison_Subtitle.png
│       ├── combined_polarity_score_comparison.png
│       └── cosine_similarity_boxplot.py
├── Data/                           # Cleaned and processed data files
│   ├── cleaned_screenplays.csv
│   ├── cleaned_subtitles.csv
│   └── screenplay files/           # Movie screenplay source PDF files
├── Reviews/                        # Review data (human and AI-generated)
│   ├── IMDb Reviews/               # IMDb review data
│   └── LLM Generated Reviews/      # AI-generated review data
│       ├── screenplays/            # Reviews based on movie screenplays
│       └── subtitles/              # Reviews based on movie subtitles
└── Utils1/                         # Main Python scripts for analysis
    ├── generate_ai_reviews.py      # ★ Main script for generating AI reviews
    ├── pairwise_cosine.py          # ★ Cosine similarity analysis
    ├── Roberta.py                  # ★ Sentiment and emotion analysis
    ├── trigram_analysis.py         # ★ Trigram analysis
    ├── consolidate_csv_files.py    # Utility for CSV consolidation
    ├── dict_to_dataframe.py        # Utility for data conversion
    ├── extract_info.py             # Data extraction utility
    ├── extract_text_from_pdf.py    # PDF text extraction
    ├── extract_text_from_pdf_v1.py # Alternative PDF extraction
    ├── get_imdb_reviews_from_kaggle.py  # Downloads IMDb reviews
    ├── get_subtitles_from_kaggle.py     # Downloads subtitles
    ├── questionnaire.py            # Questionnaire processing
    ├── selected_movie_info.csv     # Movie metadata
    └── titles with awards and categories.txt  # Movie categories
```

## Key Scripts

### Core Analysis Scripts (all located in `Utils1/`)

- **`Utils1/generate_ai_reviews.py`**: ★ Main script for generating movie reviews using different LLM APIs (GPT-4o, Gemini 2.0, DeepSeek V3).
- **`Utils1/Roberta.py`**: ★ Performs sentiment and emotion analysis on reviews using RoBERTa models.
- **`Utils1/pairwise_cosine.py`**: ★ Calculates semantic similarity between AI-generated and human reviews using cosine similarity.
- **`Utils1/trigram_analysis.py`**: ★ Generates and analyzes trigrams for AI and IMDb reviews.


## Setup and Installation

### Prerequisites
- API keys for: OpenAI, Google Gemini, DeepSeek, OMDB.
- Kaggle API token (`kaggle.json`) configured for downloading datasets.


Create a `.env` file in the root directory with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   GEMINI_API_KEY=your_gemini_api_key
   DEEPSEEK_API_KEY=your_deepseek_api_key
   # OMDB_API_KEY=your_omdb_api_key (if used)
   ```

## Usage

### Downloading Data (if not already present)
```powershell
python Utils1/get_imdb_reviews_from_kaggle.py
python Utils1/get_subtitles_from_kaggle.py
```

### Generating AI Reviews

To generate movie reviews using different LLM models:
```powershell
python Utils1/generate_ai_reviews.py
```

### Running Sentiment and Emotion Analysis

To analyze sentiment and emotions in reviews:
```powershell
python Utils1/Roberta.py
```

### Calculating Similarity

To calculate cosine similarity between different review types:
```powershell
python Utils1/pairwise_cosine.py
```

### Running Trigram Analysis

To analyze trigrams in AI and human reviews:
```powershell
python Utils1/trigram_analysis.py
```


## Results

Results from various analyses are stored in the `Analysis Results/` directory:
- **Sentiment analysis**: `Analysis Results/Polarity Analysis/`
- **Emotion analysis**: `Analysis Results/Emotion Analysis/`
- **Semantic similarity**: `Analysis Results/Cosine Similarity/`
- **Trigram analysis**: `Analysis Results/Trigram Analysis/`
- **Visualizations**: `Analysis Results/Visualizations/`

## License

This project is intended for research purposes. Please cite this repository if using the code or results in academic work.
