import pandas as pd

# Read the cleaned_subtitles.csv file
df = pd.read_csv('cleaned_subtitles.csv')

# Select the relevant columns
selected_columns = df[['movie', 'imdb_id', 'year', 'award']]

# Save the selected columns to a new CSV file
selected_columns.to_csv('selected_movie_info.csv', index=False)