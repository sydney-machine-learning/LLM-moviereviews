import requests
import pandas as pd


API_KEY = "9a116dc6"

# Read movie titles from the CSV files
oscar_movies_df = pd.read_csv('oscar_movies_and_years.csv', header=None, names=['movie', 'year'])
blockbuster_movies_df = pd.read_csv('blockbuster_movies_and_years.csv', header=None, names=['movie', 'year'])

# Combine them into a single list
movie_titles = oscar_movies_df['movie'].tolist() + blockbuster_movies_df['movie'].tolist()

# Function to get IMDb ID from movie title
def get_imdb_id(movie_title):
    url = f"http://www.omdbapi.com/?t={movie_title}&apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    
    if response.status_code == 200 and data["Response"] == "True":
        return data["imdbID"]
    else:
        return f"Error: {data.get('Error', 'Unknown error')}"

# Get the IMDb IDs for all movie titles
movie_imdb_ids = {title: get_imdb_id(title) for title in movie_titles}

# Save results to a CSV file
imdb_ids_df = pd.DataFrame(list(movie_imdb_ids.items()), columns=['movie', 'imdb_id'])
imdb_ids_df.to_csv('movies_with_imdb_ids.csv', index=False,header=False)
