import streamlit as st
import pandas as pd
import numpy as np
import pickle
# from dotenv import load_dotenv
import os
import requests
from sklearn.metrics.pairwise import cosine_similarity


# Load env file
# load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

if not TMDB_API_KEY:
    st.error("TMDB_API_KEY not found. Make sure it is set in the .env file.")
    st.stop()


import requests

url = "https://api.themoviedb.org/3/authentication"

headers = {
    "accept": "application/json",
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI4ZTJmMjRhZWJmNmFjZTE5MzdhZTQ3MTc5MDE1MWQyMSIsIm5iZiI6MTc0NzA3NDA2OC4xODcsInN1YiI6IjY4MjIzYzE0YmE1ZTk4YmJhZjczOTMyMCIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.NcDGuf_9hf2fnsPNrQNnZGE5_ThVWJS8Q9qTSHSBp-E"
}

response = requests.get(url, headers=headers)

print(response.text)

# TMDB API config
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"




# Set page config
st.set_page_config(
    page_title="Movie Recommendation System App",
    page_icon="",
    layout="wide"
)

# # Function to load models
# @st.cache_resource
# def load_models():
#     with open("./Saved Models/svd_modelv2.pkl", "rb") as f:
#         svd_model = pickle.load(f)
#         return svd_model

# Function to load data
@st.cache_data
def load_movie_data():
    df = pd.read_csv("./Datasets/ml-32m/movies.csv")
    df['genres'] = df['genres'].apply(lambda x: x.split('|'))
    return df

# @st.cache_data
# def load_ratings_data():
#     df = pd.read_csv("./Datasets/ml-32m/ratings.csv")
#     return df

@st.cache_data
def load_Q_matrix():
    return np.load("./Saved Models/Q_matrix.npy")

@st.cache_data
def load_movie_mapper():
    with open("./Saved Models/movie_mapper.pkl", "rb") as f:
        movie_mapper = pickle.load(f)
        return movie_mapper
    
@st.cache_data
def load_movie_inv_mapper():
    with open("./Saved Models/movie_inv_mapper.pkl", "rb") as f:
        movie_inv_mapper = pickle.load(f)
        return movie_inv_mapper

@st.cache_data
def load_user_inv_mapper():
    with open("./Saved Models/user_inv_mapper.pkl", "rb") as f:
        user_inv_mapper = pickle.load(f)
        return user_inv_mapper
    
@st.cache_data
def load_movie_stats():
    df = pd.read_csv("./Datasets/ml-32m/movie_stats.csv")
    df = df.drop(columns=['count', 'mean', 'title'])
    return df

@st.cache_data
def load_links():
    df = pd.read_csv("./Datasets/ml-32m/links.csv")
    return df

try:
    # Try to load the resource
    # svd_model = load_models()

    # Try to load the data
    movie_data = load_movie_data()
    movie_stats = load_movie_stats()
    links = load_links()
    movie_mapper = load_movie_mapper()
    movie_inv_mapper = load_movie_inv_mapper()
    user_inv_mapper = load_user_inv_mapper()
    Q_matrix = load_Q_matrix()

    resource_loaded = True
except Exception as e:
    st.error(f"Error loading resource: {e}")
    resource_loaded = False


# Preparing necessary data
TOP_N_MOVIES = 10
all_genres = sorted(set(g for genre_list in movie_data['genres'] for g in genre_list))
movie_data = movie_data.merge(links[['movieId', 'tmdbId']], on='movieId', how='left')
movies_with_rating = movie_data.merge(movie_stats, on='movieId')
title_to_id = dict(zip(movie_data['title'], movie_data['movieId']))
id_to_title = dict(zip(movie_data['movieId'], movie_data['title']))



# --- TMDB Data Fetcher ---
# --- TMDB Fetcher using TMDB ID ---
@st.cache_data(show_spinner=False)
def fetch_tmdb_data_by_id(tmdb_id):
    if np.isnan(tmdb_id):
        return None
    try:
        url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}"
        params = {"api_key": TMDB_API_KEY}
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException as e:
        st.warning(f"TMDB fetch error: {e}")
    return None

# --- Movie Poster Fetcher ---
@st.cache_data(show_spinner=False)
def fetch_poster_df(movies_df):
    posters = []
    for _, row in movies_df.iterrows():
        tmdb = fetch_tmdb_data_by_id(row['tmdbId'])
        if tmdb and tmdb.get("poster_path"):
            posters.append({
                "title": row['title'],
                "tmdbId": row['tmdbId'],
                "poster_url": TMDB_IMAGE_BASE + tmdb["poster_path"]
            })
    return pd.DataFrame(posters)



# --- Poster Display with Checkboxes ---
def display_movie_selection(movie_df, max_cols=3):
    selected = []
    rows = len(movie_df) // max_cols + 1
    for row_idx in range(rows):
        cols = st.columns(max_cols)
        for i in range(max_cols):
            idx = row_idx * max_cols + i
            if idx >= len(movie_df):
                break
            movie = movie_df.iloc[idx]
            with cols[i]:
                st.image(movie["poster_url"], width=150)
                st.caption(movie["title"])
                if st.checkbox(f"{movie['title']}", key=f"movie_{idx}"):
                    selected.append(movie["title"])
    return selected



# Show the movie poster based on movies selected by the user
def show_selected_movies(movie_titles, all_poster_df, max_cols=4):
    st.markdown("### 🎯 Your Selected Movies:")
    selected_df = all_poster_df[all_poster_df['title'].isin(movie_titles)]

    rows = len(selected_df) // max_cols + 1
    for row_idx in range(rows):
        cols = st.columns(max_cols)
        for i in range(max_cols):
            idx = row_idx * max_cols + i
            if idx >= len(selected_df):
                break
            row = selected_df.iloc[idx]
            with cols[i]:
                st.image(row['poster_url'], width=120)
                st.caption(row['title'])



@st.cache_data(show_spinner=False)
def fetch_popular_movie():
    url = "https://api.themoviedb.org/3/movie/popular"
    params = {"api_key": TMDB_API_KEY, "language": "en-US", "page": 1}
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            if results:
                return results[0]  # return the most popular one
    except requests.exceptions.RequestException as e:
        st.warning(f"Failed to fetch popular movie: {e}")
    return None




# Main Title
st.title("🎬 Movie Recommendation App")

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("select Page:", ['Home','Movie Explorer', 'Movie Recommendation System'])

# Home Page
if page == 'Home':

    col1, col2 = st.columns(2)

    with col1:
        st.write("""
        ### About the App
        This app recommends movies to the user 
        based on genres they like and movies they like.
                 
        - Movie Genres
        - Selected Movies user likes
                 
        This app uses Hybrid Recommendation System
        - Cold Start using Bayesian Average + Genre
        - Uses SVD to create matrix factorization
        """)


    with col2:
        st.markdown("### 🔥 Currently Popular Movie")

        pop_movie = fetch_popular_movie()
        if pop_movie:
            poster_url = TMDB_IMAGE_BASE + pop_movie.get("poster_path", "")
            title = pop_movie.get("title", "Untitled")
            overview = pop_movie.get("overview", "No description available.")
            
            st.image(poster_url, width=200)
            st.subheader(title)
            st.write(overview)
        else:
            st.info("Unable to fetch popular movie right now.")
    

# Search page
elif page == 'Movie Explorer':
    st.write("Things to do!")
    st.markdown("### 🔧 TMDB API Connectivity Test")

    # def test_tmdb_connection():
    #     url = "https://api.themoviedb.org/3/movie/550"  # Fight Club
    #     params = {"api_key": TMDB_API_KEY}
    #     try:
    #         response = requests.get(url, params=params, timeout=5)
    #         st.write("Status Code:", response.status_code)
    #         if response.status_code == 200:
    #             data = response.json()
    #             st.success("TMDB API is working!")
    #             st.json(data)  # show full JSON
    #         else:
    #             st.error(f"TMDB returned status code {response.status_code}")
    #     except Exception as e:
    #         st.error(f"Error connecting to TMDB: {e}")

    # test_tmdb_connection()

# Prediction Page
elif page == 'Movie Recommendation System':

    st.write("""
    ### Search Page
    This app recommends the user based on movie genres they like.
    """)
        
    # Step 1 Genre Selection
    selected_genres = st.multiselect("What genres do you enjoy?", all_genres, default=['Action', 'Comedy'])

    # Step 2 Favorite movies
    if selected_genres:
        filtered_movies = movies_with_rating[movies_with_rating['genres'].apply(lambda gs: all(g in gs for g in selected_genres))]

        filtered_movies = filtered_movies[filtered_movies['movieId'].isin(set(movie_mapper.keys()))]


        top_movies = filtered_movies.sort_values(by='bayesian_average', ascending=False).head(TOP_N_MOVIES)

        poster_df = fetch_poster_df(top_movies)
            
        st.markdown("### Pick a few movies you love:")
        selected_movies = display_movie_selection(poster_df)

        # show which movies the user chooses
        if selected_movies:
            show_selected_movies(selected_movies, poster_df)


        if selected_movies:
            selected_ids = [title_to_id[t] for t in selected_movies if title_to_id.get(t) in movie_mapper]
            if not selected_ids:
                st.warning("None of the selected movies are available in the model.")
            else:
                user_vecs = [Q_matrix[movie_mapper[mid]] for mid in selected_ids]
                avg_vec = np.mean(user_vecs, axis=0).reshape(1, -1)


                sims = cosine_similarity(avg_vec, Q_matrix).flatten()
                sorted_idx = sims.argsort()[::-1]


                recommendations = []
                for i in sorted_idx:
                    mid = movie_inv_mapper[i]
                    title = id_to_title[mid]
                    if title not in selected_movies and title not in recommendations:
                        recommendations.append(title)
                    if len(recommendations) >= 6:
                        break

                st.markdown("## 🍿 Because you loved those movies, try these:")

                # Create a local dict to store already-fetched TMDB data
                tmdb_cache = {}

                for rec_title in recommendations:
                    row = movies_with_rating[movies_with_rating['title'] == rec_title].dropna(subset=["tmdbId"])
                    if row.empty:
                        continue

                    tmdb_id = int(row.iloc[0]['tmdbId'])

                    # Check if we've already fetched this movie
                    if tmdb_id not in tmdb_cache:
                        tmdb_cache[tmdb_id] = fetch_tmdb_data_by_id(tmdb_id)

                    tmdb = tmdb_cache[tmdb_id]

                    if tmdb and tmdb.get("poster_path"):
                        st.image(TMDB_IMAGE_BASE + tmdb["poster_path"], width=150)
                    st.subheader(tmdb["title"] if tmdb else rec_title)
                    st.write(tmdb.get("overview", "No description available.") if tmdb else "")

