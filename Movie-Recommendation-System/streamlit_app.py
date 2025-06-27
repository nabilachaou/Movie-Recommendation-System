import streamlit as st
import pandas as pd
import requests
import random
import base64
import pickle
from surprise import SVDpp
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------- Configuration Streamlit --------------------
st.set_page_config(page_title="KNOK", layout="wide")

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('back.png')

TMDB_API_KEY = '937a082028125dd7eabb7bcc3cf3f0fa'

# -------------------- Chargement des données --------------------
@st.cache_data(show_spinner=False)
def load_data():
    movies = pd.read_csv('movies.csv')
    links = pd.read_csv('links.csv')
    ratings = pd.read_csv('ratings.csv')  # Charger les ratings aussi

    df = movies.merge(links[['movieId', 'tmdbId']], on='movieId', how='left')
    df['genres'] = df['genres'].fillna('')

    # Ajouter la moyenne des ratings
    movie_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
    movie_ratings.columns = ['movieId', 'mean_rating']
    df = df.merge(movie_ratings, on='movieId', how='left')
    df['mean_rating'] = df['mean_rating'].round(1)  # arrondir pour joli affichage

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['genres'])

    faiss_index = faiss.IndexFlatL2(tfidf_matrix.shape[1])
    faiss_index.add(tfidf_matrix.astype('float32').toarray())

    title_to_idx = {title: idx for idx, title in enumerate(df['title'])}

    return df, faiss_index, title_to_idx

movies, faiss_index, title_to_index = load_data()

# -------------------- Charger le modèle SVDpp --------------------
@st.cache_resource(show_spinner=False)
def load_svdpp_model():
    with open('svdpp_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

svdpp_model = load_svdpp_model()

# -------------------- Fonctions utilitaires --------------------
@st.cache_data(show_spinner=False)
def fetch_poster(tmdb_id):
    if pd.isna(tmdb_id):
        return "https://placekitten.com/200/300"  # Image par défaut
    try:
        url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={TMDB_API_KEY}"
        r = requests.get(url, timeout=3)
        r.raise_for_status()
        data = r.json()
        path = data.get('poster_path')
        if path:
            return f"https://image.tmdb.org/t/p/w500{path}"
        return "https://placekitten.com/200/300"
    except Exception as e:
        print(f"Error fetching poster: {e}")
        return "https://placekitten.com/200/300"

# -------------------- Recommandations basées sur SVDpp et Genre --------------------
def get_recommendations_by_genre_and_svdpp(title, n=20):
    movie_row = movies[movies['title'].str.lower().str.contains(title.lower(), na=False)]
    if movie_row.empty:
        return pd.DataFrame(), None

    movie_row = movie_row.iloc[0]  # Prendre le premier si plusieurs résultats
    movie_id = movie_row['movieId']
    genres = movie_row['genres'].split('|')
    
    genre_filtered_movies = movies[movies['genres'].str.contains('|'.join(genres), case=False)]

    user_id = 1

    all_movie_ids = genre_filtered_movies['movieId'].tolist()
    predictions = []

    for m_id in all_movie_ids:
        if m_id != movie_id:
            pred = svdpp_model.predict(user_id, m_id).est
            predictions.append((m_id, pred))

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_movie_ids = [m_id for m_id, _ in predictions[:n]]

    recommended_movies = movies[movies['movieId'].isin(top_movie_ids)]
    return recommended_movies, movie_row

# -------------------- CSS personnalisé --------------------
st.markdown("""
    <style>
    .main-title {
        font-weight: bold;
        font-size: 64px;
        color: white;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px #000;
    }
    .movie-title {
        color: white;
        font-weight: bold;
        font-size: 20px;
        margin-top: 10px;
        text-align: center;
        text-shadow: 1px 1px 2px #000;
    }
    .movie-info {
        color: white;
        font-size: 14px;
        text-align: center;
        text-shadow: 1px 1px 2px #000;
    }
    .reco-title {
        color: white;
        font-size: 28px;
        margin-bottom: 20px;
        text-align: center;
        text-shadow: 2px 2px 4px #000;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- Interface utilisateur --------------------
st.image("logo.jpeg", width=120)
st.markdown('<div class="main-title">Welcome to KNOK</div>', unsafe_allow_html=True)

with st.form("search_form", clear_on_submit=False):
    search_col1, search_col2 = st.columns([4, 1])
    with search_col1:
        movie_title = st.text_input("", placeholder="Type a movie name...", label_visibility="collapsed")
    with search_col2:
        submit = st.form_submit_button("🔍")
    if submit and movie_title:
        st.session_state["last_search"] = movie_title

# -------------------- Afficher résultats --------------------
if "last_search" not in st.session_state:
    st.markdown('<h3 class="reco-title">🎬 Explore Some Movies:</h3>', unsafe_allow_html=True)

    sample_movies = movies.sample(16, random_state=random.randint(1, 1000))  # Changer à chaque chargement

    for i in range(0, len(sample_movies), 4):
        cols = st.columns(4)
        for j, (_, row) in enumerate(sample_movies.iloc[i:i+4].iterrows()):
            with cols[j]:
                poster = fetch_poster(row['tmdbId'])
                st.image(poster, use_container_width=True)
                st.markdown(f'<div class="movie-title">{row["title"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="movie-info">🗂️ {row["genres"]}<br>⭐ {row["mean_rating"] if not pd.isna(row["mean_rating"]) else "N/A"}</div>', unsafe_allow_html=True)

else:
    search_term = st.session_state["last_search"]
    recs, searched_movie = get_recommendations_by_genre_and_svdpp(search_term, n=20)
    if searched_movie is None:
        st.warning("Movie not found.")
    else:
        st.markdown('<h3 class="reco-title">🎯 Recommended Movies:</h3>', unsafe_allow_html=True)

        # Afficher le film recherché d'abord
        cols = st.columns(4)
        with cols[0]:
            poster = fetch_poster(searched_movie['tmdbId'])
            st.image(poster, use_container_width=True)
            st.markdown(f'<div class="movie-title">{searched_movie["title"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="movie-info">🗂️ {searched_movie["genres"]}<br>⭐ {searched_movie["mean_rating"] if not pd.isna(searched_movie["mean_rating"]) else "N/A"}</div>', unsafe_allow_html=True)

        # Puis afficher les recommandations
        for i in range(0, len(recs), 4):
            cols = st.columns(4)
            for j, (_, row) in enumerate(recs.iloc[i:i+4].iterrows()):
                with cols[j]:
                    poster = fetch_poster(row['tmdbId'])
                    st.image(poster, use_container_width=True)
                    st.markdown(f'<div class="movie-title">{row["title"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="movie-info">🗂️ {row["genres"]}<br>⭐ {row["mean_rating"] if not pd.isna(row["mean_rating"]) else "N/A"}</div>', unsafe_allow_html=True)
