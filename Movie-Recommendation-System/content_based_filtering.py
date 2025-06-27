import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Charger les données et calculer une seule fois
movie_data = pd.read_csv("movies.csv")  # adapte selon ton fichier
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movie_data['description'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Mapping titre -> index
title_to_index = pd.Series(movie_data.index, index=movie_data['title'])

def get_recommendations(title, num_recommendations=5):
    idx = title_to_index.get(title)
    if idx is None:
        return []

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores[1:num_recommendations+1]]
    return movie_data['title'].iloc[movie_indices].tolist()
