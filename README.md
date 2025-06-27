# Système de Recommandation de Films - KNOK
# 📌 Description du Projet
KNOK est un système intelligent de recommandation de films basé sur des techniques avancées d'apprentissage automatique. Ce projet utilise le filtrage collaboratif et la factorisation de matrices (SVD++) pour prédire les préférences des utilisateurs et leur suggérer des films pertinents.

# 🎯 Fonctionnalités Principales
Recommandations personnalisées basées sur l'historique des notations

Interface interactive développée avec Streamlit

Recherche intelligente tolérante aux fautes de frappe

Affichage dynamique des posters et informations des films via l'API TMDB

Optimisation des performances avec FAISS pour une recherche rapide par genre

# 🛠️ Technologies Utilisées
Python (Pandas, NumPy, Scikit-learn)

Machine Learning (Surprise, XGBoost)

Traitement des données (TF-IDF, Similarité cosinus)

Interface Web (Streamlit)

Base de données (MovieLens 20M)

API (The Movie Database)

# 📊 Modèles Implémentés
BaselineOnly (Moyennes globales)

KNN-Baseline (Similarité utilisateur/film)

SlopeOne (Différences de notation)

SVD++ (Factorisation matricielle avancée)

XGBoost (Approche hybride)

