# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
import joblib
from scipy.spatial.distance import (euclidean, cityblock, jaccard, 
                                    correlation, hamming, mahalanobis, 
                                    chebyshev, minkowski, braycurtis)
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List

app = FastAPI()
actual_model_type = "kmeans_model"
actual_metric = "cosine"
features = [
    'danceability', 'energy', 'key', 'loudness', 'speechiness', 
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

# Define Pydantic model for user preferences
class UserPreferences(BaseModel):
    danceability: float
    energy: float
    key: int
    loudness: float
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float

# Load model based on model type
def load_model(model_type: str):
    model_dir = os.path.join(os.getcwd(), "models")
    model_path = os.path.join(model_dir, f"{model_type}.pkl")
    scaler_path = os.path.join(model_dir, "StandardScaler.pkl")
    training_data_path = os.path.join(os.getcwd(), "data", f"{model_type}_data.csv")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise HTTPException(status_code=404, detail="Model or scaler not found.")
    
    kmeans = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    training_data = pd.read_csv(training_data_path)
    return kmeans, scaler, training_data

# Similarity functions and recommendations
def calculate_similarity(user_song, cluster_songs, metric):
    user_song = user_song.reshape(1, -1)  # Ensure 2D array shape
    if metric == "cosine":
        return cosine_similarity(user_song, cluster_songs)
    elif metric == "euclidean":
        return np.array([euclidean(user_song[0], song) for song in cluster_songs])
    elif metric == "manhattan":
        return np.array([cityblock(user_song[0], song) for song in cluster_songs])
    elif metric == "jaccard":
        return np.array([jaccard(user_song[0], song) for song in cluster_songs])
    elif metric == "pearson":
        return np.array([1 - correlation(user_song[0], song) for song in cluster_songs])  # Convert to similarity
    elif metric == "hamming":
        return np.array([hamming(user_song[0], song) for song in cluster_songs])
    elif metric == "mahalanobis":
        VI = np.linalg.inv(np.cov(cluster_songs.T))
        return np.array([mahalanobis(user_song[0], song, VI) for song in cluster_songs])
    elif metric == "chebyshev":
        return np.array([chebyshev(user_song[0], song) for song in cluster_songs])
    elif metric == "minkowski":
        return np.array([minkowski(user_song[0], song) for song in cluster_songs])
    elif metric == "braycurtis":
        return np.array([braycurtis(user_song[0], song) for song in cluster_songs])
    else:
        raise ValueError("Unknown metric")
    
# Define helper function for recommendations
def recommend_songs(user_song, train_data, metric, num_recommendations=5):
    similarity_scores = calculate_similarity(user_song, train_data[features].values, metric)
    
    # Sort indices of similarity scores in descending order
    recommended_indices = similarity_scores.argsort()[0][-num_recommendations:][::-1]
    return train_data.iloc[recommended_indices][['track_name', 'artists']]

# FastAPI endpoint to receive user preferences and return recommendations
@app.post("/recommend")
def get_recommendations(preferences: List[UserPreferences], model_type: str = actual_model_type, metric: str = actual_metric, num_recommendations: int = 5):
    user_data = pd.DataFrame([pref.dict() for pref in preferences])

    # Load the KMeans model and scaler
    kmeans, scaler, training_data = load_model(model_type)

    # Scale the user data
    user_X = scaler.transform(user_data[features])

    # Generate recommendations
    recommendations = recommend_songs(user_X[0], training_data, metric, num_recommendations)

    return {"recommendations": recommendations.to_dict(orient="records")}
