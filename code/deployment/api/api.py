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
models = ["miniBatchKMeans_model","gmm_model","kmeans_model"]
actual_model_type = models[0]
actual_scaler = "StandardScaler"
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
def load_model(model_type: str, scaler: str):
    model_dir = os.path.join(os.getcwd(), "models")
    model_path = os.path.join(model_dir, f"{model_type}.pkl")
    scaler_path = os.path.join(model_dir, f"{scaler}.pkl")
    training_data_path = os.path.join(os.getcwd(), "code","models", f"{model_type}",f"{model_type}_data.csv")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise HTTPException(status_code=404, detail="Model or scaler not found.")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    training_data = pd.read_csv(training_data_path)
    return model, scaler, training_data

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
def recommend_songs(user_song, train_data, model, metric, num_recommendations=5):
    # Use the KMeans model to predict the cluster for the user's song
    user_cluster = model.predict(user_song.reshape(1, -1))
    
    # Filter the training data to only include songs from the same cluster
    cluster_songs = train_data[train_data['cluster'] == user_cluster[0]]

    # Now calculate similarity only within the cluster
    similarity_scores = calculate_similarity(user_song, cluster_songs[features].values, metric)
    
    # Sort indices of similarity scores in descending order
    recommended_indices = similarity_scores.argsort()[0][-num_recommendations:][::-1]
    
    # Return the recommended songs
    return cluster_songs.iloc[recommended_indices][['track_name', 'artists']]

# Adjust the FastAPI endpoint
@app.post("/recommend")
def get_recommendations(preferences: List[UserPreferences], model_type: str = actual_model_type, scaler: str = actual_scaler, metric: str = actual_metric, num_recommendations: int = 5):
    user_data = pd.DataFrame([pref.dict() for pref in preferences])

    # Load the KMeans model and scaler
    model, scaler, training_data = load_model(model_type, scaler)

    # Scale the user data
    user_X = scaler.transform(user_data[features])

    # Generate recommendations using the KMeans cluster
    recommendations = recommend_songs(user_X[0], training_data, model, metric, num_recommendations)

    return {"recommendations": recommendations.to_dict(orient="records")}
