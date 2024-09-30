from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from sklearn.model_selection import train_test_split
from typing import List
import time
app = FastAPI()

path_components = os.path.abspath(__file__).split(os.sep)
home_path = os.sep.join(path_components[:path_components.index('spot-out') + 1])
df_cleaned = pd.read_csv(os.path.join(home_path, "data", "spotify_cleaned.csv"))
scaler = joblib.load(os.path.join(home_path, "models", "model.pkl"))

features = [
    'danceability', 'energy', 'key', 'loudness', 'speechiness', 
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

# Drop rows with missing values
df_cleaned = df_cleaned.dropna(subset=features)
train_data, test_data = train_test_split(df_cleaned, test_size=0.00001, random_state=42)

# Use the loaded scaler to transform the training data
scaler.fit(train_data[features])
X_train = scaler.transform(train_data[features])


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

# Define helper function for recommendations
def recommend_songs(similarity_matrix, train_data, num_recommendations=5):
    recommendations = []
    for i in range(similarity_matrix.shape[0]):
        # Get similarity scores for the i-th user's uploaded song
        sim_scores = list(enumerate(similarity_matrix[i]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the top num_recommendations recommendations (ignoring the song itself)
        recommended_songs_indices = [sim_scores[j][0] for j in range(1, num_recommendations+1)]
        recommended_songs = train_data.iloc[recommended_songs_indices]

        recommendations.append(recommended_songs[['track_name', 'artists']])

    return recommendations

# FastAPI endpoint to receive user preferences and return recommendations
@app.post("/recommend")
def get_recommendations(preferences: List[UserPreferences]):
    # Convert preferences to a dataframe
    user_data = pd.DataFrame([pref.dict() for pref in preferences])
    # Scale the user data
    user_X = scaler.transform(user_data[features])
    
    # Calculate cosine similarity between user data and training data
    similarity_matrix = cosine_similarity(user_X, X_train)
    
    # Generate recommendations
    recommendations = recommend_songs(similarity_matrix, train_data, num_recommendations=5)
    
    # Return recommendations
    return {"recommendations": recommendations}
