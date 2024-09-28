import streamlit as st
import pandas as pd
import requests

# Streamlit app
st.title("Music Recommendation App")
st.write("Upload a CSV file containing your song preferences or input your preferences manually.")

# Function to send user preferences to FastAPI and get recommendations
def fetch_recommendations(preferences):
    api_url = "http://fastapi:8000/recommend"   
    response = requests.post(api_url, json=preferences)
    
    if response.status_code == 200:
        return response.json()["recommendations"]
    else:
        st.error(f"Error fetching recommendations: {response.status_code} - {response.text}")
        return []

# File Uploader for CSV
uploaded_file = st.file_uploader("Choose a CSV file", type='csv')

if uploaded_file is not None:
    # Read uploaded CSV
    user_uploaded_data = pd.read_csv(uploaded_file)
    
    st.write(f"User uploaded data size: {len(user_uploaded_data)}")
    st.write("User Uploaded Data:")
    st.write(user_uploaded_data[['track_name', 'artists']])

    # Check that the required columns exist
    required_features = [
        'danceability', 'energy', 'key', 'loudness', 'speechiness', 
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
    ]
    
    if all(feature in user_uploaded_data.columns for feature in required_features):
        # Get the first song's features as an example to send to the API
        first_song = user_uploaded_data.iloc[0]
        preferences = {
            'danceability': first_song['danceability'],
            'energy': first_song['energy'],
            'key': int(first_song['key']),
            'loudness': first_song['loudness'],
            'speechiness': first_song['speechiness'],
            'acousticness': first_song['acousticness'],
            'instrumentalness': first_song['instrumentalness'],
            'liveness': first_song['liveness'],
            'valence': first_song['valence'],
            'tempo': first_song['tempo']
        }

        # Fetch recommendations from the FastAPI backend
        recommendations = fetch_recommendations(preferences)

        # Show recommendations
        st.write("Recommendations for the first uploaded song:")
        st.write(recommendations)
    else:
        st.error("Uploaded file is missing required features.")
else:
    st.write("Please upload a CSV file with your listened songs.")
