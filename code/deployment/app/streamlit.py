import streamlit as st
import pandas as pd
import requests

# Streamlit app
st.title("Music Recommendation App")
st.write("Choose how you want to provide your preferences:")

# Define function to fetch recommendations from the backend
def fetch_recommendations(preferences):
    api_url = "http://fastapi:80/recommend"
    response = requests.post(api_url, json=preferences)
    if response.status_code == 200:
        return response.json()["recommendations"]
    else:
        st.error(f"Error fetching recommendations: {response.status_code} - {response.text}")
        return []

# Option to load preferences
option = st.radio("Input your preferences", 
                  ("Upload a CSV file", "Manually Enter Preferences", "Use a Sample Dataset"))

if option == "Upload a CSV file":
    uploaded_file = st.file_uploader("Choose a CSV file", type='csv')
    if uploaded_file:
        user_uploaded_data = pd.read_csv(uploaded_file)
        st.write(f"Uploaded Data: {len(user_uploaded_data)} songs")
        st.write(user_uploaded_data[['track_name', 'artists']])

        required_features = [
            'danceability', 'energy', 'key', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
        ]
        if all(feature in user_uploaded_data.columns for feature in required_features):
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
            recommendations = fetch_recommendations([preferences])
            st.write("Recommendations based on your first song:")
            st.write(recommendations)
        else:
            st.error("Uploaded file is missing required features.")

elif option == "Manually Enter Preferences":
    with st.form("manual_entry"):
        danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
        energy = st.slider("Energy", 0.0, 1.0, 0.5)
        key = st.number_input("Key", min_value=0, max_value=11, value=5)
        loudness = st.slider("Loudness (dB)", -60.0, 0.0, -20.0)
        speechiness = st.slider("Speechiness", 0.0, 1.0, 0.1)
        acousticness = st.slider("Acousticness", 0.0, 1.0, 0.3)
        instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
        liveness = st.slider("Liveness", 0.0, 1.0, 0.2)
        valence = st.slider("Valence", 0.0, 1.0, 0.6)
        tempo = st.slider("Tempo (BPM)", 50.0, 250.0, 120.0)

        submitted = st.form_submit_button("Get Recommendations")
        if submitted:
            preferences = [{
                'danceability': danceability,
                'energy': energy,
                'key': int(key),
                'loudness': loudness,
                'speechiness': speechiness,
                'acousticness': acousticness,
                'instrumentalness': instrumentalness,
                'liveness': liveness,
                'valence': valence,
                'tempo': tempo
            }]
            recommendations = fetch_recommendations(preferences)
            st.write("Recommendations based on your input:")
            st.write(recommendations)

elif option == "Use a Sample Dataset":
    sample_data = {
        "danceability": 0.8,
        "energy": 0.7,
        "key": 5,
        "loudness": -10.0,
        "speechiness": 0.05,
        "acousticness": 0.2,
        "instrumentalness": 0.0,
        "liveness": 0.15,
        "valence": 0.75,
        "tempo": 120.0
    }
    st.write("Using the following sample preferences:")
    st.write(sample_data)

    recommendations = fetch_recommendations([sample_data])
    st.write("Recommendations:")
    st.write(recommendations)
