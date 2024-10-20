import os
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def load_data(file_path):
    """
    Loads the Spotify dataset from the given file path.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Number of rows: {len(df)}")
        return df
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None


def clean_data(df):
    """
    Cleans the Spotify dataset by handling missing values and selecting relevant features.
    """
    if df is None:
        return None
    
    # Drop any rows with missing values
    df_cleaned = df.dropna()

    # Select relevant features for genre prediction
    relevant_columns = [
        'artists', 'album_name', 'track_name', 'popularity', 'duration_ms', 
        'explicit', 'danceability', 'energy', 'key', 'loudness', 'mode', 
        'speechiness', 'acousticness', 'instrumentalness', 'liveness', 
        'valence', 'tempo', 'time_signature', 'track_genre'
    ]
    df_cleaned = df_cleaned[relevant_columns]

    print(f"Data cleaned. Number of rows after cleaning: {len(df_cleaned)}")
    return df_cleaned


def load_and_clean_data(file_path):
    """
    Combines loading and cleaning into one function.
    """
    df = load_data(file_path)
    return clean_data(df)


# Exploratory Data Analysis Functions
def basic_statistics(df):
    """ Prints basic statistics of the dataset. """
    print(df.describe())
    print(df.info())


def plot_genre_distribution(df):
    """ Plots the distribution of genres in the dataset. """
    plt.figure(figsize=(10, 6))
    sns.countplot(y='track_genre', data=df, order=df['track_genre'].value_counts().index)
    plt.title("Genre Distribution")
    plt.xlabel("Count")
    plt.ylabel("Genres")
    plt.show()


def plot_audio_features(df):
    """ Plots the distribution of selected audio features. """
    audio_features = ['danceability', 'energy', 'speechiness', 'acousticness', 
                      'instrumentalness', 'liveness', 'valence']
    
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(audio_features, 1):
        plt.subplot(3, 3, i)
        sns.histplot(df[feature], kde=True)
        plt.title(f'Distribution of {feature}')
    plt.tight_layout()
    plt.show()


features = [
    'danceability', 'energy', 'key', 'loudness', 'speechiness', 
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

# Main Code Execution
if __name__ == "__main__":
    # Step 1: Data Loading
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    file_path = os.path.join(parent_dir, "data", "spotify_tracks_genre.csv")
    df = load_and_clean_data(file_path)

    if df is not None:
        print(df.iloc[0])  # Display the first row of the cleaned DataFrame

        # Step 2: Save the cleaned data
        cleaned_data_path = os.path.join(parent_dir, "data", "spotify_cleaned.csv")
        df.to_csv(cleaned_data_path, index=False)
        print(f"Cleaned data saved at: {cleaned_data_path}")

        # Step 3: Randomly select 20 rows and save to CSV
        df_sample = df.sample(n=20, random_state=random.randint(1, 100))
        downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
        sample_output_path = os.path.join(downloads_folder, "spotify_tracks_genre_sample.csv")
        df_sample.to_csv(sample_output_path, index=False)
        print(f"Sample data saved at: {sample_output_path}")

        # Step 4: Exploratory Data Analysis
        basic_statistics(df)
        plot_genre_distribution(df)
        plot_audio_features(df)
    else:
        print("No data to process.")
