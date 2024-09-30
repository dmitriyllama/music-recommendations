import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_and_preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Fill missing values (e.g., Age, Embarked, Cabin can have missing values)
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df.drop(columns=['Cabin', 'Ticket', 'Name', 'PassengerId'], inplace=True)  # Drop irrelevant features
    
    # Encode categorical data
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    # Split data into training and test sets
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    # Split the path into its components
    path_components = os.path.abspath(__file__).split(os.sep)
    X_train, X_test, y_train, y_test = load_and_preprocess_data(os.path.join(os.sep.join(path_components[:path_components.index('PML-Assignment') + 1]), 'data', 'titanic', 'train.csv'))
    print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")
    X_train_features = X_train.columns.tolist()
    print("Features in X_train:", X_train_features)
