import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data():
    """Load and preprocess the crop dataset"""
    try:
        # Load dataset
        df = pd.read_csv('data/crop_dataset.csv')
        
        # Basic info
        print(f"Dataset shape: {df.shape}")
        print(f"Crops available: {df['label'].unique()}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        
        # Encode crop labels
        le = LabelEncoder()
        df['crop_encoded'] = le.fit_transform(df['label'])
        
        # Feature scaling (optional)
        feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        # Save processed data
        df.to_csv('data/processed_crop_dataset.csv', index=False)
        
        return df, le
        
    except FileNotFoundError:
        print("Dataset not found. Please download crop_dataset.csv")
        return None, None

if __name__ == "__main__":
    df, label_encoder = load_and_preprocess_data()
