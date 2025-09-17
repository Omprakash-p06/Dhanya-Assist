import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

def create_simple_plant_classifier():
    """Create a simple rule-based plant disease classifier for demo"""
    
    # Create model directory
    os.makedirs('model', exist_ok=True)
    
    print("Creating simple plant disease classifier...")
    
    # Create a simple lookup-based classifier
    disease_rules = {
        'tomato': {
            'healthy': {'green_ratio': (0.6, 1.0), 'brightness': (0.4, 0.8)},
            'early_blight': {'green_ratio': (0.3, 0.6), 'brightness': (0.2, 0.6)},
            'late_blight': {'green_ratio': (0.2, 0.5), 'brightness': (0.1, 0.4)},
            'bacterial_spot': {'green_ratio': (0.4, 0.7), 'brightness': (0.3, 0.7)}
        },
        'potato': {
            'healthy': {'green_ratio': (0.5, 0.9), 'brightness': (0.4, 0.8)},
            'early_blight': {'green_ratio': (0.3, 0.6), 'brightness': (0.2, 0.5)},
            'late_blight': {'green_ratio': (0.2, 0.4), 'brightness': (0.1, 0.3)}
        },
        'corn': {
            'healthy': {'green_ratio': (0.6, 0.9), 'brightness': (0.5, 0.8)},
            'common_rust': {'green_ratio': (0.3, 0.6), 'brightness': (0.3, 0.6)},
            'leaf_blight': {'green_ratio': (0.2, 0.5), 'brightness': (0.2, 0.5)}
        }
    }
    
    # Save the rules as a pickle file
    with open('model/plant_disease_rules.pkl', 'wb') as f:
        pickle.dump(disease_rules, f)
    
    print("✅ Simple plant disease classifier created successfully!")
    print("File saved: model/plant_disease_rules.pkl")
    
    return True

def download_kaggle_dataset():
    """Instructions for downloading Kaggle dataset manually"""
    print("\n" + "="*50)
    print("MANUAL DOWNLOAD INSTRUCTIONS:")
    print("="*50)
    print("1. Go to: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset")
    print("2. Click 'Download' (requires Kaggle account)")
    print("3. Extract the ZIP file")
    print("4. Copy 'New Plant Diseases Dataset(Augmented)' to your project folder")
    print("5. Rename it to 'plant_disease_dataset'")
    print("\nOR use our simple rule-based classifier (already created above)")
    print("="*50)

if __name__ == "__main__":
    # Create simple classifier (always works)
    create_simple_plant_classifier()
    
    # Show manual download instructions
    download_kaggle_dataset()
