import pandas as pd
import numpy as np

def generate_sample_crop_data():
    """Generate sample data if Kaggle dataset unavailable"""
    np.random.seed(42)
    
    crops = ['rice', 'wheat', 'maize', 'cotton', 'sugarcane']
    data = []
    
    for crop in crops:
        for _ in range(50):  # 50 samples per crop
            if crop == 'rice':
                row = {
                    'N': np.random.uniform(80, 100),
                    'P': np.random.uniform(40, 60), 
                    'K': np.random.uniform(35, 45),
                    'temperature': np.random.uniform(20, 27),
                    'humidity': np.random.uniform(75, 85),
                    'ph': np.random.uniform(6.0, 7.5),
                    'rainfall': np.random.uniform(180, 280),
                    'label': crop
                }
            # Add similar logic for other crops...
            data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv('data/crop_dataset.csv', index=False)
    print("Sample dataset created!")

if __name__ == "__main__":
    generate_sample_crop_data()
