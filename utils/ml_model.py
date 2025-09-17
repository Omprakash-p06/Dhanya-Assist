import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import pickle
import os

class CropRecommendationModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
    def load_and_train(self, data_path=None):
        """Load dataset and train the model"""
        try:
            if data_path is None:
                # Get the directory where this script is located
                script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                data_path = os.path.join(script_dir, 'data', 'Crop_dataset.csv')
            
            # Load dataset
            df = pd.read_csv(data_path)
            print(f"Dataset loaded: {df.shape[0]} records, {df.shape[1]} features")
            
            # Prepare features and labels
            X = df[self.feature_columns]
            y = df['label']
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Model trained with accuracy: {accuracy:.2%}")
            
            # Save model
            self.save_model()
            
            return True, accuracy
            
        except Exception as e:
            print(f"Training failed: {str(e)}")
            return False, 0
    
    def predict_crop(self, soil_data):
        """Predict best crops for given soil conditions"""
        try:
            # Prepare input data
            features = [
                soil_data.get('N', 50),
                soil_data.get('P', 30),
                soil_data.get('K', 40),
                soil_data.get('temperature', 25),
                soil_data.get('humidity', 60),
                soil_data.get('ph', 6.5),
                soil_data.get('rainfall', 150)
            ]
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Get top 3 recommendations
            top_indices = np.argsort(probabilities)[-3:][::-1]
            
            recommendations = []
            for idx in top_indices:
                crop_name = self.label_encoder.inverse_transform([idx])[0]
                confidence = probabilities[idx] * 100
                
                recommendations.append({
                    'crop': crop_name,
                    'confidence': round(confidence, 1),
                    'yield_estimate': self._estimate_yield(crop_name, soil_data),
                    'profit_estimate': self._estimate_profit(crop_name)
                })
            
            return recommendations
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return self._get_mock_recommendations()
    
    def _estimate_yield(self, crop, soil_data):
        """Estimate crop yield based on soil conditions"""
        base_yields = {
            'rice': 4.5, 'wheat': 3.2, 'maize': 5.8, 'cotton': 2.1, 
            'sugarcane': 75.0, 'banana': 25.0, 'mango': 12.0
        }
        base_yield = base_yields.get(crop, 3.0)
        
        # Adjust based on soil quality (simplified)
        ph = soil_data.get('ph', 6.5)
        if 6.0 <= ph <= 7.5:
            multiplier = 1.1
        else:
            multiplier = 0.9
            
        return f"{base_yield * multiplier:.1f} tons/hectare"
    
    def _estimate_profit(self, crop):
        """Estimate profit per hectare"""
        base_profits = {
            'rice': 45000, 'wheat': 35000, 'maize': 40000, 'cotton': 60000,
            'sugarcane': 85000, 'banana': 120000, 'mango': 90000
        }
        profit = base_profits.get(crop, 30000)
        return f"₹{profit:,}/hectare"
    
    def save_model(self):
        """Save trained model to file"""
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(script_dir, 'model')
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'crop_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print("Model saved successfully!")
    
    def load_model(self):
        """Load trained model from file"""
        try:
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(script_dir, 'model', 'crop_model.pkl')
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                
            self.model = model_data['model']
            self.label_encoder = model_data['label_encoder']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            
            print("Model loaded successfully!")
            return True
            
        except FileNotFoundError:
            print("Model file not found. Training new model...")
            return self.load_and_train()
        except Exception as e:
            print(f"Model loading failed: {str(e)}")
            return False
    
    def _get_mock_recommendations(self):
        """Return mock recommendations if model fails"""
        return [
            {'crop': 'rice', 'confidence': 75.0, 'yield_estimate': '4.2 tons/hectare', 'profit_estimate': '₹42,000/hectare'},
            {'crop': 'wheat', 'confidence': 68.0, 'yield_estimate': '3.1 tons/hectare', 'profit_estimate': '₹31,000/hectare'},
            {'crop': 'maize', 'confidence': 62.0, 'yield_estimate': '5.5 tons/hectare', 'profit_estimate': '₹38,000/hectare'}
        ]

# Initialize global model instance
crop_model = CropRecommendationModel()
