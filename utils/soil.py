import requests
import os
from dotenv import load_dotenv
import json

load_dotenv()

class SoilService:
    def __init__(self):
        self.agromonitoring_api_key = os.getenv('AGROMONITORING_API_KEY')
        self.base_url = "http://api.agromonitoring.com/agro/1.0"
        
    def get_soil_data(self, lat, lon):
        """Get soil data for specific coordinates"""
        try:
            # Try Agromonitoring API first
            soil_data = self._get_agromonitoring_soil(lat, lon)
            if soil_data['success']:
                return soil_data
                
            # Fallback to mock data if API fails
            return self._get_mock_soil_data()
            
        except Exception as e:
            print(f"Soil data error: {str(e)}")
            return self._get_mock_soil_data()
    
    def _get_agromonitoring_soil(self, lat, lon):
        """Get soil data from Agromonitoring API"""
        try:
            url = f"{self.base_url}/soil"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.agromonitoring_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            return {
                'success': True,
                'data': {
                    'temperature': round(data.get('temp', 298) - 273.15, 1),  # Convert Kelvin to Celsius
                    'moisture': data.get('moisture', 0.3),
                    'ph': self._estimate_ph(lat, lon),  # Estimated based on location
                    'organic_matter': self._estimate_organic_matter(),
                    'nitrogen': self._estimate_nutrient('nitrogen'),
                    'phosphorus': self._estimate_nutrient('phosphorus'),
                    'potassium': self._estimate_nutrient('potassium'),
                    'soil_type': self._determine_soil_type(lat, lon)
                }
            }
            
        except requests.exceptions.RequestException:
            return {'success': False}
    
    def analyze_soil_suitability(self, soil_data, crop_type):
        """Analyze soil suitability for specific crop"""
        suitability_scores = {
            'rice': self._analyze_rice_suitability(soil_data),
            'wheat': self._analyze_wheat_suitability(soil_data),
            'sugarcane': self._analyze_sugarcane_suitability(soil_data),
            'cotton': self._analyze_cotton_suitability(soil_data),
            'maize': self._analyze_maize_suitability(soil_data),
            'tomato': self._analyze_tomato_suitability(soil_data)
        }
        
        return suitability_scores.get(crop_type.lower(), 0.5)
    
    def _analyze_rice_suitability(self, soil_data):
        """Analyze soil suitability for rice cultivation"""
        score = 0.0
        
        # pH preference: 5.5-7.0
        ph = soil_data.get('ph', 6.5)
        if 5.5 <= ph <= 7.0:
            score += 0.3
        elif 5.0 <= ph <= 8.0:
            score += 0.2
        
        # High moisture requirement
        moisture = soil_data.get('moisture', 0.3)
        if moisture > 0.7:
            score += 0.3
        elif moisture > 0.5:
            score += 0.2
        
        # Nitrogen requirement
        nitrogen = soil_data.get('nitrogen', 50)
        if nitrogen > 60:
            score += 0.2
        elif nitrogen > 40:
            score += 0.1
        
        # Clay soil preference
        soil_type = soil_data.get('soil_type', 'loam')
        if 'clay' in soil_type.lower():
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_wheat_suitability(self, soil_data):
        """Analyze soil suitability for wheat cultivation"""
        score = 0.0
        
        # pH preference: 6.0-7.5
        ph = soil_data.get('ph', 6.5)
        if 6.0 <= ph <= 7.5:
            score += 0.3
        elif 5.5 <= ph <= 8.0:
            score += 0.2
        
        # Moderate moisture requirement
        moisture = soil_data.get('moisture', 0.3)
        if 0.4 <= moisture <= 0.7:
            score += 0.3
        elif 0.3 <= moisture <= 0.8:
            score += 0.2
        
        # Well-drained soil preference
        if 'loam' in soil_data.get('soil_type', 'loam').lower():
            score += 0.2
        
        # Phosphorus requirement
        phosphorus = soil_data.get('phosphorus', 30)
        if phosphorus > 25:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_sugarcane_suitability(self, soil_data):
        """Analyze soil suitability for sugarcane cultivation"""
        score = 0.0
        
        # pH preference: 6.5-7.5
        ph = soil_data.get('ph', 6.5)
        if 6.5 <= ph <= 7.5:
            score += 0.3
        elif 6.0 <= ph <= 8.0:
            score += 0.2
        
        # High moisture and nutrient requirement
        moisture = soil_data.get('moisture', 0.3)
        if moisture > 0.6:
            score += 0.2
        
        nitrogen = soil_data.get('nitrogen', 50)
        if nitrogen > 70:
            score += 0.3
        elif nitrogen > 50:
            score += 0.2
        
        potassium = soil_data.get('potassium', 40)
        if potassium > 50:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_cotton_suitability(self, soil_data):
        """Analyze soil suitability for cotton cultivation"""
        score = 0.0
        
        # pH preference: 5.8-8.0
        ph = soil_data.get('ph', 6.5)
        if 5.8 <= ph <= 8.0:
            score += 0.3
        
        # Deep, well-drained soil preference
        if 'clay' in soil_data.get('soil_type', 'loam').lower():
            score += 0.2
        
        # Moderate to high nutrient requirement
        nitrogen = soil_data.get('nitrogen', 50)
        if nitrogen > 40:
            score += 0.3
        
        phosphorus = soil_data.get('phosphorus', 30)
        if phosphorus > 20:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_maize_suitability(self, soil_data):
        """Analyze soil suitability for maize cultivation"""
        score = 0.0
        
        # pH preference: 6.0-7.0
        ph = soil_data.get('ph', 6.5)
        if 6.0 <= ph <= 7.0:
            score += 0.3
        elif 5.5 <= ph <= 7.5:
            score += 0.2
        
        # Well-drained soil
        if 'loam' in soil_data.get('soil_type', 'loam').lower():
            score += 0.3
        
        # High nitrogen requirement
        nitrogen = soil_data.get('nitrogen', 50)
        if nitrogen > 60:
            score += 0.4
        elif nitrogen > 40:
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_tomato_suitability(self, soil_data):
        """Analyze soil suitability for tomato cultivation"""
        score = 0.0
        
        # pH preference: 6.0-6.8
        ph = soil_data.get('ph', 6.5)
        if 6.0 <= ph <= 6.8:
            score += 0.3
        elif 5.5 <= ph <= 7.0:
            score += 0.2
        
        # Well-drained, organic-rich soil
        organic_matter = soil_data.get('organic_matter', 3)
        if organic_matter > 4:
            score += 0.3
        elif organic_matter > 2:
            score += 0.2
        
        # Balanced nutrients
        if all(soil_data.get(nutrient, 0) > 30 for nutrient in ['nitrogen', 'phosphorus', 'potassium']):
            score += 0.4
        
        return min(score, 1.0)
    
    def _estimate_ph(self, lat, lon):
        """Estimate pH based on geographical location"""
        # Simple estimation based on Indian regions
        if 8 <= lat <= 37 and 68 <= lon <= 97:  # India bounds
            if lat > 30:  # Northern regions (typically alkaline)
                return round(7.2 + (lat - 30) * 0.1, 1)
            elif lat < 15:  # Southern regions (typically acidic)
                return round(6.8 - (15 - lat) * 0.1, 1)
            else:  # Central regions
                return 6.5
        return 6.5  # Default neutral
    
    def _estimate_organic_matter(self):
        """Estimate organic matter percentage"""
        import random
        return round(random.uniform(2.5, 5.0), 1)
    
    def _estimate_nutrient(self, nutrient_type):
        """Estimate nutrient content"""
        import random
        ranges = {
            'nitrogen': (30, 80),
            'phosphorus': (15, 50),
            'potassium': (25, 70)
        }
        min_val, max_val = ranges.get(nutrient_type, (20, 60))
        return round(random.uniform(min_val, max_val), 1)
    
    def _determine_soil_type(self, lat, lon):
        """Determine soil type based on location"""
        # Simplified soil type mapping for India
        if lat > 28:  # Northern plains
            return "alluvial loam"
        elif lat < 15:  # Southern regions
            return "red clay loam"
        else:  # Central regions
            return "black clay"
    
    def _get_mock_soil_data(self):
        """Return mock soil data for demo purposes"""
        return {
            'success': True,
            'data': {
                'temperature': 25.5,
                'moisture': 0.45,
                'ph': 6.8,
                'organic_matter': 3.2,
                'nitrogen': 55.0,
                'phosphorus': 32.0,
                'potassium': 48.0,
                'soil_type': 'loam'
            }
        }
