import requests
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class WeatherService:
    def __init__(self):
        self.api_key = os.getenv('OPENWEATHER_API_KEY')
        self.base_url = "http://api.openweathermap.org/data/2.5"
        
    def get_current_weather(self, lat, lon):
        """Get current weather data by coordinates"""
        try:
            url = f"{self.base_url}/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'  # For Celsius
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            return {
                'success': True,
                'data': {
                    'temperature': round(data['main']['temp'], 1),
                    'humidity': data['main']['humidity'],
                    'pressure': data['main']['pressure'],
                    'description': data['weather'][0]['description'].title(),
                    'wind_speed': data.get('wind', {}).get('speed', 0),
                    'visibility': data.get('visibility', 0) / 1000,  # Convert to km
                    'location': data['name'],
                    'country': data['sys']['country'],
                    'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M'),
                    'sunset': datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M'),
                    'icon': data['weather'][0]['icon']
                }
            }
            
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f'Weather API request failed: {str(e)}',
                'data': self._get_mock_weather()
            }
        except KeyError as e:
            return {
                'success': False,
                'error': f'Invalid weather data format: {str(e)}',
                'data': self._get_mock_weather()
            }
    
    def get_weather_by_city(self, city_name):
        """Get current weather data by city name"""
        try:
            url = f"{self.base_url}/weather"
            params = {
                'q': city_name,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            return {
                'success': True,
                'data': {
                    'temperature': round(data['main']['temp'], 1),
                    'humidity': data['main']['humidity'],
                    'pressure': data['main']['pressure'],
                    'description': data['weather'][0]['description'].title(),
                    'wind_speed': data.get('wind', {}).get('speed', 0),
                    'location': data['name'],
                    'country': data['sys']['country'],
                    'coordinates': {
                        'lat': data['coord']['lat'],
                        'lon': data['coord']['lon']
                    }
                }
            }
            
        except requests.exceptions.RequestException:
            return {
                'success': False,
                'error': 'Unable to fetch weather data',
                'data': self._get_mock_weather()
            }
    
    def get_forecast(self, lat, lon, days=5):
        """Get weather forecast for specified days"""
        try:
            url = f"{self.base_url}/forecast"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric',
                'cnt': days * 8  # 8 forecasts per day (3-hour intervals)
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Process forecast data
            forecast_list = []
            for item in data['list'][:days]:
                forecast_list.append({
                    'date': datetime.fromtimestamp(item['dt']).strftime('%Y-%m-%d'),
                    'temperature': round(item['main']['temp'], 1),
                    'humidity': item['main']['humidity'],
                    'description': item['weather'][0]['description'].title(),
                    'rain_probability': item.get('pop', 0) * 100
                })
            
            return {
                'success': True,
                'forecast': forecast_list
            }
            
        except requests.exceptions.RequestException:
            return {
                'success': False,
                'error': 'Unable to fetch forecast data',
                'forecast': self._get_mock_forecast()
            }
    
    def _get_mock_weather(self):
        """Return mock weather data for demo purposes"""
        return {
            'temperature': 28.5,
            'humidity': 65,
            'pressure': 1013,
            'description': 'Partly Cloudy',
            'wind_speed': 3.2,
            'visibility': 10,
            'location': 'Demo Location',
            'country': 'IN',
            'sunrise': '06:30',
            'sunset': '18:45',
            'icon': '02d'
        }
    
    def _get_mock_forecast(self):
        """Return mock forecast data"""
        return [
            {'date': '2025-09-16', 'temperature': 29, 'humidity': 68, 'description': 'Sunny', 'rain_probability': 10},
            {'date': '2025-09-17', 'temperature': 31, 'humidity': 72, 'description': 'Partly Cloudy', 'rain_probability': 25},
            {'date': '2025-09-18', 'temperature': 27, 'humidity': 80, 'description': 'Light Rain', 'rain_probability': 70}
        ]
