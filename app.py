from flask import Flask, render_template, request, jsonify, session
import os
import json
from dotenv import load_dotenv
from utils.ml_model import crop_model
from utils.weather import WeatherService
from utils.soil import SoilService
from utils.vision import VisionService
from db.init_db import init_database, save_user_session, save_crop_prediction

# Set working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Load environment variables
load_dotenv()

# Initialize database
init_database()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dhanya-assist-secret-key-2024')

# Initialize services
weather_service = WeatherService()
soil_service = SoilService() 
vision_service = VisionService()

# Load ML model
crop_model.load_model()

# Language configuration
LANGUAGES = {
    'en': 'English',
    'hi': 'हिन्दी',
    'kn': 'ಕನ್ನಡ'
}

def load_translations():
    """Load all translation files"""
    translations = {}
    for lang_code in LANGUAGES.keys():
        try:
            translation_file = os.path.join(script_dir, 'translations', f'{lang_code}.json')
            with open(translation_file, 'r', encoding='utf-8') as f:
                translations[lang_code] = json.load(f)
                print(f"✅ Translation file loaded: {lang_code}.json")
        except FileNotFoundError:
            print(f"⚠️ Translation file not found: {lang_code}.json")
            translations[lang_code] = {}
    return translations

# Load translations at startup
translations = load_translations()

def get_text(key, lang='en'):
    """Get translated text for a given key and language"""
    return translations.get(lang, {}).get(key, key)

@app.route('/')
def index():
    """Main dashboard route"""
    # Get language from session or default to English
    current_lang = session.get('language', 'en')
    
    return render_template('index.html', 
                         current_lang=current_lang,
                         languages=LANGUAGES,
                         get_text=get_text)

@app.route('/set_language/<lang>')
def set_language(lang):
    """Set user's preferred language"""
    if lang in LANGUAGES:
        session['language'] = lang
    return jsonify({'status': 'success', 'language': lang})

@app.route('/recommend', methods=['POST'])
def recommend_crops():
    """Enhanced crop recommendation with ML model"""
    current_lang = session.get('language', 'en')
    session_id = session.get('session_id', 'anonymous')
    
    try:
        # Get input data
        data = request.get_json()
        
        # Use ML model for predictions
        recommendations = crop_model.predict_crop(data)
        
        # Save prediction to database
        save_crop_prediction(session_id, data, recommendations)
        
        return jsonify({
            'status': 'success',
            'recommendations': recommendations,
            'message': get_text('recommendations_ready', current_lang)
        })
        
    except Exception as e:
        print(f"Recommendation error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Unable to generate recommendations',
            'recommendations': []
        })

@app.route('/weather')
def get_weather():
    """Get weather data for location"""
    current_lang = session.get('language', 'en')

    # Get latitude and longitude from query parameters with defaults
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)

    # If no coordinates provided, use default location (India center)
    if lat is None or lon is None:
        lat = 20.5937  # India center latitude
        lon = 78.9629  # India center longitude

    try:
        # Use real weather service
        weather_result = weather_service.get_current_weather(lat, lon)
        
        if weather_result['success']:
            weather_data = weather_result['data']
            return jsonify({
                'temperature': weather_data['temperature'],
                'humidity': weather_data['humidity'],
                'rainfall': 0,  # OpenWeather doesn't provide current rainfall in basic plan
                'description': weather_data['description'],
                'location': weather_data['location'],
                'country': weather_data['country'],
                'wind_speed': weather_data['wind_speed'],
                'pressure': weather_data['pressure'],
                'sunrise': weather_data['sunrise'],
                'sunset': weather_data['sunset'],
                'icon': weather_data['icon']
            })
        else:
            # Fallback to mock data if API fails
            return jsonify({
                'temperature': 28,
                'humidity': 65,
                'rainfall': 12,
                'description': get_text('weather_desc', current_lang),
                'location': 'Demo Location',
                'country': 'IN',
                'error': weather_result.get('error', 'Weather service unavailable')
            })
            
    except Exception as e:
        print(f"Weather error: {str(e)}")
        return jsonify({
            'temperature': 28,
            'humidity': 65,
            'rainfall': 12,
            'description': get_text('weather_desc', current_lang),
            'location': 'Demo Location',
            'country': 'IN',
            'error': 'Weather service error'
        })

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle crop image upload for disease detection using real computer vision"""
    current_lang = session.get('language', 'en')
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': get_text('no_file_selected', current_lang)})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': get_text('no_file_selected', current_lang)})
        
        # Use the real VisionService to analyze the uploaded image
        analysis_result = vision_service.analyze_crop_image(file)
        
        if analysis_result.get('success'):
            if analysis_result.get('object_type') == 'person':
                return jsonify({
                    'status': 'error',
                    'error': 'Human detected in image',
                    'message': analysis_result.get('recommendation', 'Please upload an image of a plant for disease analysis'),
                    'confidence': analysis_result.get('confidence', 0),
                    'object_type': 'person'
                })
            
            elif analysis_result.get('object_type') == 'plant':
                analysis = analysis_result.get('analysis', {})
                return jsonify({
                    'status': 'success',
                    'diagnosis': {
                        'disease': analysis.get('disease_detected', 'Unknown'),
                        'confidence': analysis.get('confidence', 0),
                        'crop_type': analysis.get('crop_detected', 'Unknown'),
                        'severity': analysis.get('severity', 'Unknown'),
                        'treatment': analysis.get('treatment_recommendations', []),
                        'prevention': analysis.get('prevention_tips', []),
                        'health_score': analysis.get('technical_analysis', {}).get('health_score', 0),
                        'analysis_method': analysis.get('analysis_method', 'Computer Vision'),
                        'image_quality': analysis.get('image_quality_assessment', 'Good')
                    },
                    'technical_details': analysis.get('technical_analysis', {}),
                    'confidence_factors': analysis.get('confidence_factors', {}),
                    'message': get_text('analysis_complete', current_lang)
                })
            
            else:  # unknown object
                return jsonify({
                    'status': 'error',
                    'error': 'Object not recognized',
                    'message': analysis_result.get('recommendation', 'Please upload a clear image of a plant'),
                    'confidence': analysis_result.get('confidence', 0),
                    'object_type': 'unknown'
                })
        
        else:
            # Analysis failed
            return jsonify({
                'status': 'error',
                'error': analysis_result.get('error', 'Image analysis failed'),
                'message': 'Please try uploading a different image with better lighting'
            })
            
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': f'Processing failed: {str(e)}',
            'message': 'Please try again with a different image'
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
