from flask import Flask, render_template, request, jsonify, session
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dhanya-assist-secret-key-2024')

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
            with open(f'translations/{lang_code}.json', 'r', encoding='utf-8') as f:
                translations[lang_code] = json.load(f)
        except FileNotFoundError:
            print(f"Translation file not found: {lang_code}.json")
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
    """API endpoint for crop recommendations"""
    current_lang = session.get('language', 'en')
    
    # Get input data
    data = request.get_json()
    
    # Mock response for MVP (replace with actual ML model later)
    mock_recommendations = [
        {
            'crop': get_text('crop_rice', current_lang),
            'confidence': 85,
            'yield_estimate': '4.5 tons/hectare',
            'profit_estimate': '₹45,000/hectare'
        },
        {
            'crop': get_text('crop_wheat', current_lang),
            'confidence': 78,
            'yield_estimate': '3.8 tons/hectare',
            'profit_estimate': '₹38,000/hectare'
        },
        {
            'crop': get_text('crop_sugarcane', current_lang),
            'confidence': 72,
            'yield_estimate': '65 tons/hectare',
            'profit_estimate': '₹85,000/hectare'
        }
    ]
    
    return jsonify({
        'status': 'success',
        'recommendations': mock_recommendations,
        'message': get_text('recommendations_ready', current_lang)
    })

@app.route('/weather')
def get_weather():
    """Get weather data for location"""
    # Mock weather data for MVP
    current_lang = session.get('language', 'en')
    
    mock_weather = {
        'temperature': 28,
        'humidity': 65,
        'rainfall': 12,
        'description': get_text('weather_desc', current_lang)
    }
    
    return jsonify(mock_weather)

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle crop image upload for disease detection"""
    current_lang = session.get('language', 'en')
    
    if 'file' not in request.files:
        return jsonify({'error': get_text('no_file_selected', current_lang)})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': get_text('no_file_selected', current_lang)})
    
    # Mock disease detection response for MVP
    mock_diagnosis = {
        'disease': get_text('disease_leaf_blight', current_lang),
        'confidence': 82,
        'treatment': get_text('treatment_fungicide', current_lang),
        'prevention': get_text('prevention_spacing', current_lang)
    }
    
    return jsonify({
        'status': 'success',
        'diagnosis': mock_diagnosis,
        'message': get_text('analysis_complete', current_lang)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
