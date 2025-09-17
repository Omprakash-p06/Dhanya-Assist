import sqlite3
import os
from datetime import datetime

def init_database():
    """Initialize SQLite database for user sessions and data"""
    
    # Create db directory if it doesn't exist
    os.makedirs('db', exist_ok=True)
    
    # Connect to database (creates file if doesn't exist)
    conn = sqlite3.connect('db/dhanya_assist.db')
    cursor = conn.cursor()
    
    # Create users/sessions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT UNIQUE,
        language TEXT DEFAULT 'en',
        location_lat REAL,
        location_lon REAL,
        last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create crop predictions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS crop_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        input_data TEXT, -- JSON string of soil/weather data
        predictions TEXT, -- JSON string of crop recommendations
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (session_id) REFERENCES user_sessions (session_id)
    )
    ''')
    
    # Create image analysis table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS image_analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        filename TEXT,
        analysis_result TEXT, -- JSON string of disease detection results
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (session_id) REFERENCES user_sessions (session_id)
    )
    ''')
    
    # Create feedback table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        feedback_text TEXT,
        rating INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (session_id) REFERENCES user_sessions (session_id)
    )
    ''')
    
    conn.commit()
    conn.close()
    
    print("Database initialized successfully!")

def get_db_connection():
    """Get database connection"""
    return sqlite3.connect('db/dhanya_assist.db')

def save_user_session(session_id, language='en', lat=None, lon=None):
    """Save or update user session"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT OR REPLACE INTO user_sessions (session_id, language, location_lat, location_lon, last_accessed)
    VALUES (?, ?, ?, ?, ?)
    ''', (session_id, language, lat, lon, datetime.now()))
    
    conn.commit()
    conn.close()

def save_crop_prediction(session_id, input_data, predictions):
    """Save crop prediction results"""
    import json
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO crop_predictions (session_id, input_data, predictions)
    VALUES (?, ?, ?)
    ''', (session_id, json.dumps(input_data), json.dumps(predictions)))
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_database()
