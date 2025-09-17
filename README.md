# 🌾 Dhanya Assist - AI-Powered Agricultural Assistant

An intelligent web application that helps farmers and agricultural enthusiasts with crop recommendations, disease detection, weather monitoring, and soil analysis using machine learning and computer vision.

## 🚀 Features

- **� Smart Crop Recommendation**: ML-powered crop suggestions based on soil and climate conditions
- **🔍 Plant Disease Detection**: Advanced image analysis for identifying plant diseases
- **🌤️ Real-time Weather Integration**: Location-based weather data with farming insights
- **🗺️ Interactive Maps**: Geolocation-based weather and agricultural data
- **🌍 Multi-language Support**: Available in English, Hindi, and Kannada
- **💾 Data Persistence**: SQLite database for storing user sessions and predictions

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8 or higher (recommended: 3.9-3.12)
- **Operating System**: Windows 10/11, Linux (Ubuntu 18.04+), macOS 10.15+
- **Memory**: At least 4GB RAM
- **Storage**: 500MB free space

### API Keys Required
- **OpenWeatherMap API Key**: For weather data
  - Sign up at [OpenWeatherMap](https://openweathermap.org/api)
  - Get your free API key

## 🛠️ Installation Guide

### 📁 1. Clone or Download the Project

```bash
# If using Git
git clone https://github.com/Omprakash-p06/Dhanya-Assist.git
cd Dhanya-Assist

# Or download and extract the ZIP file
```

### 🔧 2. Set up Python Environment

#### Option A: Using Virtual Environment (Recommended)

**On Windows:**
```cmd
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Verify activation (should show venv path)
where python
```

**On Linux/Mac:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify activation (should show venv path)
which python
```

#### Option B: Using Conda (Alternative)

```bash
# Create conda environment
conda create -n dhanya-assist python=3.10
conda activate dhanya-assist
```

### 📦 3. Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

#### 🚨 Troubleshooting Dependency Issues

**If you encounter NumPy/OpenCV compatibility issues:**

```bash
# Option 1: Install compatible versions
pip uninstall opencv-python numpy -y
pip install "numpy<2.0" opencv-python==4.8.1.78

# Option 2: Use basic mode (OpenCV optional)
pip install -r requirements.txt --no-deps
pip install Flask python-dotenv requests Pillow pandas scikit-learn numpy
```

**For Python 3.13 users:**
```bash
# Use pre-compiled wheels only
pip install --only-binary=all -r requirements.txt
```

### 🔑 4. Configure Environment Variables

Create a `.env` file in the project root:

**On Windows:**
```cmd
copy .env.example .env
```

**On Linux/Mac:**
```bash
cp .env.example .env
```

If `.env.example` doesn't exist, create `.env` manually:

```env
# OpenWeatherMap API Configuration
WEATHER_API_KEY=your_openweathermap_api_key_here

# Application Configuration
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your_secret_key_here

# Optional: Database Configuration
DATABASE_URL=sqlite:///db/dhanya_assist.db
```

**🔗 Get your API key:**
1. Visit [OpenWeatherMap](https://openweathermap.org/api)
2. Sign up for a free account
3. Generate an API key
4. Replace `your_openweathermap_api_key_here` with your actual key

### 🗄️ 5. Initialize Database

```bash
python -c "from db.init_db import init_database; init_database()"
```

### ▶️ 6. Run the Application

```bash
# Start the Flask application
python app.py
```

The application will start on `http://localhost:5000`

## 🖥️ Platform-Specific Instructions

### 🪟 Windows Setup

```cmd
# Using Command Prompt
cd "C:\path\to\Dhanya-Assist"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

```powershell
# Using PowerShell
Set-Location "C:\path\to\Dhanya-Assist"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

### 🐧 Linux Setup

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip python3-venv
cd /path/to/Dhanya-Assist
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

### 🍎 macOS Setup

```bash
# Install Python via Homebrew (if not installed)
brew install python3

cd /path/to/Dhanya-Assist
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

## 🚀 Usage Guide

### 1. **Home Page**
- Access the main interface at `http://localhost:5000`
- Choose your preferred language (English/Hindi/Kannada)

### 2. **Crop Recommendation**
- Fill in soil parameters (N, P, K values)
- Enter climate data (temperature, humidity, rainfall)
- Get AI-powered crop suggestions

### 3. **Disease Detection**
- Upload plant/crop images (JPG, PNG, JPEG)
- Receive detailed disease analysis
- Get treatment recommendations

### 4. **Weather Information**
- Allow location access for real-time weather
- View farming-relevant weather insights
- Check forecasts and agricultural alerts

## 🔧 Configuration Options

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `WEATHER_API_KEY` | OpenWeatherMap API key | None | Yes |
| `FLASK_ENV` | Flask environment | `production` | No |
| `FLASK_DEBUG` | Debug mode | `False` | No |
| `SECRET_KEY` | Flask secret key | Auto-generated | No |
| `DATABASE_URL` | Database connection | `sqlite:///db/dhanya_assist.db` | No |

### Application Settings

Edit `app.py` to modify:
- Default language settings
- Image upload limits
- Model confidence thresholds
- Database configuration

## 🛠️ Development

### Project Structure

```
Dhanya-Assist/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables
├── README.md             # This file
├── data/                 # Dataset and data processing
│   ├── Crop_dataset.csv
│   └── process_dataset.py
├── db/                   # Database files and initialization
│   ├── init_db.py
│   └── dhanya_assist.db
├── model/                # ML models and training
│   └── crop_model.pkl
├── templates/            # HTML templates
│   └── index.html
├── translations/         # Language files
│   ├── en.json
│   ├── hi.json
│   └── kn.json
└── utils/               # Utility modules
    ├── ml_model.py      # Machine learning logic
    ├── vision.py        # Image processing
    ├── weather.py       # Weather API integration
    └── soil.py          # Soil analysis
```

### Adding New Features

1. **New Language Support**: Add JSON files to `translations/`
2. **Additional Models**: Place trained models in `model/`
3. **New APIs**: Create modules in `utils/`
4. **UI Changes**: Modify `templates/index.html`

## 🐛 Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'cv2'**
```bash
pip install opencv-python==4.8.1.78
# If still failing, the app will use fallback mode automatically
```

**2. NumPy compatibility errors**
```bash
pip uninstall numpy opencv-python -y
pip install "numpy<2.0" opencv-python==4.8.1.78
```

**3. Flask app not starting**
```bash
# Check if already running on port 5000
netstat -an | grep 5000
# Kill the process if needed
```

**4. Database initialization errors**
```bash
# Manually create database
python -c "from db.init_db import init_database; init_database()"
```

**5. API key issues**
- Verify your `.env` file exists and contains the correct API key
- Check API key validity at OpenWeatherMap dashboard
- Ensure no extra spaces in the API key

### Performance Issues

- **Slow image processing**: OpenCV will automatically fall back to basic analysis if not available
- **Memory usage**: Close other applications if experiencing slowdowns
- **Database locks**: Restart the application if database operations fail

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

If you encounter any issues:

1. Check this README for troubleshooting steps
2. Verify all dependencies are correctly installed
3. Ensure your Python version is compatible (3.8+)
4. Check that your API keys are valid and properly configured

## 🙏 Acknowledgments

- OpenWeatherMap for weather data API
- Scikit-learn for machine learning capabilities
- OpenCV for computer vision features
- Flask framework for web application structure

---

**Happy Farming! 🌾**
