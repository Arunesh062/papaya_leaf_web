# Papaya Leaf Disease Detection API

A Flask-based web application for detecting papaya leaf diseases using deep learning (ConvNeX) with AI-powered treatment advice.

## Features

- **Disease Detection**: Uses ConvNeX tiny model to classify 8 papaya leaf diseases (Anthracnose, Bacterial spot, Curl, Healthy, Mealybug, Mite disease, Ringspot, Mosaic)
- **AI Assistance**: Integrates Groq LLM for generating organic treatment recommendations
- **Image Enhancement**: Applies CLAHE and sharpening for better detection accuracy
- **Visualization**: GradCAM heatmaps to visualize model decision regions
- **SMS Notifications**: Send disease alerts via SMS
- **Test-Time Augmentation**: Enhanced prediction accuracy through multiple augmentations

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- Papaya leaf model file: `models/best_convnext_tiny.pth` (or `models/best_convnext_tiny.zip`)

## Installation

### Step 1: Clone the Repository
```bash
git clone <your-repo-url>
cd papaya_leaf_web
```

### Step 2: Create a Virtual Environment
```bash
# Using venv (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## Configuration

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```
GROQ_API_KEY=your_groq_api_key_here
SMS_API_KEY=your_sms_api_key_here (optional)
SMS_PHONE=your_phone_number (optional)
```

**To get a GROQ API key:**
1. Visit [Groq Console](https://console.groq.com)
2. Sign up/Log in
3. Create an API key
4. Add it to your `.env` file

## Running the Application

### Local Development Server

```bash
python app.py
```

The application will start on `http://localhost:5000`

### Production with Gunicorn

```bash
gunicorn -w 4 -b 0.0.0.0:8000 app.py
```

## API Endpoints

### 1. Home Page
- **URL**: `/`
- **Method**: GET
- **Description**: Serves the web interface (index.html)

### 2. Disease Prediction
- **URL**: `/predict`
- **Method**: POST
- **Parameters**: 
  - `file`: Image file (multipart/form-data)
- **Response**: JSON with disease classification and confidence scores

### Example cURL Request
```bash
curl -X POST -F "file=@leaf_image.jpg" http://localhost:5000/predict
```

## Project Structure

```
papaya_leaf_web/
├── app.py                    # Main Flask application
├── llm_advisor.py           # Groq LLM integration for treatment advice
├── sms_service.py           # SMS notification handler
├── model_engine.py          # Model utilities
├── check_model.py           # Model verification script
├── diagnose_model.py        # Model diagnosis tools
├── requirements.txt         # Python dependencies
├── models/
│   └── best_convnext_tiny.pth  # Trained model weights
├── static/
│   ├── css/style.css        # Frontend styles
│   └── js/script.js         # Frontend logic
└── templates/
    └── index.html           # Web interface
```

## Dependencies

- **torch**: Deep learning framework
- **torchvision**: Computer vision utilities
- **timm**: PyTorch Image Models
- **opencv-python-headless**: Image processing
- **flask**: Web framework
- **flask-cors**: Cross-origin support
- **google-generativeai**: AI services
- **groq**: Groq API client
- **Pillow**: Image processing
- **numpy**: Numerical computing
- **gunicorn**: Production WSGI server
- **python-dotenv**: Environment variable management

## Usage

1. Open your browser and go to `http://localhost:5000`
2. Upload a papaya leaf image
3. The model will:
   - Detect the disease type
   - Show confidence scores
   - Display GradCAM heatmap
   - Generate organic treatment advice in English, Tamil, and Hindi
4. Optionally, send SMS notification with the results

## Testing

Run the test scripts:

```bash
# Simple model test
python test_model_simple.py

# SMS service test
python test_sms.py

# Check model integrity
python check_model.py

# Diagnose model issues
python diagnose_model.py
```

## Disease Classes

The model can detect the following papaya leaf conditions:

1. **Anthracnose** - Fungal disease with dark spots
2. **Bacterial Spot** - Bacterial infection causing lesions
3. **Curl** - Leaf curling and distortion
4. **Healthy** - No disease detected
5. **Mealybug** - Pest infestation damage
6. **Mite Disease** - Mite pest damage
7. **Ringspot** - Viral disease with ring patterns
8. **Mosaic** - Viral mosaic pattern

## Troubleshooting

### Model Not Found
- Ensure `models/best_convnext_tiny.pth` exists
- If only `.zip` file exists, the app will auto-extract it
- Download the model file if missing

### GROQ_API_KEY Warning
- Check that `.env` file is created correctly
- Verify API key is valid and active
- Ensure `python-dotenv` is installed

### Import Errors
- Reinstall requirements: `pip install -r requirements.txt --force-reinstall`
- Verify Python version is 3.8+

### CUDA/GPU Issues
- The model runs on CPU by default
- For GPU support, install `torch` with CUDA support

## Performance Notes

- Test-Time Augmentation (TTA) ensures robust predictions but takes ~4x longer
- Image enhancement (CLAHE) improves accuracy for low-quality images
- GradCAM visualization helps interpret model decisions

## License

[Add your license information here]

## Support

For issues or questions, please create an issue on the repository.
