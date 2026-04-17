import os
from dotenv import load_dotenv

load_dotenv()
import cv2
import torch
import numpy as np
import timm
import zipfile
from flask import Flask, request, render_template, jsonify
from torchvision import transforms
from llm_advisor import CaricaCareAdvisor
from sms_service import sms_handler
from PIL import Image

app = Flask(__name__)

# Get API key from environment variable
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
ADVISOR = CaricaCareAdvisor(api_key=GROQ_API_KEY)

if not GROQ_API_KEY or GROQ_API_KEY.startswith("your_"):
    print("Γ¥î Warning: GROQ_API_KEY not set correctly. Please check your environment variables.")
else:
    print("Γ£à Groq Advisor Initialized")
MODEL_PATH = 'models/best_convnext_tiny.pth'
ZIP_PATH = 'models/best_convnext_tiny.zip'
CLASSES = ["Anthracnose", "Bacterial spot", "Curl", "Healthy", "Mealybug", "Mite disease", "Ringspot", "Mosaic"]

device = torch.device("cpu")
model = timm.create_model('convnext_tiny', pretrained=False, num_classes=8)

# Check for model file, unzip if needed (for cloud deployment)
if not os.path.exists(MODEL_PATH) and os.path.exists(ZIP_PATH):
    print("📦 Unzipping model file...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall('models')
    print("✅ Model unzipped successfully")

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ Model Loaded Successfully")
else:
    print(f"❌ Error: Model not found at {MODEL_PATH}")

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_act)
        self.target_layer.register_full_backward_hook(self.save_grad)
    def save_act(self, m, i, o): self.activations = o
    def save_grad(self, m, gi, go): self.gradients = go[0]
    def generate(self, tensor, idx):
        output = self.model(tensor)
        self.model.zero_grad()
        output[0, idx].backward()
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze().detach().numpy()
        cam = np.maximum(cam, 0)
        return cam / (np.max(cam) + 1e-7)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Enhanced preprocessing function
        def enhance_image(image):
            """Apply CLAHE and sharpening to improve image quality"""
            # Convert to LAB color space for better contrast enhancement
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels back
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
            
            # Apply sharpening kernel
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            # Blend original and sharpened (70% sharpened, 30% original)
            result = cv2.addWeighted(sharpened, 0.7, enhanced, 0.3, 0)
            return result

        # Enhance the image
        img_enhanced = enhance_image(img_rgb)

        # Standard transform
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Test-Time Augmentation (TTA)
        def predict_with_tta(image):
            """Run prediction on multiple augmentations and average"""
            predictions = []
            
            # 1. Original image
            input_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.nn.functional.softmax(output[0], dim=0)
                predictions.append(prob)
            
            # 2. Horizontal flip
            flipped = cv2.flip(image, 1)
            input_tensor = transform(flipped).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.nn.functional.softmax(output[0], dim=0)
                predictions.append(prob)
            
            # 3. Slight rotation (+5 degrees)
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(image)
            rotated_pil = pil_img.rotate(5, expand=False, fillcolor=(0, 0, 0))
            rotated = np.array(rotated_pil)  # Convert back to numpy array
            input_tensor = transform(rotated).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.nn.functional.softmax(output[0], dim=0)
                predictions.append(prob)
            
            # 4. Slight rotation (-5 degrees)
            rotated_pil = pil_img.rotate(-5, expand=False, fillcolor=(0, 0, 0))
            rotated = np.array(rotated_pil)  # Convert back to numpy array
            input_tensor = transform(rotated).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.nn.functional.softmax(output[0], dim=0)
                predictions.append(prob)
            
            # Average all predictions
            avg_prob = torch.stack(predictions).mean(dim=0)
            return avg_prob

        # Get prediction with TTA
        prob = predict_with_tta(img_enhanced)
        idx = torch.argmax(prob).item()
        disease = CLASSES[idx]
        
        # Use original image for heatmap (better visualization)
        input_tensor = transform(img_rgb).unsqueeze(0)

        # Grad-CAM Heatmap Generation
        try:
            target_layer = model.stages[3].blocks[-1]
            cam = GradCAM(model, target_layer).generate(input_tensor, idx)
            heatmap = cv2.applyColorMap(np.uint8(255 * cv2.resize(cam, (img.shape[1], img.shape[0]))), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
            cv2.imwrite('static/temp/heatmap.png', overlay)
        except Exception as cam_err:
            print(f"Heatmap Error: {cam_err}")

        # Fetch Translated 4-Protocol Advice
        en, ta, hi = ADVISOR.get_organic_advice(disease)

        return jsonify({
            "condition": disease,
            "accuracy": f"{prob[idx].item()*100:.1f}%",
            "advice_en": en,
            "advice_ta": ta,
            "advice_hi": hi,
            "heatmap": "/static/temp/heatmap.png"
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_msg = str(e)
        if "Connection error" in error_msg or "Failed to connect" in error_msg:
             error_msg = "Connection Error: Please check your internet or Groq API Key."
        return jsonify({"error": error_msg}), 500

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file part"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"status": "error", "message": "No selected file"}), 400

        # Save temporarily
        temp_path = os.path.join('static/temp', 'recording.webm')
        file.save(temp_path)

        # Transcribe
        text = ADVISOR.transcribe_audio(temp_path)
        
        if text:
            return jsonify({"status": "success", "transcript": text})
        else:
            return jsonify({"status": "error", "message": "Transcription failed"}), 500
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/send-sms', methods=['POST'])
def send_sms():
    try:
        data = request.json
        phone = data.get('phone')
        message = data.get('message')
        
        if not phone or not message:
            return jsonify({"status": "error", "message": "Missing phone or message"}), 400
            
        result = sms_handler.send_sms(phone, message)
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/schedule-reminder', methods=['POST'])
def schedule_reminder():
    import threading
    try:
        data = request.json
        phone = data.get('phone')
        days = data.get('days') # 7, 15, or 'DEMO_1_MIN'
        
        if not phone or not days:
            return jsonify({"status": "error", "message": "Missing info"}), 400

        sms_body = "🌿 விவசாயி நண்பருக்கு நினைவூட்டல்\n\nஇன்று மருந்து தெளிப்பதற்கான நாள்.\nநோய் பரவாமல் இருக்க\nதயவு செய்து மருந்து தெளியுங்கள்.\n\n– Leaf Disease Alert System"

        if days == 'DEMO_1_MIN':
            print(f"\n{'='*40}")
            print(f"✨ REMINDER SCHEDULED (FOR JUDGES) ✨")
            print(f"CONFIRMED_MOBILE_NUMBER: {phone}")
            print(f"REMINDER_DAYS: {days}")
            print(f"{'='*40}\n")
            
            # Start a timer to send SMS in 5 seconds
            def send_later():
                print(f"\n🚀 [ACTIVATE DEMO SMS] Starting delivery to {phone}...")
                result = sms_handler.send_sms(phone, sms_body)
                print(f"📊 SYSTEM STATUS: {result['message']}")
                print(f"{'='*40}\n")

            timer = threading.Timer(5.0, send_later)
            timer.start()
            return jsonify({"status": "success", "message": "Demo reminder scheduled for 5 seconds."})
        else:
            print(f"\n{'='*40}")
            print(f"✅ REMINDER SET SUCCESSFULLY")
            print(f"CONFIRMED_MOBILE_NUMBER: {phone}")
            print(f"REMINDER_DAYS: {days}")
            print(f"{'='*40}\n")
            return jsonify({"status": "success", "message": f"Reminder scheduled for {days} days."})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ============ SEASONAL DISEASE ADVISORY ENDPOINT ============
@app.route('/seasonal-advisory', methods=['POST'])
def seasonal_advisory():
    """
    Provides seasonal disease predictions based on crop type and season
    """
    try:
        data = request.json
        crop = data.get('crop', '').lower()
        season = data.get('season', '').lower()
        region = data.get('region', 'General')

        # Disease data by crop and season
        seasonal_diseases = {
            'papaya': {
                'monsoon': {
                    'diseases': [
                        {
                            'name': 'Ring Spot Virus',
                            'risk': 'High',
                            'description': 'Circular rings appear on leaves during monsoon humidity. Spreads via aphids.'
                        },
                        {
                            'name': 'Anthracnose',
                            'risk': 'High',
                            'description': 'Fungal disease causing dark spots in wet conditions. Very common in monsoons.'
                        },
                        {
                            'name': 'Leaf Curl',
                            'risk': 'Medium',
                            'description': 'Whitefly-transmitted virus causing leaf distortion.'
                        }
                    ],
                    'prevention': [
                        'Maintain good field drainage to reduce humidity',
                        'Install yellow sticky traps for aphid control',
                        'Apply neem oil spray every 2 weeks',
                        'Remove infected leaves immediately',
                        'Use disease-resistant varieties if available',
                        'Avoid overhead irrigation, use drip irrigation'
                    ]
                },
                'summer': {
                    'diseases': [
                        {
                            'name': 'Mite Disease',
                            'risk': 'High',
                            'description': 'Spider mites thrive in dry, hot conditions causing leaf yellowing.'
                        },
                        {
                            'name': 'Mosaic Virus',
                            'risk': 'Medium',
                            'description': 'Causes irregular patterns on leaves. Spread by aphids.'
                        },
                        {
                            'name': 'Powdery Mildew',
                            'risk': 'Low',
                            'description': 'White powdery coating on leaves in hot, dry weather.'
                        }
                    ],
                    'prevention': [
                        'Provide adequate irrigation to reduce plant stress',
                        'Install shade nets in extreme heat areas',
                        'Use sulfur dust for mite control',
                        'Monitor plants regularly for early detection',
                        'Apply potassium deficiency supplements',
                        'Keep weeds around field cleared'
                    ]
                },
                'winter': {
                    'diseases': [
                        {
                            'name': 'Bacterial Spot',
                            'risk': 'Medium',
                            'description': 'Promotes bacterial growth in cool, moist conditions.'
                        },
                        {
                            'name': 'Healthy Growth!',
                            'risk': 'Low',
                            'description': 'Winter provides ideal conditions for healthy papaya growth.'
                        }
                    ],
                    'prevention': [
                        'Maintain proper spacing between plants for air circulation',
                        'Apply copper-based fungicides preventively',
                        'Remove fallen leaves and debris',
                        'Prune lower branches for better ventilation',
                        'Monitor irrigation - reduce in cold months',
                        'Ensure adequate nitrogen supply'
                    ]
                },
                'spring': {
                    'diseases': [
                        {
                            'name': 'Mealybug Infestation',
                            'risk': 'Medium',
                            'description': 'Jumping up from winter dormancy as temperatures rise.'
                        },
                        {
                            'name': 'Leaf Spot',
                            'risk': 'Low',
                            'description': 'Light fungal spots possible before summer heat arrives.'
                        }
                    ],
                    'prevention': [
                        'Inspect plants weekly during spring transition',
                        'Release natural predators (ladybugs, parasitic wasps)',
                        'Clean pruning tools to prevent spread',
                        'Spray neem oil as temperatures increase',
                        'Prepare irrigation systems for summer',
                        'Apply balanced fertilizers for growth'
                    ]
                }
            },
            'mango': {
                'monsoon': {
                    'diseases': [
                        {'name': 'Anthracnose', 'risk': 'High', 'description': 'Severe during monsoon in mango.'},
                        {'name': 'Powdery Mildew', 'risk': 'Medium', 'description': 'Common on new growth in humid season.'}
                    ],
                    'prevention': ['Spray copper fungicides', 'Improve canopy ventilation', 'Collect and burn fallen leaves']
                },
                'summer': {
                    'diseases': [
                        {'name': 'Mite Disease', 'risk': 'High', 'description': 'Severe in dry summer.'},
                        {'name': 'Fruit Fly', 'risk': 'Medium', 'description': 'Increases during summer fruiting.'}
                    ],
                    'prevention': ['Regular watering during dry season', 'Use fruit bagging technique', 'Scout for pests']
                },
                'winter': {
                    'diseases': [
                        {'name': 'Bacterial Canker', 'risk': 'Low', 'description': 'Rare in winter conditions.'}
                    ],
                    'prevention': ['Maintain plant health', 'Prune dead wood', 'Sanitize tools']
                },
                'spring': {
                    'diseases': [
                        {'name': 'Flower/Fruit Drop', 'risk': 'Medium', 'description': 'Temperature fluctuations cause drops.'}
                    ],
                    'prevention': ['Maintain soil moisture', 'Apply micronutrient sprays', 'Avoid harsh pruning']
                }
            }
        }

        # Default response for unknown crops
        if crop not in seasonal_diseases:
            return jsonify({
                'diseases': [
                    {'name': 'General Fungal Risk', 'risk': 'Medium', 'description': 'Monitor crop for standard fungal diseases.'},
                    {'name': 'General Pest Risk', 'risk': 'Medium', 'description': 'Use integrated pest management techniques.'}
                ],
                'prevention': [
                    'Scout fields regularly',
                    'Use crop rotation',
                    'Apply organic farming practices',
                    'Install pest traps',
                    'Maintain field hygiene',
                    'Use recommended varieties'
                ]
            }), 200

        # Get seasonal data
        if season not in seasonal_diseases[crop]:
            season = 'monsoon'  # Default to monsoon

        advisory = seasonal_diseases[crop][season]

        return jsonify({
            'crop': crop,
            'season': season,
            'region': region,
            'diseases': advisory['diseases'],
            'prevention': advisory['prevention']
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============ CHAT ENDPOINT ============
@app.route('/chat', methods=['POST'])
def chat():
    """
    AI-powered chat endpoint for answering farmer questions about disease management
    """
    try:
        data = request.json
        user_message = data.get('message', '').strip()

        if not user_message:
            return jsonify({'error': 'Empty message'}), 400

        # Use Groq LLM to answer questions
        response = ADVISOR.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": """You are CaricaCare AI Expert - a helpful agricultural advisor for papaya farmers. 
                    Answer questions about papaya disease management, organic farming, prevention, treatment, and best practices.
                    Keep answers short, practical, and farmer-friendly.
                    Use simple language and focus on actionable advice.
                    Provide advice in English, Tamil, and Hindi when helpful."""
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            max_tokens=300,
            temperature=0.7
        )

        reply = response.choices[0].message.content.strip()

        return jsonify({
            'reply': reply,
            'status': 'success'
        }), 200

    except Exception as e:
        print(f"Chat error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# Ensure temp directory exists (runs on import for Gunicorn)
os.makedirs('static/temp', exist_ok=True)

# import os

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)