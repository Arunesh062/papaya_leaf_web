import torch
import timm
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import sys

# Model setup
MODEL_PATH = 'models/best_convnext_tiny.pth'
CLASSES = ["Anthracnose", "Bacterial spot", "Curl", "Healthy", "Mealybug", "Mite disease", "Ringspot", "Mosaic"]

device = torch.device("cpu")
model = timm.create_model('convnext_tiny', pretrained=False, num_classes=8)

print("Loading model...")
if os.path.exists(MODEL_PATH):
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully\n")
else:
    print(f"ERROR: Model not found at {MODEL_PATH}")
    sys.exit(1)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("Testing with 3 random images:")
print("=" * 60)

for i in range(3):
    # Create random RGB image
    random_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    random_img = Image.fromarray(random_array)
    
    # Transform and predict
    input_tensor = transform(random_img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item()
    
    print(f"\nTest {i+1}:")
    print(f"  Predicted: {CLASSES[predicted_idx]} ({confidence*100:.1f}%)")
    
    # Show top 3 predictions
    top3_probs, top3_indices = torch.topk(probabilities, 3)
    print(f"  Top 3:")
    for j in range(3):
        print(f"    {j+1}. {CLASSES[top3_indices[j]]}: {top3_probs[j].item()*100:.1f}%")

print("\n" + "=" * 60)
print("If all predictions are the same class, the model has an issue!")
