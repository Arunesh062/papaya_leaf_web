import torch
import timm
import numpy as np
from torchvision import transforms
from PIL import Image
import os

# Model setup
MODEL_PATH = 'models/best_convnext_tiny.pth'
CLASSES = ["Anthracnose", "Bacterial spot", "Curl", "Healthy", "Mealybug", "Mite disease", "Ringspot", "Mosaic"]

device = torch.device("cpu")
model = timm.create_model('convnext_tiny', pretrained=False, num_classes=8)

print("=" * 60)
print("MODEL DIAGNOSTIC TEST")
print("=" * 60)

# Load model
if os.path.exists(MODEL_PATH):
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Model loaded successfully")
else:
    print(f"❌ Model not found at {MODEL_PATH}")
    exit(1)

# Test with random inputs
print("\n" + "=" * 60)
print("TEST 1: Random Input Predictions")
print("=" * 60)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Test with 5 different random images
for i in range(5):
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
    
    print(f"\nRandom Image {i+1}:")
    print(f"  Predicted: {CLASSES[predicted_idx]}")
    print(f"  Confidence: {confidence*100:.2f}%")
    print(f"  All probabilities:")
    for idx, prob in enumerate(probabilities):
        print(f"    {CLASSES[idx]}: {prob.item()*100:.2f}%")

# Test model weights statistics
print("\n" + "=" * 60)
print("TEST 2: Model Weight Statistics")
print("=" * 60)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Check if weights are actually different (not all zeros or all same)
first_layer_weights = None
for name, param in model.named_parameters():
    if 'weight' in name:
        first_layer_weights = param.data.cpu().numpy().flatten()
        print(f"\nFirst layer: {name}")
        print(f"  Shape: {param.shape}")
        print(f"  Mean: {first_layer_weights.mean():.6f}")
        print(f"  Std: {first_layer_weights.std():.6f}")
        print(f"  Min: {first_layer_weights.min():.6f}")
        print(f"  Max: {first_layer_weights.max():.6f}")
        break

# Check final classifier layer
print("\n" + "=" * 60)
print("TEST 3: Final Classifier Layer")
print("=" * 60)

for name, module in model.named_modules():
    if 'head' in name or 'fc' in name or 'classifier' in name:
        print(f"Found classifier: {name}")
        print(f"  Module: {module}")
        if hasattr(module, 'weight'):
            print(f"  Weight shape: {module.weight.shape}")
            print(f"  Bias shape: {module.bias.shape if module.bias is not None else 'No bias'}")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
