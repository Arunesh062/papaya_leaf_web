import torch
import torch.nn.functional as F
import timm
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import os

class LeafAnalyzer:
    def __init__(self, model_path, labels):
        self.labels = labels
        # Load ConvNeXt Tiny Architecture
        self.model = timm.create_model('convnext_tiny', pretrained=False, num_classes=len(labels))
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        
        # Hooks for Grad-CAM Heatmap
        self.gradients = None
        self.activations = None

    def save_gradient(self, grad): self.gradients = grad
    def save_activation(self, act): self.activations = act

    def run_inference(self, image_path):
        # 1. Prepare Image
        raw_img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        
        input_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])(img_resized).unsqueeze(0)

        # 2. Set Up Heatmap Hooks
        target_layer = self.model.stages[-1].blocks[-1]
        h1 = target_layer.register_forward_hook(lambda m, i, o: self.save_activation(o))
        h2 = target_layer.register_full_backward_hook(lambda m, i, o: self.save_gradient(o[0]))

        # 3. Predict
        output = self.model(input_tensor)
        idx = torch.argmax(output, dim=1).item()
        confidence = F.softmax(output, dim=1)[0][idx].item()

        # 4. Generate Heatmap Math
        self.model.zero_grad()
        output[0, idx].backward()
        
        # Pull hooks
        h1.remove()
        h2.remove()

        # 5. Create the Visual Heatmap (Red/Yellow/Green)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze().detach().numpy()
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (raw_img.shape[1], raw_img.shape[0]))
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        result_overlay = cv2.addWeighted(raw_img, 0.6, heatmap, 0.4, 0)
        
        # Save output
        save_name = "heat_" + os.path.basename(image_path)
        save_path = os.path.join("static/heatmaps", save_name)
        cv2.imwrite(save_path, result_overlay)
        
        return self.labels[idx], round(confidence * 100, 1), save_name