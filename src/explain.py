import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) implementation 
    for visual explanations of CNN decisions.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.embeddings = None
        self.handlers = []
        
        # Register forward and backward hooks
        self.handlers.append(target_layer.register_forward_hook(self.save_embeddings))
        if hasattr(target_layer, 'register_full_backward_hook'):
            self.handlers.append(target_layer.register_full_backward_hook(self.save_gradients))
        else:
            self.handlers.append(target_layer.register_backward_hook(self.save_gradients))
            
    def save_embeddings(self, module, input, output):
        self.embeddings = output
        
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, input_tensor):
        """
        Generates the Grad-CAM heatmap for the input tensor.
        """
        # Reset gradients
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # For binary classification, backward on the single output logit
        logit = output[0, 0]
        logit.backward()
        
        # Check if gradients were captured
        if self.gradients is None or self.embeddings is None:
            raise RuntimeError("Gradients or activations were not captured. Check if hooks are registered correctly.")
            
        grads = self.gradients.cpu().data.numpy()[0]
        embs = self.embeddings.cpu().data.numpy()[0]
        
        # Calculate weights: Global Average Pooling of gradients
        weights = np.mean(grads, axis=(1, 2))
        
        # Linear combination of channels
        cam = np.zeros(embs.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * embs[i, :, :]
            
        # Apply ReLU
        cam = np.maximum(cam, 0)
        
        # Resize to 224x224
        cam = cv2.resize(cam, (224, 224))
        
        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
            
        return cam
        
    def remove_hooks(self):
        for h in self.handlers:
            h.remove()

def get_gradcam_target_layer(model, model_name="mobilenet_v3"):
    """
    Returns the target layer for hooks based on the model architecture.
    """
    model_name = model_name.lower()
    if model_name == "mobilenet_v3":
        return model.features[-1]
    elif model_name == "efficientnet_b0":
        return model.features[-1]
    else:
        raise ValueError(f"Unknown target layer for model: {model_name}")

def overlay_heatmap(image_path, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Loads original image and overlays the Grad-CAM heatmap.
    """
    original_img = cv2.imread(str(image_path))
    if original_img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
        
    h, w, c = original_img.shape
    
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = (heatmap_resized * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, colormap)
    
    overlayed = cv2.addWeighted(original_img, 1 - alpha, heatmap_colored, alpha, 0)
    overlayed_rgb = cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB)
    
    return overlayed_rgb

def generate_and_save_gradcam(model, model_name, image_path, save_path, device="cpu"):
    """
    Generates Grad-CAM for a single X-ray image and saves the overlaid result.
    Temporarily overrides frozen layers to enable gradient calculations.
    """
    img = Image.open(image_path).convert('RGB')
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    
    # 1. Temporarily save require_grad status and set all to True to allow gradient computation
    # (Since transfer learning froze the backbone, we need requires_grad=True to calculate Grad-CAM gradients).
    original_requires_grad = {}
    for name, param in model.named_parameters():
        original_requires_grad[name] = param.requires_grad
        param.requires_grad = True
        
    # Ensure model is in eval mode
    model.eval()
    
    # 2. Get target layer
    target_layer = get_gradcam_target_layer(model, model_name)
    
    # 3. Compute Grad-CAM with gradients temporarily enabled (in case caller disabled them)
    with torch.enable_grad():
        cam_generator = GradCAM(model, target_layer)
        try:
            heatmap = cam_generator(input_tensor)
        finally:
            # Crucial: clean up hooks
            cam_generator.remove_hooks()
            
            # Restore original requires_grad status to preserve frozen state
            for name, param in model.named_parameters():
                param.requires_grad = original_requires_grad[name]
        
    # 4. Create overlay and save
    overlayed_img = overlay_heatmap(image_path, heatmap)
    result_img = Image.fromarray(overlayed_img)
    result_img.save(save_path)
    print(f"Grad-CAM overlay saved to {save_path}")
    
    return heatmap, save_path
