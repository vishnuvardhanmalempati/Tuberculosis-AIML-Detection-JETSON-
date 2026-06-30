import torch
import torch.nn as nn
import torchvision.models as models

def get_tuberculosis_model(model_name="mobilenet_v3", pretrained=True, freeze_backbone=True):
    """
    Returns a PyTorch model (MobileNetV3-Large or EfficientNet-B0) modified for 
    binary classification (Normal vs Tuberculosis).
    
    freeze_backbone: If True, freezes the convolutional layers of the model, 
                     which is essential to prevent overfitting on small datasets.
    """
    model_name = model_name.lower()
    
    if model_name == "mobilenet_v3":
        try:
            # Modern torchvision syntax
            from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
            weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            model = mobilenet_v3_large(weights=weights)
        except ImportError:
            # Legacy torchvision syntax fallback
            model = models.mobilenet_v3_large(pretrained=pretrained)
            
        # Freeze backbone parameters
        if pretrained and freeze_backbone:
            print("[INFO] Freezing MobileNetV3 backbone features.")
            for param in model.features.parameters():
                param.requires_grad = False
                
        in_features = model.classifier[0].in_features
        
        # Replace the classifier block with a custom binary classifier
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(256, 1) # Output logits
        )
        
    elif model_name == "efficientnet_b0":
        try:
            # Modern torchvision syntax
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = efficientnet_b0(weights=weights)
        except ImportError:
            # Legacy torchvision syntax fallback
            model = models.efficientnet_b0(pretrained=pretrained)
            
        # Freeze backbone parameters
        if pretrained and freeze_backbone:
            print("[INFO] Freezing EfficientNet-B0 backbone features.")
            for param in model.features.parameters():
                param.requires_grad = False
                
        in_features = model.classifier[1].in_features
        
        # Replace the classifier block
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, 1) # Output logits
        )
        
    else:
        raise ValueError(f"Unsupported model name: {model_name}. Choose 'mobilenet_v3' or 'efficientnet_b0'.")
        
    return model

if __name__ == "__main__":
    model = get_tuberculosis_model("mobilenet_v3", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print("Model output shape:", out.shape)
