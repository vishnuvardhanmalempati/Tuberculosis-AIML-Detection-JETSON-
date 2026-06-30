import os
import sys
import time
from pathlib import Path
import numpy as np
import cv2

# We don't import PyTorch here to minimize memory footprint on the Jetson Nano
# (PyTorch import alone can consume 400MB+ RAM, which is critical on a 4GB shared-RAM Jetson Nano).

def preprocess_image_numpy(image_path):
    """
    Loads and preprocesses an image for MobileNetV3/EfficientNet using only OpenCV and NumPy.
    Replicates the PyTorch/Torchvision normalization pipeline.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
        
    # Read image using OpenCV (BGR format)
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
        
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to 224x224
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    
    # Normalize: Scale to [0.0, 1.0]
    img = img.astype(np.float32) / 255.0
    
    # ImageNet Mean and Standard Deviation
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    img = (img - mean) / std
    
    # HWC (Height, Width, Channels) to CHW (Channels, Height, Width)
    img = img.transpose(2, 0, 1)
    
    # Add batch dimension: [1, Channels, Height, Width]
    img = np.expand_dims(img, axis=0)
    
    return img

def sigmoid_numpy(x):
    """
    Computes sigmoid activation.
    """
    return 1 / (1 + np.exp(-x))

def run_edge_inference(onnx_model_path, image_path):
    """
    Runs inference on Jetson Nano using ONNX Runtime with GPU acceleration.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("[ERROR] onnxruntime not installed. Please install it using: pip install onnxruntime-gpu")
        sys.exit(1)
        
    onnx_model_path = Path(onnx_model_path)
    if not onnx_model_path.exists():
        raise FileNotFoundError(f"ONNX model file not found at {onnx_model_path}")
        
    # Initialize execution providers. 
    # For NVIDIA Jetson Nano, TensorrtExecutionProvider and CUDAExecutionProvider give GPU acceleration.
    providers = [
        'TensorrtExecutionProvider',
        'CUDAExecutionProvider',
        'CPUExecutionProvider'
    ]
    
    print(f"Initializing ONNX Runtime session with providers: {providers}")
    session_options = ort.SessionOptions()
    # Enable all optimizations
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Start Inference Session
    start_init = time.time()
    session = ort.InferenceSession(str(onnx_model_path), session_options, providers=providers)
    print(f"ONNX Model loaded in {time.time() - start_init:.4f} seconds.")
    print(f"Active Provider: {session.get_providers()[0]}")
    
    # Get inputs and outputs
    input_name = session.get_inputs()[0].name
    
    # Preprocess image
    print(f"Loading and preprocessing image: {image_path}")
    input_data = preprocess_image_numpy(image_path)
    
    # Warm up (recommended on Jetson Nano to stabilize GPU clock speeds)
    print("Warming up inference engine...")
    for _ in range(5):
        _ = session.run(None, {input_name: input_data})
        
    # Run benchmark
    print("Running diagnostic inference...")
    start_inf = time.time()
    outputs = session.run(None, {input_name: input_data})
    inference_time = (time.time() - start_inf) * 1000 # in ms
    
    # Postprocess output
    # Model output is raw logits shape: [1, 1]
    logit = outputs[0][0][0]
    probability = sigmoid_numpy(logit)
    
    label = "Tuberculosis" if probability >= 0.5 else "Normal"
    
    print("\n================== DIAGNOSTIC RESULT ==================")
    print(f"Result Label    : {label}")
    print(f"Probability     : {probability*100:.2f}%")
    print(f"Inference Time  : {inference_time:.2f} ms")
    print(f"Throughput      : {1000/inference_time:.1f} FPS")
    print("========================================================\n")
    
    return label, probability, inference_time

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/deploy_jetson.py <onnx_model_path> <image_path>")
        sys.exit(1)
        
    model_path = sys.argv[1]
    img_path = sys.argv[2]
    
    run_edge_inference(model_path, img_path)
