import os
import subprocess
import shutil
from pathlib import Path
import numpy as np
import torch

def export_pytorch_to_onnx(model_path, model_name, onnx_save_path, device="cpu"):
    """
    Exports a trained PyTorch model to ONNX format.
    """
    from src.model import get_tuberculosis_model
    
    model_path = Path(model_path)
    onnx_save_path = Path(onnx_save_path)
    onnx_save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not model_path.exists():
        raise FileNotFoundError(f"PyTorch model not found at {model_path}")
        
    print(f"Loading PyTorch model weights from {model_path}...")
    model = get_tuberculosis_model(model_name, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Define dummy input matching shape [batch_size, channels, height, width]
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    
    print(f"Exporting model to ONNX at {onnx_save_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_save_path),
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("ONNX export completed successfully.")
    
    # Verify the exported ONNX model matches PyTorch outputs
    verify_onnx_model(model, onnx_save_path, dummy_input)
    
    return onnx_save_path

def verify_onnx_model(pytorch_model, onnx_path, dummy_input):
    """
    Verifies that the ONNX model predictions match the PyTorch model predictions.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("[WARNING] onnxruntime not installed. Skipping output verification.")
        return
        
    # PyTorch inference
    with torch.no_grad():
        pytorch_out = pytorch_model(dummy_input).cpu().numpy()
        
    # ONNX inference
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    onnx_input = dummy_input.cpu().numpy()
    onnx_out = session.run(None, {session.get_inputs()[0].name: onnx_input})[0]
    
    # Check difference
    diff = np.abs(pytorch_out - onnx_out).max()
    print(f"Inference output validation: Maximum absolute difference is {diff:.6f}")
    if diff < 1e-4:
        print("[SUCCESS] PyTorch and ONNX models produce identical outputs.")
    else:
        print("[WARNING] Significant difference between PyTorch and ONNX outputs detected.")

def convert_onnx_to_tensorrt(onnx_path, trt_engine_path, fp16=True):
    """
    Compiles the ONNX model to a TensorRT engine.
    Tries using the command-line compiler `trtexec`, which is standard in NVIDIA environments.
    """
    onnx_path = Path(onnx_path)
    trt_engine_path = Path(trt_engine_path)
    trt_engine_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found at {onnx_path}")
        
    # 1. Search for trtexec in PATH
    trtexec_bin = shutil.which("trtexec")
    if trtexec_bin is None:
        # Check standard Jetson Nano paths
        possible_paths = [
            "/usr/src/tensorrt/bin/trtexec",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\TensorRT\\bin\\trtexec.exe"
        ]
        for p in possible_paths:
            if os.path.exists(p):
                trtexec_bin = p
                break
                
    if trtexec_bin is None:
        print("\n[INFO] 'trtexec' not found in PATH or standard directories.")
        print("To convert the ONNX model to TensorRT manually on your Jetson Nano, run:")
        print(f"trtexec --onnx={onnx_path} --saveEngine={trt_engine_path} --fp16")
        return False
        
    print(f"Found trtexec compiler: {trtexec_bin}")
    print(f"Compiling ONNX model to TensorRT engine...")
    
    command = [
        trtexec_bin,
        f"--onnx={onnx_path}",
        f"--saveEngine={trt_engine_path}"
    ]
    if fp16:
        command.append("--fp16")
        
    print(f"Executing: {' '.join(command)}")
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        print("TensorRT compilation completed successfully.")
        print(f"Engine saved to {trt_engine_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] TensorRT compilation failed: {e}")
        print(f"Compiler Stderr: {e.stderr}")
        return False
        
if __name__ == "__main__":
    # Test script with arguments
    import sys
    if len(sys.argv) > 3:
        export_pytorch_to_onnx(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Usage: python src/optimize.py <pytorch_model_path> <model_name> <onnx_output_path>")
