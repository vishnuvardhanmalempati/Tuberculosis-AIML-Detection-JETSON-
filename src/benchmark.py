import time
import os
import psutil
from pathlib import Path
import numpy as np
import torch

def get_ram_usage():
    """
    Returns the current process memory usage in MB.
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def benchmark_pytorch(model, input_tensor, device, num_runs=50, warmup_runs=10):
    """
    Benchmarks PyTorch model inference latency and memory.
    """
    model.eval()
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
            
    # Measure VRAM before run
    vram_before = 0.0
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        vram_before = torch.cuda.memory_allocated() / (1024 * 1024)
        
    start_time = time.time()
    latencies = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            run_start = time.time()
            _ = model(input_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            latencies.append((time.time() - run_start) * 1000) # in ms
            
    total_time = time.time() - start_time
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    throughput = num_runs / total_time
    
    # Measure VRAM after run
    vram_peak = 0.0
    if device.type == 'cuda':
        vram_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
    return {
        "device": str(device),
        "avg_latency_ms": avg_latency,
        "std_latency_ms": std_latency,
        "throughput_fps": throughput,
        "vram_peak_mb": vram_peak - vram_before if device.type == 'cuda' else 0.0,
        "ram_process_mb": get_ram_usage()
    }

def benchmark_onnx(onnx_path, input_tensor, num_runs=50, warmup_runs=10):
    """
    Benchmarks ONNX model inference latency and memory using ONNX Runtime.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("[WARNING] onnxruntime not installed. Skipping ONNX benchmarking.")
        return None
        
    if not Path(onnx_path).exists():
        print(f"[WARNING] ONNX model file not found at {onnx_path}. Skipping.")
        return None
        
    # Check if GPU is available in ONNX Runtime
    providers = ort.get_available_providers()
    selected_provider = 'CPUExecutionProvider'
    
    # Prioritize CUDA if available
    if 'CUDAExecutionProvider' in providers:
        selected_provider = 'CUDAExecutionProvider'
        
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(str(onnx_path), session_options, providers=[selected_provider])
    input_name = session.get_inputs()[0].name
    
    # ONNX Runtime input is a numpy array
    input_numpy = input_tensor.numpy()
    
    # Warmup
    for _ in range(warmup_runs):
        _ = session.run(None, {input_name: input_numpy})
        
    start_time = time.time()
    latencies = []
    
    for _ in range(num_runs):
        run_start = time.time()
        _ = session.run(None, {input_name: input_numpy})
        latencies.append((time.time() - run_start) * 1000) # in ms
        
    total_time = time.time() - start_time
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    throughput = num_runs / total_time
    
    return {
        "device": f"ONNX Runtime ({selected_provider})",
        "avg_latency_ms": avg_latency,
        "std_latency_ms": std_latency,
        "throughput_fps": throughput,
        "vram_peak_mb": 0.0, # ONNX runtime direct VRAM tracking not easily accessible in python standard API
        "ram_process_mb": get_ram_usage()
    }

def run_benchmarks(pytorch_model=None, model_name="mobilenet_v3", pth_path=None, onnx_path=None):
    """
    Runs CPU, GPU (if available), and ONNX benchmarks and prints results.
    """
    print("\n================== BENCHMARKING MODELS ==================")
    input_tensor = torch.randn(1, 3, 224, 224)
    results = {}
    
    # 1. Load PyTorch model if path provided
    if pytorch_model is None and pth_path is not None and Path(pth_path).exists():
        from src.model import get_tuberculosis_model
        pytorch_model = get_tuberculosis_model(model_name, pretrained=False)
        pytorch_model.load_state_dict(torch.load(pth_path, map_location='cpu'))
        
    if pytorch_model is not None:
        # Benchmark PyTorch CPU
        print("Benchmarking PyTorch on CPU...")
        res_cpu = benchmark_pytorch(pytorch_model, input_tensor, torch.device('cpu'))
        results['pytorch_cpu'] = res_cpu
        
        # Benchmark PyTorch GPU
        if torch.cuda.is_available():
            print("Benchmarking PyTorch on GPU...")
            res_gpu = benchmark_pytorch(pytorch_model, input_tensor, torch.device('cuda'))
            results['pytorch_gpu'] = res_gpu
        else:
            print("CUDA GPU not available. Skipping PyTorch GPU benchmarking.")
            
    # 2. Benchmark ONNX model
    if onnx_path is not None and Path(onnx_path).exists():
        print("Benchmarking ONNX Runtime...")
        res_onnx = benchmark_onnx(onnx_path, input_tensor)
        if res_onnx:
            results['onnx'] = res_onnx
            
    # Print results summary
    print("\n------------------ BENCHMARK RESULTS ------------------")
    print(f"{'Engine/Device':35s} | {'Latency (ms)':12s} | {'Throughput (FPS)':16s} | {'RAM (MB)':10s}")
    print("-" * 83)
    for name, metrics in results.items():
        print(f"{metrics['device']:35s} | {metrics['avg_latency_ms']:9.2f} ms  | {metrics['throughput_fps']:12.2f} FPS     | {metrics['ram_process_mb']:7.1f} MB")
    print("========================================================\n")
    
    return results

if __name__ == "__main__":
    # Test script with dummy model
    from src.model import get_tuberculosis_model
    model = get_tuberculosis_model("mobilenet_v3", pretrained=False)
    run_benchmarks(pytorch_model=model)
