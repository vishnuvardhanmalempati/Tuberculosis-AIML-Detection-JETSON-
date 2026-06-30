#!/bin/bash
# setup_jetson.sh
# Automates the system and library dependencies installation for NVIDIA Jetson Nano.

echo "================================================================"
echo "    NVIDIA JETSON NANO ENVIRONMENT SETUP - TUBERCULOSIS AI      "
echo "================================================================"

# 1. Install System Dependencies for OpenCV, SQLite, and Pillow
echo "[1/4] Installing system packages..."
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    libjpeg-dev \
    zlib1g-dev \
    libopenblas-dev \
    liblapack-dev \
    libsqlite3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev

# 2. Set up Python Virtual Environment
echo "[2/4] Initializing Python Virtual Environment (.venv)..."
# We inherit system site packages to leverage pre-installed NVIDIA modules (like cv2 or TensorRT)
python3 -m venv --system-site-packages .venv
source .venv/bin/activate

# 3. Upgrade pip and install standard packages
echo "[3/4] Installing standard python dependencies..."
pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib seaborn opencv-python reportlab onnx psutil streamlit onnxscript

# 4. Guide for Jetson-specific GPU acceleration wheels
echo "----------------------------------------------------------------"
echo " [4/4] JETSON NANO GPU WHEELS INSTALLATION INSTRUCTIONS"
echo "----------------------------------------------------------------"
echo "Official ARM64 PyTorch wheels must be installed from NVIDIA:"
echo "1. Download the PyTorch wheel for your JetPack version (e.g. JetPack 4.6 / Python 3.6):"
echo "   wget https://nvidia.box.com/shared/static/p57jw1qc47ax67vi328245s1x214eyii.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl"
echo "   pip install torch-1.10.0-cp36-cp36m-linux_aarch64.whl"
echo ""
echo "2. Install the matching Torchvision version from source:"
echo "   git clone --branch v0.11.1 https://github.com/pytorch/vision torchvision"
echo "   cd torchvision"
echo "   export BUILD_VERSION=0.11.1"
echo "   python setup.py install --user"
echo "   cd .."
echo ""
echo "3. For PyTorch-free GPU inference, install ONNX Runtime GPU:"
echo "   pip install onnxruntime-gpu"
echo "----------------------------------------------------------------"

echo "Setup script completed. Run 'source .venv/bin/activate' to begin."
echo "================================================================"
