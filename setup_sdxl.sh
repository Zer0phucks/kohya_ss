#!/bin/bash
set -e

echo "================================================"
echo "  SDXL LoRA Training + Inference Setup"
echo "  DigitalOcean GPU Droplet"
echo "================================================"

# Update system
echo "[1/8] Updating system packages..."
sudo apt-get update -qq
sudo apt-get install -y git python3-pip python3-venv wget curl build-essential

# Install NVIDIA drivers and CUDA (if not present)
echo "[2/8] Checking NVIDIA drivers..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "Installing NVIDIA drivers..."
    sudo apt-get install -y nvidia-driver-535 nvidia-utils-535
fi

# Verify GPU
nvidia-smi || { echo "ERROR: GPU not detected"; exit 1; }

# Create workspace
echo "[3/8] Creating workspace..."
mkdir -p ~/sdxl-workspace
cd ~/sdxl-workspace

# Install ComfyUI for inference
echo "[4/8] Installing ComfyUI..."
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
deactivate

# Install Kohya_ss for LoRA training
echo "[5/8] Installing Kohya_ss LoRA trainer (with pre-loaded dataset)..."
cd ~/sdxl-workspace
git clone https://github.com/Zer0phucks/kohya_ss.git
cd kohya_ss
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
deactivate

echo "Linking pre-loaded dataset..."
mkdir -p ~/sdxl-workspace/lora-training
ln -s ~/sdxl-workspace/kohya_ss/dataset/images ~/sdxl-workspace/lora-training/dataset

# Download SDXL base model (standard)
echo "[6/8] Downloading SDXL base model..."
mkdir -p ~/sdxl-workspace/ComfyUI/models/checkpoints
cd ~/sdxl-workspace/ComfyUI/models/checkpoints
wget -c https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors

# Download uncensored SDXL model
echo "[7/8] Downloading uncensored SDXL model..."
wget -c https://civitai.com/api/download/models/351306 -O sdxl_uncensored.safetensors

# Create directories for LoRA training
echo "[8/8] Setting up LoRA training directories..."
mkdir -p ~/sdxl-workspace/lora-training/{dataset,output}
mkdir -p ~/sdxl-workspace/ComfyUI/models/loras

# Create convenience scripts
cat > ~/sdxl-workspace/start-comfyui.sh << 'EOF'
#!/bin/bash
cd ~/sdxl-workspace/ComfyUI
source venv/bin/activate
python main.py --listen 0.0.0.0 --port 8188
EOF

cat > ~/sdxl-workspace/start-kohya.sh << 'EOF'
#!/bin/bash
cd ~/sdxl-workspace/kohya_ss
source venv/bin/activate
python gui.py --listen 0.0.0.0 --server_port 7860
EOF

chmod +x ~/sdxl-workspace/start-*.sh

echo ""
echo "================================================"
echo "  ✓ Setup Complete!"
echo "================================================"
echo ""
echo "Installed components:"
echo "  • ComfyUI (inference): ~/sdxl-workspace/ComfyUI"
echo "  • Kohya_ss (training): ~/sdxl-workspace/kohya_ss"
echo "  • SDXL base model"
echo "  • SDXL uncensored model"
echo "  • Your custom dataset (pre-loaded from GitHub)"
echo ""
echo "To start services:"
echo "  ComfyUI:  bash ~/sdxl-workspace/start-comfyui.sh"
echo "  Kohya:    bash ~/sdxl-workspace/start-kohya.sh"
echo ""
echo "Access interfaces:"
echo "  ComfyUI:  http://YOUR_DROPLET_IP:8188"
echo "  Kohya:    http://YOUR_DROPLET_IP:7860"
echo ""
echo "Training workflow:"
echo "  1. Your dataset is already loaded at ~/sdxl-workspace/kohya_ss/dataset/images/"
echo "  2. Open Kohya at port 7860"
echo "  3. Train LoRA (outputs to ~/sdxl-workspace/lora-training/output/)"
echo "  4. Copy .safetensors to ~/sdxl-workspace/ComfyUI/models/loras/"
echo "  5. Use in ComfyUI with uncensored model"
echo ""
