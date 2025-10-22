# Windows GPU Setup Guide

Quick guide for setting up DeepSeek OCR on Windows with NVIDIA GPU.

## Prerequisites

### Required:
- Windows 10/11
- Python 3.9 or higher ([Download](https://www.python.org/downloads/))
- NVIDIA GPU with CUDA support
- At least 8GB GPU VRAM (recommended: 12GB+)
- CUDA 11.8 ([Download](https://developer.nvidia.com/cuda-11-8-0-download-archive))

### Check Your Setup:

Open PowerShell or Command Prompt and run:

```bash
# Check Python
python --version

# Check NVIDIA GPU
nvidia-smi
```

## Quick Setup (Automated)

### 1. Clone the Repository

```bash
git clone https://github.com/AntonioDmEPM/deepseek-ocr-app.git
cd deepseek-ocr-app
```

### 2. Run the Setup Script

Simply double-click `setup_windows.bat` or run:

```bash
setup_windows.bat
```

This will:
- Create a virtual environment
- Install PyTorch with CUDA 11.8
- Install all GPU dependencies
- Create configuration file

### 3. Configure for GPU

Edit `.env` file and ensure:

```ini
DEVICE_MODE=gpu
USE_FLASH_ATTENTION=true
```

### 4. Download the Model (Recommended)

```bash
python download_model.py
```

This downloads ~4GB model files.

### 5. Start the Application

```bash
python app.py
```

Open browser: **http://localhost:5001**

## Manual Setup

If the automated script fails, follow these steps:

### 1. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install PyTorch with CUDA

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install Dependencies

```bash
pip install -r requirements-gpu.txt
```

### 4. (Optional) Install Flash Attention

```bash
pip install flash-attn==2.7.3 --no-build-isolation
```

**Note**: Flash Attention requires Visual Studio Build Tools. If it fails, you can skip it - the app will work without it (just slower).

### 5. Configure

```bash
copy .env.example .env
```

Edit `.env`:
```ini
DEVICE_MODE=gpu
USE_FLASH_ATTENTION=true  # Set to false if flash-attn installation failed
```

### 6. Run

```bash
python download_model.py  # Download model first (optional but recommended)
python app.py             # Start the application
```

## Troubleshooting

### Flash Attention Installation Fails

**Solution 1**: Skip it and use standard attention
```ini
# In .env file
USE_FLASH_ATTENTION=false
```

**Solution 2**: Install Visual Studio Build Tools
- Download from: https://visualstudio.microsoft.com/downloads/
- Select "Desktop development with C++"
- Try installing flash-attn again

### CUDA Out of Memory

**Solutions**:
- Reduce base_size in the web interface (try 512 or 640)
- Close other GPU-intensive applications
- Use smaller images/PDFs

### "Torch not compiled with CUDA"

**Check CUDA Installation**:
```bash
# In Python
import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))  # Should show your GPU name
```

If False:
- Reinstall PyTorch with CUDA: `pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118`
- Check CUDA is installed: `nvcc --version`

### Model Download Fails

**Solutions**:
- Check internet connection
- Ensure ~10GB free disk space
- Try again (downloads resume automatically)
- Set custom cache:
  ```bash
  set HF_HOME=C:\path\to\cache
  python download_model.py
  ```

## Performance Tips

### Optimal Settings:
- **Base Size**: 1024 (good balance)
- **Flash Attention**: Enabled (2-3x faster)
- **GPU**: RTX 3060 or better recommended

### Expected Performance:
- **First upload**: 30-60 seconds (model loading)
- **Subsequent uploads**: 2-10 seconds depending on image size
- **Processing speed**: ~2,500 tokens/second on A100

## System Requirements

### Minimum:
- GPU: GTX 1060 6GB
- RAM: 16GB
- Storage: 15GB free
- CUDA: 11.8

### Recommended:
- GPU: RTX 3060 12GB or better
- RAM: 32GB
- Storage: 20GB free SSD
- CUDA: 11.8

## Next Steps

1. Upload a document (PNG, JPG, or PDF)
2. Select OCR mode (Markdown, Free OCR, or Figure)
3. Choose quality settings
4. Click "Upload and Process"
5. Copy results to clipboard

## Support

Having issues? Check:
1. CUDA installation: `nvidia-smi`
2. PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. GPU memory: Task Manager → Performance → GPU
4. Application logs in the terminal

## Additional Resources

- [CUDA Toolkit Download](https://developer.nvidia.com/cuda-downloads)
- [PyTorch Installation](https://pytorch.org/get-started/locally/)
- [DeepSeek-OCR Repo](https://github.com/deepseek-ai/DeepSeek-OCR)
