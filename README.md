# DeepSeek OCR Web Application

A simple Python web application for OCR (Optical Character Recognition) using DeepSeek-OCR. Upload documents and extract text with intelligent processing.

## Features

- Web-based file upload interface
- Support for multiple image formats (PNG, JPG, PDF, BMP, TIFF)
- Multiple OCR modes:
  - Document to Markdown conversion
  - Free OCR
  - Figure parsing
- **CPU and GPU support** with easy configuration
- Adjustable resolution settings for quality/speed tradeoff
- Clean, modern UI with drag-and-drop support
- Copy results to clipboard
- REST API endpoints

## Prerequisites

### Minimum Requirements (CPU Mode)
- Python 3.9 or higher
- 8GB RAM (16GB recommended)
- Any modern CPU

### For GPU Acceleration (Optional)
- Python 3.9 or higher
- CUDA 11.8
- NVIDIA GPU with at least 8GB VRAM
- 16GB System RAM

## Installation

### Option 1: CPU-Only Setup (No GPU Required)

Perfect for development, testing, or systems without NVIDIA GPU.

```bash
# 1. Clone this repository
git clone https://github.com/AntonioDmEPM/deepseek-ocr-app.git
cd deepseek-ocr-app

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install CPU dependencies
pip install -r requirements-cpu.txt

# 4. Copy environment configuration
cp .env.example .env

# 5. Ensure .env has CPU mode (default)
# DEVICE_MODE=cpu
# USE_FLASH_ATTENTION=false

# 6. (Optional but recommended) Pre-download the model
python download_model.py
```

### Option 2: GPU Setup (For NVIDIA GPU Users)

For faster processing with GPU acceleration.

```bash
# 1. Clone this repository
git clone https://github.com/AntonioDmEPM/deepseek-ocr-app.git
cd deepseek-ocr-app

# 2. Create conda environment (recommended for GPU)
conda create -n deepseek-ocr python=3.12 -y
conda activate deepseek-ocr

# 3. Install PyTorch with CUDA support
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Install GPU dependencies
pip install -r requirements-gpu.txt

# 5. Install Flash Attention (optional, for faster inference)
pip install flash-attn==2.7.3 --no-build-isolation

# 6. Configure for GPU
cp .env.example .env
# Edit .env and set:
# DEVICE_MODE=gpu
# USE_FLASH_ATTENTION=true
```

## Configuration

The application uses environment variables for configuration. Create a `.env` file from the example:

```bash
cp .env.example .env
```

### Configuration Options

Edit `.env` to customize:

```bash
# Device Configuration
DEVICE_MODE=cpu           # Options: 'cpu', 'gpu', 'auto'
USE_FLASH_ATTENTION=false # Set to 'true' only if you have GPU + flash-attn installed

# Server Configuration
HOST=0.0.0.0
PORT=5000
DEBUG=true
```

**Device Mode Options:**
- `cpu`: Force CPU usage (works on all systems)
- `gpu`: Use GPU if available, fallback to CPU if not
- `auto`: Automatically detect and use the best available device

**Flash Attention:**
- Only works with GPU mode
- Requires `flash-attn==2.7.3` to be installed
- Provides 2-3x faster inference on compatible GPUs
- Set to `false` for CPU mode

## Usage

### Start the Application

```bash
python app.py
```

The application will start on `http://localhost:5000` and display device information:

```
============================================================
Loading DeepSeek-OCR model...
Device Mode: cpu
Target Device: cpu
Device Name: CPU
CUDA Available: False
Flash Attention: False
============================================================
```

### Using the Web Interface

1. Open your browser and navigate to `http://localhost:5000`
2. Upload a document by clicking the upload area or dragging and dropping
3. Select your preferred OCR mode:
   - **Document to Markdown**: Best for structured documents
   - **Free OCR**: General purpose text extraction
   - **Parse Figure**: Optimized for diagrams and figures
4. Choose base size (higher = better quality but slower):
   - 512x512: Fastest
   - 640x640: Fast
   - 1024x1024: Recommended (default)
   - 1280x1280: Best quality
5. Click "Upload and Process"
6. Copy the results to clipboard when done

### API Endpoints

#### Upload and Process Document

```bash
curl -X POST -F "file=@document.png" \
     -F "prompt_type=markdown" \
     -F "base_size=1024" \
     -F "image_size=640" \
     http://localhost:5000/upload
```

#### Check Device Configuration

```bash
curl http://localhost:5000/device-info
```

#### Health Check

```bash
curl http://localhost:5000/health
```

## Performance

### CPU Mode
- Processing time: Depends on CPU performance
- Suitable for: Testing, development, occasional use
- Memory: ~4-8GB RAM per inference

### GPU Mode
- Processing speed: ~2,500 tokens/second on A100-40G GPU
- Faster with Flash Attention enabled
- Memory: ~6-10GB VRAM depending on resolution
- Suitable for: Production, high-volume processing

**Note:** First request will be slower due to model loading (~30-60 seconds).

## Troubleshooting

### CPU Mode Issues

**Slow Processing:**
- Use smaller base_size (512 or 640)
- Reduce image dimensions before upload
- This is normal for CPU mode - consider GPU upgrade for production

**Memory Errors:**
- Close other applications
- Use smaller base_size
- Ensure at least 8GB RAM available

### GPU Mode Issues

**CUDA Out of Memory:**
- Use a smaller base_size (512 or 640)
- Reduce image dimensions before upload
- Close other GPU-intensive applications
- Disable Flash Attention: `USE_FLASH_ATTENTION=false`

**Flash Attention Installation Failed:**
- Ensure CUDA is properly installed: `nvcc --version`
- Try: `pip install flash-attn==2.7.3 --no-build-isolation`
- Verify GPU compatibility with Flash Attention 2
- Can still run without it: `USE_FLASH_ATTENTION=false`

**GPU Not Detected:**
- Verify CUDA installation: `nvidia-smi`
- Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Reinstall PyTorch with CUDA support
- Fallback to CPU mode if needed

### Model Download Issues

The model (~4GB) downloads automatically from Hugging Face on first run:

- Set custom cache: `export HF_HOME=/path/to/cache`
- Use mirror if needed: `export HF_ENDPOINT=https://hf-mirror.com`
- Check disk space (need ~10GB free)
- Check internet connection

## Project Structure

```
.
├── app.py                 # Main Flask application
├── config.py              # Configuration management
├── templates/
│   └── index.html        # Web interface
├── uploads/              # Temporary upload directory (auto-created)
├── requirements.txt      # Base dependencies
├── requirements-cpu.txt  # CPU-only dependencies
├── requirements-gpu.txt  # GPU dependencies
├── .env.example          # Environment configuration template
├── .env                  # Your configuration (create from .env.example)
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## Switching Between CPU and GPU

You can easily switch modes by editing `.env`:

**For CPU:**
```bash
DEVICE_MODE=cpu
USE_FLASH_ATTENTION=false
```

**For GPU:**
```bash
DEVICE_MODE=gpu
USE_FLASH_ATTENTION=true  # Only if flash-attn is installed
```

Then restart the application:
```bash
python app.py
```

## License

This application uses DeepSeek-OCR. Please refer to the [DeepSeek-OCR repository](https://github.com/deepseek-ai/DeepSeek-OCR) for licensing information.

## Contributing

Feel free to open issues or submit pull requests for improvements.

## Acknowledgments

- [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) for the OCR model
- Flask for the web framework
- Hugging Face Transformers for model integration
