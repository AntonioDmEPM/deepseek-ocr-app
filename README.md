# DeepSeek OCR Web Application

A simple Python web application for OCR (Optical Character Recognition) using DeepSeek-OCR. Upload documents and extract text with intelligent processing.

## Features

- Web-based file upload interface
- Support for multiple image formats (PNG, JPG, PDF, BMP, TIFF)
- Multiple OCR modes:
  - Document to Markdown conversion
  - Free OCR
  - Figure parsing
- Adjustable resolution settings for quality/speed tradeoff
- Clean, modern UI with drag-and-drop support
- Copy results to clipboard

## Prerequisites

- Python 3.12.9
- CUDA 11.8 (for GPU acceleration)
- NVIDIA GPU with at least 8GB VRAM (recommended)

## Installation

### 1. Clone the DeepSeek-OCR Repository

```bash
git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
cd DeepSeek-OCR
```

### 2. Create Conda Environment

```bash
conda create -n deepseek-ocr python=3.12.9 -y
conda activate deepseek-ocr
```

### 3. Install PyTorch with CUDA Support

```bash
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install Flash Attention

```bash
pip install flash-attn==2.7.3 --no-build-isolation
```

### 5. Install Application Dependencies

Navigate to this project directory and install:

```bash
pip install -r requirements.txt
```

### 6. Install Transformers with Trust Remote Code

```bash
pip install transformers==4.48.1
```

## Usage

### Start the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

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

### API Endpoint

You can also use the API directly:

```bash
curl -X POST -F "file=@document.png" \
     -F "prompt_type=markdown" \
     -F "base_size=1024" \
     -F "image_size=640" \
     http://localhost:5000/upload
```

## Configuration

You can modify the following settings in `app.py`:

- `MAX_CONTENT_LENGTH`: Maximum file upload size (default: 16MB)
- `ALLOWED_EXTENSIONS`: Supported file formats
- `UPLOAD_FOLDER`: Directory for temporary file storage

## Performance

- Processing speed: ~2,500 tokens/second on A100-40G GPU
- The model will automatically use GPU if available, otherwise falls back to CPU
- First request may be slower due to model loading

## Troubleshooting

### CUDA Out of Memory

If you encounter GPU memory errors:
- Use a smaller base_size (512 or 640)
- Reduce image dimensions before upload
- Close other GPU-intensive applications

### Model Download Issues

The model will be automatically downloaded from Hugging Face on first run. If you have connection issues:
- Set up Hugging Face cache: `export HF_HOME=/path/to/cache`
- Use a mirror if needed: `export HF_ENDPOINT=https://hf-mirror.com`

### Flash Attention Installation Failed

If flash-attn fails to install:
- Ensure CUDA is properly installed: `nvcc --version`
- Try installing with: `pip install flash-attn==2.7.3 --no-build-isolation`
- Verify your GPU is compatible with Flash Attention 2

## Project Structure

```
.
├── app.py                 # Main Flask application
├── templates/
│   └── index.html        # Web interface
├── uploads/              # Temporary upload directory (auto-created)
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## License

This application uses DeepSeek-OCR. Please refer to the [DeepSeek-OCR repository](https://github.com/deepseek-ai/DeepSeek-OCR) for licensing information.

## Contributing

Feel free to open issues or submit pull requests for improvements.

## Acknowledgments

- [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) for the OCR model
- Flask for the web framework
- Hugging Face Transformers for model integration
