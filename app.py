import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from transformers import AutoModel, AutoTokenizer
import torch
from config import Config

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
app.config['ALLOWED_EXTENSIONS'] = Config.ALLOWED_EXTENSIONS

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model and tokenizer
model = None
tokenizer = None
device_info = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_model():
    """Load the DeepSeek-OCR model and tokenizer."""
    global model, tokenizer, device_info

    if model is None:
        # Get device configuration
        device_info = Config.get_device_info()

        print("=" * 60)
        print(f"Loading DeepSeek-OCR model...")
        print(f"Device Mode: {Config.DEVICE_MODE}")
        print(f"Target Device: {device_info['device']}")
        print(f"Device Name: {device_info['device_name']}")
        print(f"CUDA Available: {device_info['cuda_available']}")
        print(f"Flash Attention: {device_info['use_flash_attention']}")
        print("=" * 60)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            Config.MODEL_NAME,
            trust_remote_code=True
        )

        # Prepare model loading arguments
        model_kwargs = {
            'trust_remote_code': True,
        }

        # Set dtype based on device
        if device_info['device'] == 'cuda' and Config.TORCH_DTYPE == 'float16':
            model_kwargs['torch_dtype'] = torch.float16
        else:
            model_kwargs['torch_dtype'] = torch.float32

        # Add flash attention if enabled and on GPU
        if device_info['use_flash_attention']:
            model_kwargs['attn_implementation'] = 'flash_attention_2'
            print("Using Flash Attention 2 for faster inference")
        else:
            print("Using standard attention (CPU-compatible)")

        # Load model
        model = AutoModel.from_pretrained(
            Config.MODEL_NAME,
            **model_kwargs
        )

        # Move model to device
        model = model.to(device_info['device'])

        print(f"Model loaded successfully on {device_info['device']}!")
        print("=" * 60)

    return model, tokenizer


def perform_ocr(image_path, prompt_type='markdown', base_size=1024, image_size=640):
    """
    Perform OCR on the uploaded image or PDF.

    Args:
        image_path: Path to the image/PDF file
        prompt_type: Type of OCR ('markdown', 'free', 'figure')
        base_size: Base size for image processing (512, 640, 1024, 1280)
        image_size: Image size parameter

    Returns:
        OCR result text
    """
    import fitz  # PyMuPDF
    from PIL import Image

    model, tokenizer = load_model()

    # Define prompts based on type
    prompts = {
        'markdown': '<image>\n<|grounding|>Convert document to markdown',
        'free': '<image>\nFree OCR.',
        'figure': '<image>\nParse the figure.'
    }

    prompt = prompts.get(prompt_type, prompts['markdown'])

    # Create temp output directory with absolute path
    output_dir = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], 'temp_output'))
    os.makedirs(output_dir, exist_ok=True)

    # Handle PDF files - convert to image
    converted_image_path = image_path
    if image_path.lower().endswith('.pdf'):
        print(f"Converting PDF to image: {image_path}")
        doc = fitz.open(image_path)
        page = doc[0]  # Get first page
        pix = page.get_pixmap(dpi=300)  # High resolution

        # Save as PNG
        converted_image_path = image_path.rsplit('.', 1)[0] + '_converted.png'
        pix.save(converted_image_path)
        doc.close()
        print(f"PDF converted to: {converted_image_path}")

    print(f"Output directory: {output_dir}")
    print(f"Image file: {converted_image_path}")
    print(f"Prompt: {prompt}")

    result = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=converted_image_path,
        output_path=output_dir,
        base_size=base_size,
        image_size=image_size,
        crop_mode=True,
        save_results=False,  # Don't save intermediate files
        test_compress=False  # Don't test compression
    )

    # Clean up converted image if it was created
    if converted_image_path != image_path and os.path.exists(converted_image_path):
        os.remove(converted_image_path)

    return result


@app.route('/')
def index():
    """Render the main upload page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and perform OCR."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Get OCR parameters from request
            prompt_type = request.form.get('prompt_type', 'markdown')
            base_size = int(request.form.get('base_size', 1024))
            image_size = int(request.form.get('image_size', 640))

            # Perform OCR
            result = perform_ocr(filepath, prompt_type, base_size, image_size)

            return jsonify({
                'success': True,
                'filename': filename,
                'result': result
            })

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print("=" * 70)
            print("ERROR during OCR processing:")
            print(error_details)
            print("=" * 70)
            return jsonify({'error': str(e)}), 500

        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)

    return jsonify({'error': 'File type not allowed'}), 400


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'device_info': device_info if device_info else 'Model not loaded yet'
    })


@app.route('/device-info')
def get_device_info():
    """Get current device configuration."""
    if device_info:
        return jsonify(device_info)
    else:
        return jsonify({
            'message': 'Model not loaded yet',
            'config': {
                'device_mode': Config.DEVICE_MODE,
                'use_flash_attention': Config.USE_FLASH_ATTENTION
            }
        })


if __name__ == '__main__':
    app.run(debug=Config.DEBUG, host=Config.HOST, port=Config.PORT)
