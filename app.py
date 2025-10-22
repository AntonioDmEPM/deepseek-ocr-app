import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from transformers import AutoModel, AutoTokenizer
import torch

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'pdf', 'bmp', 'tiff'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model and tokenizer
model = None
tokenizer = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_model():
    """Load the DeepSeek-OCR model and tokenizer."""
    global model, tokenizer

    if model is None:
        print("Loading DeepSeek-OCR model...")
        tokenizer = AutoTokenizer.from_pretrained(
            'deepseek-ai/DeepSeek-OCR',
            trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            'deepseek-ai/DeepSeek-OCR',
            attn_implementation='flash_attention_2',
            trust_remote_code=True,
            torch_dtype=torch.float16
        )

        # Move model to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()

        print("Model loaded successfully!")

    return model, tokenizer


def perform_ocr(image_path, prompt_type='markdown', base_size=1024, image_size=640):
    """
    Perform OCR on the uploaded image.

    Args:
        image_path: Path to the image file
        prompt_type: Type of OCR ('markdown', 'free', 'figure')
        base_size: Base size for image processing (512, 640, 1024, 1280)
        image_size: Image size parameter

    Returns:
        OCR result text
    """
    model, tokenizer = load_model()

    # Define prompts based on type
    prompts = {
        'markdown': '<image>\n<|grounding|>Convert document to markdown',
        'free': '<image>\nFree OCR.',
        'figure': '<image>\nParse the figure.'
    }

    prompt = prompts.get(prompt_type, prompts['markdown'])

    # Perform inference
    result = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_path,
        base_size=base_size,
        image_size=image_size,
        crop_mode=True
    )

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
            return jsonify({'error': str(e)}), 500

        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)

    return jsonify({'error': 'File type not allowed'}), 400


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
