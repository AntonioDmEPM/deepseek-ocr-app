"""
Script to pre-download the DeepSeek-OCR model and dependencies
"""
import os
from transformers import AutoModel, AutoTokenizer
from config import Config

def download_model():
    """Download and cache the DeepSeek-OCR model"""
    print("=" * 70)
    print("DeepSeek-OCR Model Downloader")
    print("=" * 70)
    print()
    print(f"Model: {Config.MODEL_NAME}")
    print(f"Device Mode: {Config.DEVICE_MODE}")
    print()
    print("This will download approximately 4GB of model files.")
    print("The download may take several minutes depending on your connection.")
    print()
    print("=" * 70)
    print()

    # Get device info
    device_info = Config.get_device_info()
    print(f"Target Device: {device_info['device']}")
    print(f"Device Name: {device_info['device_name']}")
    print(f"CUDA Available: {device_info['cuda_available']}")
    print(f"Flash Attention: {device_info['use_flash_attention']}")
    print()

    try:
        print("Step 1/2: Downloading tokenizer...")
        print("-" * 70)
        tokenizer = AutoTokenizer.from_pretrained(
            Config.MODEL_NAME,
            trust_remote_code=True
        )
        print("✓ Tokenizer downloaded successfully!")
        print()

        print("Step 2/2: Downloading model...")
        print("-" * 70)
        print("This is the largest file (~4GB) and may take several minutes...")

        # Prepare model loading arguments
        model_kwargs = {
            'trust_remote_code': True,
        }

        # Set dtype based on device
        import torch
        if device_info['device'] == 'cuda' and Config.TORCH_DTYPE == 'float16':
            model_kwargs['torch_dtype'] = torch.float16
            print("Using torch.float16 (GPU optimized)")
        else:
            model_kwargs['torch_dtype'] = torch.float32
            print("Using torch.float32 (CPU compatible)")

        # Add flash attention if enabled and on GPU
        if device_info['use_flash_attention']:
            model_kwargs['attn_implementation'] = 'flash_attention_2'
            print("Using Flash Attention 2")
        else:
            print("Using standard attention")

        print()

        model = AutoModel.from_pretrained(
            Config.MODEL_NAME,
            **model_kwargs
        )

        print()
        print("✓ Model downloaded successfully!")
        print()
        print("=" * 70)
        print("SUCCESS! DeepSeek-OCR model is ready to use.")
        print("=" * 70)
        print()
        print("The model files are cached in:")
        print(f"  {os.path.expanduser('~/.cache/huggingface/hub/')}")
        print()
        print("You can now start the application with: python app.py")
        print()

        return True

    except Exception as e:
        print()
        print("=" * 70)
        print("ERROR: Failed to download model")
        print("=" * 70)
        print(f"Error: {str(e)}")
        print()
        print("Possible solutions:")
        print("1. Check your internet connection")
        print("2. Ensure you have enough disk space (~10GB free)")
        print("3. Try running again (downloads can be resumed)")
        print("4. Check Hugging Face status: https://status.huggingface.co/")
        print()
        return False

if __name__ == "__main__":
    success = download_model()
    exit(0 if success else 1)
