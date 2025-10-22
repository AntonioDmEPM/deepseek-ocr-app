import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration for DeepSeek OCR Application"""

    # Device configuration - Options: 'cpu', 'gpu', 'auto'
    # 'auto' will automatically detect and use GPU if available
    DEVICE_MODE = os.getenv('DEVICE_MODE', 'cpu')

    # Model configuration
    MODEL_NAME = 'deepseek-ai/DeepSeek-OCR'

    # Flash Attention (only works with GPU)
    # Set to False for CPU-only mode
    USE_FLASH_ATTENTION = os.getenv('USE_FLASH_ATTENTION', 'false').lower() == 'true'

    # Data type - float16 for GPU (faster), float32 for CPU (compatible)
    TORCH_DTYPE = 'float16' if DEVICE_MODE == 'gpu' and USE_FLASH_ATTENTION else 'float32'

    # Upload settings
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf', 'bmp', 'tiff'}

    # Server settings
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    DEBUG = os.getenv('DEBUG', 'true').lower() == 'true'

    @staticmethod
    def get_device_info():
        """Get information about the configured device"""
        import torch

        if Config.DEVICE_MODE == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif Config.DEVICE_MODE == 'gpu':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if device == 'cpu':
                print("WARNING: GPU mode requested but CUDA not available. Falling back to CPU.")
        else:
            device = 'cpu'

        return {
            'device': device,
            'device_name': torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU',
            'cuda_available': torch.cuda.is_available(),
            'use_flash_attention': Config.USE_FLASH_ATTENTION and device == 'cuda'
        }
