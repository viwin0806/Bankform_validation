"""
BankForm-AI Configuration
Central configuration for the banking challan processing system
"""

import os
from pathlib import Path

# Base Directories
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent
UPLOAD_FOLDER = BASE_DIR / 'uploads'
MODEL_FOLDER = BASE_DIR / 'models' / 'trained'
TEMPLATE_FOLDER = BASE_DIR / 'templates'

# Ensure directories exist
UPLOAD_FOLDER.mkdir(exist_ok=True)
MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

# Flask Configuration
class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or f'sqlite:///{PROJECT_ROOT}/bankform.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Upload settings
    UPLOAD_FOLDER = str(UPLOAD_FOLDER)
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf', 'tiff'}
    
    # Model settings
    MODEL_PATH = str(MODEL_FOLDER / 'mnist_cnn.h5')
    CONFIDENCE_THRESHOLD_HIGH = 0.90
    CONFIDENCE_THRESHOLD_LOW = 0.70
    
    # Processing settings
    IMAGE_SIZE = (28, 28)  # MNIST standard
    BATCH_SIZE = 32
    
    # ─── OCR Engine Configuration ───
    # Priority: EasyOCR (primary) → Tesseract (fallback) → CNN (handwritten digits only)
    OCR_ENGINE_PRIMARY = 'easyocr'           # 'easyocr' or 'tesseract'
    OCR_ENGINE_FALLBACK = 'tesseract'        # fallback engine
    EASYOCR_LANGUAGES = ['en']               # ['en', 'hi'] for Hindi+English
    EASYOCR_GPU = False                      # Set True if CUDA GPU available
    TESSERACT_CMD = None                     # Path to tesseract.exe (auto-detect if None)
    TESSERACT_LANG = 'eng'                   # Tesseract language
    USE_CNN_FOR_DIGITS = True                # Use CNN model for handwritten digit fields
    
    # ─── Indian Bank Challan Field Keywords ───
    # Used by smart field matcher to identify fields on real challans
    FIELD_KEYWORDS = {
        'account_number': ['account', 'a/c', 'acct', 'acc no', 'account no', 'account number', 'a/c no'],
        'amount': ['amount', 'amt', 'rs', 'rupees', 'total', 'sum', 'rs.'],
        'date': ['date', 'dt', 'dated', 'dd/mm/yyyy', 'dd-mm-yyyy'],
        'name': ['name', 'depositor', 'account holder', 'a/c holder', 'customer name', 'applicant'],
        'branch': ['branch', 'br', 'branch name', 'branch code'],
        'ifsc': ['ifsc', 'ifsc code', 'ifs code', 'micr'],
        'cheque_number': ['cheque', 'chq', 'check', 'cheque no', 'chq no', 'instrument no'],
        'reference_number': ['reference', 'ref', 'ref no', 'transaction', 'txn', 'slip no'],
        'pan': ['pan', 'pan no', 'pan number', 'pan card'],
        'mobile': ['mobile', 'phone', 'contact', 'mob', 'tel', 'cell'],
        'deposit_type': ['cash', 'cheque', 'demand draft', 'dd', 'neft', 'rtgs', 'upi'],
    }
    
    # Banking validation rules
    ACCOUNT_NUMBER_MIN_LENGTH = 9
    ACCOUNT_NUMBER_MAX_LENGTH = 18
    MAX_AMOUNT = 10000000  # 1 Crore INR
    
    # Export settings
    EXPORT_FORMATS = ['csv', 'json', 'excel']
    
    # CORS settings
    CORS_ORIGINS = ['http://localhost:8080', 'http://127.0.0.1:8080', 'http://localhost:8000', 'http://127.0.0.1:8000']

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    # Override with environment variables
    SECRET_KEY = os.environ.get('SECRET_KEY')

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

# Get active configuration
def get_config():
    env = os.environ.get('FLASK_ENV', 'development')
    return config.get(env, config['default'])
