"""
Train Digit Recognition Model
Train CNN model on MNIST dataset for banking form digit recognition
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from models.digit_model import DigitRecognitionModel
from config import get_config

def main():
    """Train and save the digit recognition model"""
    print("="*60)
    print("🚀 BankForm-AI Model Training (Enhanced)")
    print("="*60)
    
    # Get configuration
    config = get_config()
    model_path = config.MODEL_PATH
    
    # Create model
    print("\n📦 Creating enhanced model architecture...")
    model = DigitRecognitionModel()
    model.build_model()
    
    print("\n📊 Model Summary:")
    model.model.summary()
    
    # Train model
    print("\n🏋️  Training model on MNIST dataset with data augmentation...")
    print("This may take 10-20 minutes depending on your hardware.\n")
    
    # Increased epochs with data augmentation and learning rate scheduling
    history = model.train(epochs=25, batch_size=64)
    
    # Save model
    print(f"\n💾 Saving model to {model_path}...")
    model.save_model(model_path)
    
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)
    print(f"📍 Model saved at: {model_path}")
    print("🎯 You can now start the application with: python app.py")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
