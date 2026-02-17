"""
Simple Quick Test - Test model on sample images
"""

import os
import sys
from pathlib import Path
import glob

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from models.digit_model import DigitRecognitionModel
from config import get_config

def quick_test():
    """Quick test on sample images"""
    print("=" * 70)
    print("🧪 Quick Test - Digit Recognition Model")
    print("=" * 70)
    
    # Get configuration
    config = get_config()
    model_path = config.MODEL_PATH
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\n❌ Model not found at {model_path}")
        print("Please wait for training to complete: python train_model.py")
        return
    
    # Load model
    print(f"\n📦 Loading model...")
    model = DigitRecognitionModel(model_path)
    
    # Check if sample images exist
    sample_dir = 'sample_test_images'
    if not os.path.exists(sample_dir):
        print(f"⚠️  Sample images not found!")
        return
    
    # Get sample images
    image_files = sorted(glob.glob(f'{sample_dir}/*.png'))[:10]  # Just test first 10
    
    if not image_files:
        print(f"❌ No test images found!")
        return
    
    print(f"\n📊 Testing on {len(image_files)} sample images:\n")
    
    correct = 0
    for image_path in image_files:
        filename = os.path.basename(image_path)
        expected_digit = int(filename.split('_')[1])
        
        result = model.predict(image_path)
        predicted = result['prediction']
        confidence = result['confidence']
        
        is_correct = predicted == expected_digit
        correct += is_correct
        
        status = "✅" if is_correct else "❌"
        print(f"{status} {filename:<40} Predicted: {predicted} ({confidence:.1%})")
    
    accuracy = (correct / len(image_files) * 100)
    
    print("\n" + "=" * 70)
    print(f"Accuracy: {accuracy:.0f}% ({correct}/{len(image_files)})")
    print("=" * 70 + "\n")

if __name__ == '__main__':
    quick_test()
