"""
Test Digit Recognition Model with Sample Images
Test the trained model on generated sample digit images
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from models.digit_model import DigitRecognitionModel
from config import get_config
import glob

def test_model():
    """Test model on sample images"""
    print("=" * 70)
    print("[TEST] Digit Recognition Model Testing")
    print("=" * 70)
    
    # Get configuration
    config = get_config()
    model_path = config.MODEL_PATH
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please train the model first: python train_model.py")
        return
    
    # Load model
    print(f"\n[LOAD] Loading model from {model_path}...")
    model = DigitRecognitionModel(model_path)
    
    # Check if sample images exist
    sample_dir = 'sample_test_images'
    if not os.path.exists(sample_dir):
        print(f"\n[INFO] Sample images not found. Generating them...")
        from generate_test_images import create_sample_digits
        create_sample_digits()
    
    # Get all sample images
    image_files = sorted(glob.glob(f'{sample_dir}/*.png'))
    
    if not image_files:
        print(f"[ERROR] No test images found in {sample_dir}")
        return
    
    print(f"\n[INFO] Found {len(image_files)} test images\n")
    
    # Test each image
    results = {}
    correct = 0
    total = 0
    
    print("-" * 70)
    print(f"{'Image':<40} {'Predicted':<12} {'Confidence':<12} {'Result':<12}")
    print("-" * 70)
    
    for image_path in image_files:
        # Extract expected digit from filename
        filename = os.path.basename(image_path)
        expected_digit = int(filename.split('_')[1])
        
        # Predict
        result = model.predict(image_path)
        predicted = result['prediction']
        confidence = result['confidence']
        
        is_correct = predicted == expected_digit
        correct += is_correct
        total += 1
        
        result_str = "[OK]" if is_correct else f"[FAIL] Expected {expected_digit}"
        
        print(f"{filename:<40} {predicted:<12} {confidence:>10.2%}   {result_str:<12}")
    
    # Summary
    accuracy = (correct / total * 100) if total > 0 else 0
    
    print("-" * 70)
    print("\n" + "=" * 70)
    print("[RESULTS] Test Results Summary")
    print("=" * 70)
    print(f"Total Tests:     {total}")
    print(f"Correct:         {correct}")
    print(f"Incorrect:       {total - correct}")
    print(f"Accuracy:        {accuracy:.1f}%")
    print("=" * 70)
    
    if accuracy == 100:
        print("[SUCCESS] Perfect! Model is working flawlessly!")
    elif accuracy >= 90:
        print("[GOOD] Great! Model is performing well!")
    elif accuracy >= 80:
        print("[OK] Good! Model is performing reasonably well!")
    else:
        print("[WARNING] Model needs improvement. Check training or preprocessing!")
    
    print("=" * 70 + "\n")


if __name__ == '__main__':
    test_model()
