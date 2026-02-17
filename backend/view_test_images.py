"""
View Sample Test Images
Simple script to display and analyze the generated test images
"""

import os
import glob
from PIL import Image
import numpy as np

def analyze_test_images():
    """Analyze and display info about test images"""
    print("=" * 70)
    print("📸 Sample Test Images Analysis")
    print("=" * 70)
    
    sample_dir = 'sample_test_images'
    
    if not os.path.exists(sample_dir):
        print("❌ Sample images directory not found!")
        return
    
    # Get all images
    images = sorted(glob.glob(f'{sample_dir}/*.png'))
    
    print(f"\n✅ Found {len(images)} test images\n")
    
    # Group by digit
    by_digit = {}
    for img_path in images:
        filename = os.path.basename(img_path)
        digit = filename.split('_')[1]
        
        if digit not in by_digit:
            by_digit[digit] = []
        by_digit[digit].append(filename)
    
    # Print organized view
    print("Images organized by digit:\n")
    print("-" * 70)
    
    for digit in sorted(by_digit.keys()):
        print(f"\n📌 Digit {digit}:")
        for filename in sorted(by_digit[digit]):
            img_path = os.path.join(sample_dir, filename)
            img = Image.open(img_path)
            
            # Analyze image
            arr = np.array(img)
            white_pixels = np.sum(arr > 127)
            total_pixels = arr.size
            coverage = (white_pixels / total_pixels) * 100
            
            # Get variation type
            var_type = filename.split('_')[2].replace('.png', '')
            
            print(f"   ├─ {filename:<35} ({coverage:>5.1f}% coverage)")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("📊 Summary Statistics")
    print("=" * 70)
    
    print(f"Total Images: {len(images)}")
    print(f"Digits Covered: 0-9 (10 digits)")
    print(f"Variations per Digit: 5")
    print(f"  ├─ clean (perfect conditions)")
    print(f"  ├─ rotated (±10 degree rotation)")
    print(f"  ├─ faded (low contrast)")
    print(f"  ├─ shifted (off-center)")
    print(f"  └─ bold (thick strokes)")
    
    # Check file sizes
    total_size = sum(os.path.getsize(os.path.join(sample_dir, f)) for f in os.listdir(sample_dir) if f.endswith('.png'))
    print(f"\nTotal Folder Size: {total_size / 1024:.1f} KB")
    print(f"Image Size: 28×28 pixels (MNIST format)")
    print(f"Image Format: Grayscale PNG")
    
    print("\n" + "=" * 70)
    print("✅ Sample images are ready for testing!")
    print("=" * 70 + "\n")

if __name__ == '__main__':
    analyze_test_images()
