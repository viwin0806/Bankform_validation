"""
Generate Sample Test Images for Digit Recognition
Creates synthetic digit images (0-9) in MNIST format for testing
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def create_sample_digits(output_dir='sample_test_images'):
    """
    Create sample digit images (0-9) in MNIST format (28x28)
    
    Args:
        output_dir: Directory to save the images
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create variations of each digit
    for digit in range(10):
        print(f"Creating sample images for digit {digit}...")
        
        # Create multiple variations (clean, rotated, shifted)
        variations = []
        
        # Variation 1: Clean digit
        img = create_clean_digit(digit)
        variations.append(('clean', img))
        
        # Variation 2: Rotated
        img = create_clean_digit(digit)
        img = img.rotate(15, fillcolor='black')
        variations.append(('rotated', img))
        
        # Variation 3: Slightly faded/low contrast
        img = create_faded_digit(digit)
        variations.append(('faded', img))
        
        # Variation 4: Shifted
        img = create_shifted_digit(digit)
        variations.append(('shifted', img))
        
        # Variation 5: Thick/bold
        img = create_bold_digit(digit)
        variations.append(('bold', img))
        
        # Save all variations
        for var_name, img in variations:
            filename = f"{output_dir}/digit_{digit}_{var_name}.png"
            img = img.resize((28, 28), Image.Resampling.LANCZOS)
            img.save(filename)
            print(f"  ✓ Saved {filename}")
    
    print(f"\n✅ Sample images created in '{output_dir}' directory!")
    return output_dir


def create_clean_digit(digit):
    """Create a clean digit image"""
    img = Image.new('L', (50, 50), color=0)  # Black background
    draw = ImageDraw.Draw(img)
    
    # Try to use system font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    # Draw white digit on black background
    draw.text((5, 5), str(digit), fill=255, font=font)
    return img


def create_faded_digit(digit):
    """Create a faded/low contrast digit"""
    img = create_clean_digit(digit)
    arr = np.array(img, dtype=np.float32)
    
    # Reduce contrast
    arr = (arr - 128) * 0.5 + 128
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    
    return Image.fromarray(arr, mode='L')


def create_shifted_digit(digit):
    """Create a shifted digit (off-center)"""
    img = Image.new('L', (50, 50), color=0)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    # Shift digit to right and down
    draw.text((10, 8), str(digit), fill=255, font=font)
    return img


def create_bold_digit(digit):
    """Create a bold/thick digit"""
    img = Image.new('L', (50, 50), color=0)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 42)
    except:
        font = ImageFont.load_default()
    
    # Draw multiple times to make it bold
    for offset in [-1, 0, 1]:
        draw.text((5+offset, 5), str(digit), fill=255, font=font)
    
    return img


if __name__ == '__main__':
    print("=" * 60)
    print("🎨 Generating Sample Test Images")
    print("=" * 60 + "\n")
    
    output_dir = create_sample_digits()
    
    print("\n" + "=" * 60)
    print("📊 Summary:")
    print("=" * 60)
    print(f"Created 50 test images (5 variations × 10 digits)")
    print(f"Location: {output_dir}")
    print(f"Format: 28×28 grayscale PNG")
    print("\nNow you can test with: python test_model.py")
    print("=" * 60)
