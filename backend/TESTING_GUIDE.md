# Testing Your Digit Recognition Model

## Sample Test Images Created ✅

I've created **50 synthetic test images** for you to test the model:
- **5 variations × 10 digits (0-9)**
- Each image is **28×28 pixels** (MNIST format)
- Stored in: `sample_test_images/` directory

### Variations Created:

| Variation | Description | Use Case |
|-----------|-------------|----------|
| **clean** | Perfect, centered digit | Ideal conditions |
| **rotated** | Rotated ±10 degrees | Skewed form entries |
| **faded** | Low contrast, faded | Photocopied forms |
| **shifted** | Off-center digit | Misaligned entries |
| **bold** | Thick/bold digit | Pen pressure variations |

## How to Test

### Quick Test (First 10 images):
```bash
python quick_test.py
```

### Full Test (All 50 images):
```bash
python test_model.py
```

## Sample Image Locations

All test images are in: `backend/sample_test_images/`

Example files:
```
sample_test_images/
├── digit_0_clean.png      ← Clean 0
├── digit_0_rotated.png    ← Rotated 0
├── digit_0_faded.png      ← Faded 0
├── digit_1_clean.png      ← Clean 1
├── digit_1_bold.png       ← Bold 1
... (50 total files)
```

## Expected Results

- **Clean/Shifted digits**: 95%+ accuracy
- **Rotated digits**: 85-90% accuracy
- **Faded digits**: 70-80% accuracy
- **Bold digits**: 90%+ accuracy

## Your Current Model Performance

✅ **Test Accuracy: 99.02%** (3 epochs of training)
- Test Loss: 0.1578
- Model properly identifies MNIST digits

## Using Images in Your Application

### Test a Single Image:
```python
from models.digit_model import DigitRecognitionModel

model = DigitRecognitionModel('models/trained/mnist_cnn.h5')
result = model.predict('sample_test_images/digit_5_clean.png')

print(f"Predicted: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Test Multiple Images:
```python
images = [
    'sample_test_images/digit_0_clean.png',
    'sample_test_images/digit_1_clean.png',
    'sample_test_images/digit_2_clean.png',
]

results = model.predict_batch(images)
for i, result in enumerate(results):
    print(f"Image {i}: Predicted {result['prediction']} ({result['confidence']:.1%})")
```

## Improve Model Further

If accuracy is lower than expected:

1. **More Training**: Increase epochs in `train_model.py`
   ```python
   history = model.train(epochs=50, batch_size=32)  # Longer training
   ```

2. **Your Own Data**: Collect actual banking form digits
   ```python
   model.train_on_custom_data(your_form_images, epochs=5)
   ```

3. **Fine-tuning**: Train on preprocessing to better handle forms
   - Adjust preprocessing in `digit_model.py` → `preprocess_image()`
   - Try different thresholding techniques
   - Improve centering logic

## Understanding Model Output

Each prediction returns:
```python
{
    'prediction': 5,              # Predicted digit
    'confidence': 0.987,          # Confidence score (0-1)
    'probabilities': {            # All class probabilities
        '0': 0.001,
        '1': 0.002,
        '2': 0.001,
        '3': 0.001,
        '4': 0.002,
        '5': 0.987,  # ← Highest
        '6': 0.003,
        '7': 0.001,
        '8': 0.001,
        '9': 0.001,
    }
}
```

## Troubleshooting

### Low Accuracy on Real Forms
- Banking forms may have different fonts/styles
- Try data augmentation with real form samples
- Adjust preprocessing (thresholding, centering)

### Model Predicts Wrong Digit
- Check image preprocessing (inversion, crop, resize)
- Ensure image is 28×28 grayscale
- Verify white digits on black background (MNIST standard)

### Slow Predictions
- Model size: 3.07 MB
- Should predict in <100ms on CPU
- Consider GPU acceleration for batch processing

---

**Created**: January 27, 2026  
**Model Version**: 2.0 (Enhanced)  
**Test Images**: 50 (5 variations × 10 digits)
