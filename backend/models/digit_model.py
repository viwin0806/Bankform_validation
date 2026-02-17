"""
Enhanced Digit Recognition Model
CNN-based digit recognition for banking form processing with data augmentation
"""

import numpy as np
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'  # Set backend before importing keras
import keras
from keras import layers, regularizers
from keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import cv2

class DigitRecognitionModel:
    """Enhanced digit recognition model for banking forms"""
    
    def __init__(self, model_path=None):
        """
        Initialize the digit recognition model
        
        Args:
            model_path: Path to pre-trained model file
        """
        self.model = None
        self.model_path = model_path
        self.img_rows, self.img_cols = 28, 28
        
        if model_path:
            self.load_model(model_path)
    
    def build_model(self):
        """Build enhanced CNN architecture for digit recognition"""
        model = keras.Sequential([
            # First convolutional block - more filters for better feature extraction
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', 
                         input_shape=(self.img_rows, self.img_cols, 1),
                         kernel_regularizer=regularizers.l2(0.0001)),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu',
                         kernel_regularizer=regularizers.l2(0.0001)),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(128, (3, 3), activation='relu',
                         kernel_regularizer=regularizers.l2(0.0001)),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu',
                         kernel_regularizer=regularizers.l2(0.0001)),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block for deeper feature extraction
            layers.Conv2D(256, (3, 3), activation='relu',
                         kernel_regularizer=regularizers.l2(0.0001)),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu',
                        kernel_regularizer=regularizers.l2(0.0001)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu',
                        kernel_regularizer=regularizers.l2(0.0001)),
            layers.Dropout(0.4),
            layers.Dense(10, activation='softmax')
        ])
        
        # Use Adam optimizer with custom learning rate
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        
        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, epochs=25, batch_size=64):
        """
        Train the model on MNIST dataset with data augmentation
        
        Args:
            epochs: Number of training epochs (default: 25)
            batch_size: Batch size for training (default: 64)
        
        Returns:
            Training history
        """
        # Load MNIST data
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # Preprocess data
        x_train = x_train.reshape(x_train.shape[0], self.img_rows, self.img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], self.img_rows, self.img_cols, 1)
        
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Convert labels to categorical
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Data augmentation to increase dataset variety
        # This helps with real-world form digit variations
        train_datagen = ImageDataGenerator(
            rotation_range=10,           # Rotate digits up to 10 degrees
            width_shift_range=0.1,       # Horizontal shift
            height_shift_range=0.1,      # Vertical shift
            shear_range=0.2,             # Shear transformation
            zoom_range=0.1,              # Random zoom
            fill_mode='constant',
            cval=0
        )
        
        # Learning rate scheduler - reduce LR when stuck
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001,
            verbose=1
        )
        
        # Early stopping to prevent overfitting
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train model with data augmentation
        try:
            history = self.model.fit(
                train_datagen.flow(x_train, y_train, batch_size=batch_size),
                epochs=epochs,
                verbose=1,
                validation_data=(x_test, y_test),
                callbacks=[lr_scheduler, early_stop],
                steps_per_epoch=len(x_train) // batch_size
            )
        except KeyboardInterrupt:
            print("\n[WARN] Training interrupted by user")
            print("Evaluating current model performance...")
            history = None
        
        # Evaluate
        score = self.model.evaluate(x_test, y_test, verbose=0)
        print(f'Test loss: {score[0]:.4f}')
        print(f'Test accuracy: {score[1]:.4f}')
        
        return history
    
    def save_model(self, path):
        """Save model to file"""
        if self.model:
            self.model.save(path)
            print(f"[OK] Model saved to {path}")
        else:
            print("[ERROR] No model to save")
    
    def load_model(self, path):
        """Load model from file"""
        try:
            self.model = keras.models.load_model(path)
            self.model_path = path
            print(f"[OK] Model loaded from {path}")
            return True
        except Exception as e:
            print(f"[ERROR] Error loading model: {e}")
            return False
    
    def preprocess_image(self, image):
        """
        Preprocess image for prediction with improved contrast handling
        
        Args:
            image: PIL Image, numpy array, or file path
        
        Returns:
            Preprocessed numpy array ready for prediction
        """
        # Handle different input types
        if isinstance(image, str):
            image = Image.open(image).convert('L')
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("Image must be PIL Image, numpy array, or file path")
        
        img_array = np.array(image, dtype=np.float32)
        
        # Convert to float for processing
        if img_array.dtype != np.float32:
            img_array = img_array.astype(np.float32)
        
        # Handle 3D arrays
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # 1. Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This helps with faded/low-contrast images
        img_uint8 = (np.clip(img_array, 0, 255)).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        img_uint8 = clahe.apply(img_uint8)
        img_array = img_uint8.astype(np.float32)
        
        # 2. Apply threshold for better binary separation
        # This converts gray pixels to pure black or white
        _, img_array = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY)
        
        # 3. Check inversion (ensure background is black, digits are white)
        if np.mean(img_array) > 127:
            img_array = 255 - img_array
            
        # 4. Find bounding box of the digit to crop tight
        coords = cv2.findNonZero(img_array.astype(np.uint8))
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            # Add small padding if possible
            pad = 2
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(img_array.shape[1] - x, w + 2*pad)
            h = min(img_array.shape[0] - y, h + 2*pad)
            digit_crop = img_array[y:y+h, x:x+w]
        else:
            digit_crop = img_array

        # 5. Resize preserving aspect ratio to fit in 20x20 box
        target_size = 20
        h, w = digit_crop.shape
        
        if h > w:
            new_h = target_size
            new_w = int(w * (target_size / h))
        else:
            new_w = target_size
            new_h = int(h * (target_size / w))
            
        # Ensure new dimensions are at least 1
        new_w = max(1, new_w)
        new_h = max(1, new_h)
        
        # Resize using INTER_AREA for downscaling
        resized_digit = cv2.resize(digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 6. Paste into center of 28x28 black canvas
        final_image = np.zeros((self.img_rows, self.img_cols), dtype=np.float32)
        
        # Calculate centering offsets
        pad_x = (self.img_rows - new_w) // 2
        pad_y = (self.img_cols - new_h) // 2
        
        final_image[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized_digit

        # 7. Normalize to 0-1 range
        final_image = final_image / 255.0
        
        # 8. Reshape for model input
        img_array = final_image.reshape(1, self.img_rows, self.img_cols, 1)
        
        return img_array
    
    def predict(self, image):
        """
        Predict digit from image
        
        Args:
            image: Image to classify
        
        Returns:
            dict with prediction, confidence, and all probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded. Train or load a model first.")
        
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Get predictions
        predictions = self.model.predict(processed_image, verbose=0)
        probabilities = predictions[0]
        
        # Get predicted digit and confidence
        predicted_digit = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_digit])
        
        # Get all probabilities
        all_probs = {str(i): float(probabilities[i]) for i in range(10)}
        
        return {
            'prediction': predicted_digit,
            'confidence': confidence,
            'probabilities': all_probs
        }
    
    def predict_batch(self, images):
        """
        Predict multiple images at once
        
        Args:
            images: List of images
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        for image in images:
            results.append(self.predict(image))
        return results
    
    def get_model_info(self):
        """Get model information"""
        return {
            'model_type': 'CNN',
            'input_shape': f'{self.img_rows}x{self.img_cols} grayscale',
            'num_classes': 10,
            'framework': 'TensorFlow/Keras',
            'model_loaded': self.model is not None
        }
