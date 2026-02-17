# 🚀 BankForm-AI Quick Start Guide

Get up and running with **BankForm-AI** in minutes! This guide will help you start the application, which now includes a powerful **Hybrid OCR Engine** (EasyOCR + Tesseract + CNN).

---

## 📋 Prerequisites

- **Python 3.8+** installed on your system.
- **Visual C++ Redistributable** (Windows only, usually installed, but required for some Python libraries).

---

## 🛠️ Step 1: Install Dependencies

1. Open your terminal or command prompt.
2. Navigate to the `backend` directory:
   ```bash
   cd backend
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: This installs EasyOCR, Flask, PyTorch (CPU), and other dependencies. It may take a few minutes.)*

---

## 🏃 Step 2: Start the Server

Run the Flask application from the `backend` folder:

```bash
python app.py
```

**What happens next?**
- **First Run:** EasyOCR will download its recognition models (~100MB). **Please wait** for this to complete.
- **Initialization:** The server initializes the database, OCR engines, and the CNN digit model.
- **Ready:** You will see the server start message:

```text
============================================================
 BANKFORM-AI SERVER STARTING...
============================================================
Server: http://localhost:5050
...
 * Running on http://127.0.0.1:5050
```

---

## 🌐 Step 3: Use the Application

1. Open your web browser (Chrome, Firefox, Edge).
2. Go to: **[http://localhost:5050](http://localhost:5050)**
3. You will see the **BankForm-AI** interface.

### How to Process a Form:
1. **Drag & Drop** your bank challan image (JPG, PNG) into the upload area.
   - *Tip: Ensure the image is fairly clear and legible.*
2. Click **"Process Form"**.
3. View the extracted data!
   - **Green** confidence means high accuracy.
   - **Yellow/Red** indicates needs review.
   - Fields: Account Number, Amount, Date, Reference No.

---

## 🧠 (Optional) Train the CNN Model

While EasyOCR handles most text, you can train the specialized CNN model for handwritten digits to improve accuracy on specific digit-only fields.

```bash
cd backend
python train_model.py
```
- This downloads the MNIST dataset and trains a model.
- Saved to: `backend/models/trained/mnist_cnn.h5`.

---

## ⚙️ Troubleshooting

### ❌ "EasyOCR not found" or Import Errors
- Ensure you installed requirements: `pip install -r requirements.txt`
- If you see `Microsoft Visual C++ 14.0 is required`, download and install the **Visual C++ Build Tools**.

### ❌ Server won't start
- Check if port **5050** is free.
- Check the terminal output for specific error messages.

### ❌ Poor Recognition Accuracy
- **Image Quality:** Ensure the photo is well-lit and not blurry.
- **Orientation:** The form should be upright (not rotated 90 degrees).

---

## 📁 Project Structure

- **`backend/`**: Flask API, OCR Logic (`services/ocr_service.py`), Models.
- **`frontend/`**: HTML/CSS/JS (served by the backend).
- **`uploads/`**: Stores processed images.
- **`database/`**: SQLite database (`bankform.db`).

---

**🎉 You're all set! Enjoy BankForm-AI!**
