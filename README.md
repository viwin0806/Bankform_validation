# 🏦 BankForm-AI

**Intelligent Banking Challan Processing System**

Transform handwritten banking forms into digital data using state-of-the-art deep learning and computer vision. BankForm-AI automates the extraction of digits from deposit slips, withdrawal forms, and fund transfer documents with industry-leading accuracy.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)

---

## ✨ Features

### 🎯 Core Capabilities
- **Multi-Field Digit Recognition** - Extract all numeric fields from scanned forms automatically
- **Template-Based Extraction** - Support for common Indian banking form formats
- **Intelligent Validation** - Banking-specific rules for account numbers, amounts, dates
- **Confidence Scoring** - Automatic flagging of low-confidence extractions for review
- **Batch Processing** - Process multiple forms efficiently
- **Multiple Export Formats** - CSV, JSON, Excel with customizable templates

### 🏦 Banking Features
- Support for deposit slips, withdrawal forms, and fund transfers
- Account number validation with check digit verification
- Amount validation with configurable limits
- Date format validation
- IFSC code validation
- Audit trail for all processing activities

### 💎 Premium UX
- Modern dark-mode banking UI
- Drag-and-drop file upload
- Real-time processing visualization
- Interactive results with edit capability
- Dashboard for review and approval
- Mobile-responsive design

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Modern web browser (Chrome, Firefox, Edge)
- 4GB RAM minimum (8GB recommended for training)

### Installation

1. **Navigate to project directory:**
   ```bash
   cd BankForm-AI
   ```

2. **Install backend dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Train the digit recognition model:**
   ```bash
   python train_model.py
   ```
   *This will download MNIST data and train a CNN model (~5-10 minutes)*

### Running the Application

1. **Start the backend server:**
   ```bash
   cd backend
   python app.py
   ```
   Server runs at `http://localhost:5000`

2. **Open the frontend:**
   - Option A: Open `frontend/index.html` in your browser
   - Option B: Use a local server:
     ```bash
     cd frontend
     python -m http.server 8000
     ```
     Navigate to `http://localhost:8000`

3. **Start processing forms!**
   - Select a form template
   - Upload a scanned challan
   - View extracted data with confidence scores
   - Export results

---

## 📁 Project Structure

```
BankForm-AI/
├── backend/
│   ├── app.py                      # Flask REST API server
│   ├── config.py                   # Configuration settings
│   ├── train_model.py              # Model training script
│   ├── requirements.txt            # Python dependencies
│   │
│   ├── models/
│   │   ├── digit_model.py          # CNN digit recognition
│   │   ├── form_detector.py        # Field detection & extraction
│   │   └── trained/                # Saved models
│   │       └── mnist_cnn.h5
│   │
│   ├── services/
│   │   ├── ocr_service.py          # Multi-field OCR orchestration
│   │   ├── validation_service.py   # Banking validation rules
│   │   └── export_service.py       # Data export utilities
│   │
│   ├── database/
│   │   └── models.py               # SQLAlchemy models
│   │
│   ├── templates/
│   │   ├── deposit_slip.json       # Form template definitions
│   │   ├── withdrawal_form.json
│   │   └── fund_transfer.json
│   │
│   └── uploads/                    # Uploaded form images
│
├── frontend/
│   ├── index.html                  # Main application
│   ├── dashboard.html              # Admin dashboard
│   │
│   └── assets/
│       ├── css/
│       │   └── style.css           # Premium banking UI
│       ├── js/
│       │   └── app.js              # Application logic
│       └── images/
│           └── sample-challans/    # Demo forms
│
├── docs/
│   ├── API.md                      # API documentation
│   ├── USER_GUIDE.md               # User manual
│   └── DEPLOYMENT.md               # Deployment guide
│
├── tests/
│   └── sample_forms/               # Test images
│
└── README.md                       # This file
```

---

## 🎯 How It Works

### Processing Pipeline

```
┌─────────────┐
│Upload Form  │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│Detect Form Type │ (Template matching or auto-detect)
└──────┬──────────┘
       │
       ▼
┌──────────────────┐
│Extract Fields    │ (Computer vision segmentation)
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│Recognize Digits  │ (CNN-based OCR)
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│Validate Data     │ (Banking rules validation)
└──────┬───────────┘
       │
       ▼
┌─────────────────────┐
│Confidence Check     │
└─────┬──────┬────────┘
      │      │
  High│      │Low
      │      │
      ▼      ▼
 ┌──────┐ ┌──────────────┐
 │Approve│ │Flag for Review│
 └───┬──┘ └──────┬───────┘
     │           │
     └─────┬─────┘
           ▼
    ┌────────────┐
    │Export Data │
    └────────────┘
```

### Confidence Scoring

| Confidence | Status | Action |
|-----------|--------|--------|
| 90-100% | ✅ High | Auto-approve |
| 70-89% | ⚠️ Medium | Review recommended |
| <70% | ❌ Low | Manual review required |

---

## 📊 Supported Form Types

### 1. **Deposit Slip**
- Account Number
- Deposit Amount
- Date
- Reference Number

### 2. **Withdrawal Form**
- Account Number
- Withdrawal Amount
- Date
- Cheque Number (optional)

### 3. **Fund Transfer**
- Source Account Number
- Destination Account Number
- Transfer Amount
- Date
- Reference Number

### Custom Templates
Create custom templates by adding JSON files to `backend/templates/` following this format:

```json
{
  "name": "Your Form Name",
  "type": "custom_type",
  "fields": [
    {
      "id": "field_name",
      "type": "numeric",
      "validation": "account_number",
      "bbox": {"x": 0.15, "y": 0.25, "width": 0.40, "height": 0.08}
    }
  ]
}
```

---

## 🔧 Configuration

Edit `backend/config.py` to customize:

```python
# Confidence Thresholds
CONFIDENCE_THRESHOLD_HIGH = 0.90  # Auto-approve threshold
CONFIDENCE_THRESHOLD_LOW = 0.70   # Review threshold

# Validation Rules
ACCOUNT_NUMBER_MIN_LENGTH = 9
ACCOUNT_NUMBER_MAX_LENGTH = 18
MAX_AMOUNT = 10000000  # ₹1 Crore

# File Upload
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf', 'tiff'}
```

---

## 🛠️ Technology Stack

### Backend
| Technology | Purpose |
|-----------|---------|
| **Flask** | REST API framework |
| **TensorFlow/Keras** | Deep learning framework |
| **OpenCV** | Computer vision & image processing |
| **SQLAlchemy** | Database ORM |
| **Pandas** | Data export |

### Frontend
| Technology | Purpose |
|-----------|---------|
| **HTML5** | Structure |
| **CSS3** | Premium banking UI |
| **Vanilla JavaScript** | Application logic |

### AI/ML
| Component | Description |
|-----------|-------------|
| **CNN Architecture** | Custom 5-layer convolutional network |
| **Training Data** | MNIST dataset (60,000 samples) |
| **Accuracy** | ~98-99% on test set |

---

## 📖 API Endpoints

### Health Check
```
GET /
```

### Upload Form
```
POST /api/upload
Content-Type: multipart/form-data
Body: file
```

### Process Form
```
POST /api/process
Content-Type: application/json
Body: {
  "filepath": "string",
  "form_type": "deposit_slip|withdrawal_form|transfer",
  "use_template": boolean
}
```

### Get Templates
```
GET /api/templates
```

### Get History
```
GET /api/history?limit=50&status=approved
```

### Export Data
```
POST /api/export
Content-Type: application/json
Body: {
  "format": "csv|json|excel",
  "status": "approved|flagged|all"
}
```

### Get Statistics
```
GET /api/stats
```

*For complete API documentation, see [docs/API.md](docs/API.md)*

---

## 🧪 Testing

Run tests with sample forms:
```bash
cd tests
pytest test_api.py -v
pytest test_ocr.py -v
```

---

## 🚀 Deployment

### Production Checklist

- [ ] Set `FLASK_ENV=production`
- [ ] Configure secret key
- [ ] Use PostgreSQL instead of SQLite
- [ ] Set up HTTPS/SSL
- [ ] Configure CORS for production domain
- [ ] Set up backup for database
- [ ] Configure logging
- [ ] Set up monitoring

*See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for detailed instructions*

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.



## 🙏 Acknowledgments

- **MNIST Dataset** by Yann LeCun
- **TensorFlow** team for the amazing framework
- **OpenCV** community
- **Flask** for the lightweight web framework

---

## 📧 Support

For issues and questions:
- Create an issue on GitHub
- Email: viwinrajamanickam@gmail.com

