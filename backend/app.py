"""
BankForm-AI Main Application
Flask REST API for banking form processing
"""

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from pathlib import Path
from datetime import datetime

from config import get_config
from database.models import db, init_db, ProcessedForm, FormField, AuditLog
from models import DigitRecognitionModel, FormDetector
from services import OCRService, ValidationService, ExportService

# Initialize Flask app — serve frontend from ../frontend
FRONTEND_DIR = Path(__file__).parent.parent / 'frontend'
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path='')
config = get_config()
app.config.from_object(config)

# Enable CORS
CORS(app, resources={r"/api/*": {"origins": config.CORS_ORIGINS}})

# Initialize database
init_db(app)

# Initialize services
BASE_DIR = Path(__file__).parent
model_path = config.MODEL_PATH
ocr_service = None
validation_service = ValidationService(config.__dict__)
export_service = ExportService(export_dir=BASE_DIR / 'exports')

def init_services():
    """Initialize AI services with hybrid OCR engine"""
    global ocr_service
    
    try:
        # CNN model is optional now — EasyOCR/Tesseract handle most work
        if not os.path.exists(model_path):
            print(f"[INFO] CNN model not found at {model_path}")
            print("   This is OK — EasyOCR/Tesseract will handle OCR")
            print("   Run train_model.py if you want handwritten digit support")
        
        ocr_service = OCRService(
            model_path=model_path,
            template_dir=BASE_DIR / 'templates',
            config=config
        )
        print("[OK] OCR Service initialized successfully")
        return True
    except Exception as e:
        print(f"[ERROR] Error initializing services: {e}")
        return False

# Helper functions
def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

def save_uploaded_file(file):
    """Save uploaded file and return path"""
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Add timestamp to avoid conflicts
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(config.UPLOAD_FOLDER, filename)
        file.save(filepath)
        return filepath
    return None

# API Routes

@app.route('/', methods=['GET'])
def serve_frontend():
    """Serve the frontend index.html"""
    return send_from_directory(str(FRONTEND_DIR), 'index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'BankForm-AI API',
        'version': '1.0.0',
        'ocr_service_loaded': ocr_service is not None
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload challan image"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    filepath = save_uploaded_file(file)
    
    if filepath:
        return jsonify({
            'success': True,
            'filepath': filepath,
            'filename': os.path.basename(filepath)
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/process', methods=['POST'])
def process_form():
    """Process uploaded form using hybrid OCR pipeline"""
    if not ocr_service:
        return jsonify({'error': 'OCR service not initialized. Check server logs.'}), 503
    
    data = request.get_json()
    
    if not data or 'filepath' not in data:
        return jsonify({'error': 'No filepath provided'}), 400
    
    filepath = data['filepath']
    form_type = data.get('form_type', 'generic')
    use_template = data.get('use_template', False)
    
    try:
        # Get template file if specified
        template_file = None
        if use_template and form_type != 'generic':
            template_file = BASE_DIR / 'templates' / f'{form_type}.json'
            if not template_file.exists():
                template_file = None
        
        # Process form with hybrid OCR
        result = ocr_service.process_form(
            filepath,
            form_type=form_type,
            template_file=str(template_file) if template_file else None
        )
        
        # Validate fields
        validated_fields = []
        for field in result.get('fields', []):
            # Get the extracted value (may already be set by OCR)
            extracted_value = field.get('extracted_value', field.get('corrected_value', ''))
            
            validation_result = validation_service.validate_field(
                field['field_name'],
                extracted_value,
                field['field_type'],
                field.get('validation')
            )
            
            field.update({
                'is_valid': validation_result['is_valid'],
                'validation_message': validation_result['message'],
                'corrected_value': validation_result['corrected_value']
            })
            
            validated_fields.append(field)
        
        result['fields'] = validated_fields
        
        # Get confidence flag
        confidence_flag = validation_service.get_confidence_flag(
            result.get('overall_confidence', 0)
        )
        result['confidence_flag'] = confidence_flag
        
        # Save to database
        form_record = ProcessedForm(
            form_type=form_type,
            image_path=filepath,
            status=confidence_flag['status'].replace('auto_approve', 'approved')
                   .replace('review_recommended', 'flagged')
                   .replace('manual_review', 'flagged'),
            confidence_score=result.get('overall_confidence', 0)
        )
        db.session.add(form_record)
        db.session.flush()
        
        # Save fields
        for field in validated_fields:
            field_record = FormField(
                form_id=form_record.id,
                field_name=field['field_name'],
                field_type=field['field_type'],
                extracted_value=field.get('corrected_value', field.get('extracted_value', '')),
                original_value=field.get('extracted_value', ''),
                confidence=field.get('confidence', 0),
                is_valid=field.get('is_valid', True),
                validation_message=field.get('validation_message', ''),
                bbox_x=field['bbox'].get('x') if field.get('bbox') else None,
                bbox_y=field['bbox'].get('y') if field.get('bbox') else None,
                bbox_width=field['bbox'].get('width') if field.get('bbox') else None,
                bbox_height=field['bbox'].get('height') if field.get('bbox') else None
            )
            db.session.add(field_record)
        
        # Add audit log
        engine_used = result.get('ocr_engine_used', 'unknown')
        audit = AuditLog(
            form_id=form_record.id,
            action='created',
            user='system',
            details=f'Form processed via {engine_used} OCR pipeline'
        )
        db.session.add(audit)
        
        db.session.commit()
        
        result['form_id'] = form_record.id
        
        # Remove numpy arrays and image data before serializing
        for field in result.get('fields', []):
            field.pop('image', None)
            field.pop('label_bbox', None)
        result.pop('all_detections', None)  # Can be large
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/templates', methods=['GET'])
def get_templates():
    """Get available form templates"""
    templates_dir = BASE_DIR / 'templates'
    templates = []
    
    for template_file in templates_dir.glob('*.json'):
        import json
        with open(template_file, 'r') as f:
            template_data = json.load(f)
            templates.append({
                'id': template_file.stem,
                'name': template_data.get('name', template_file.stem),
                'type': template_data.get('type', 'generic'),
                'bank_name': template_data.get('bank_name', 'Generic')
            })
    
    return jsonify({'templates': templates})

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get processing history"""
    limit = request.args.get('limit', 50, type=int)
    status_filter = request.args.get('status', None)
    
    query = ProcessedForm.query.order_by(ProcessedForm.processed_at.desc())
    
    if status_filter:
        query = query.filter_by(status=status_filter)
    
    forms = query.limit(limit).all()
    
    return jsonify({
        'forms': [form.to_dict() for form in forms]
    })

@app.route('/api/form/<int:form_id>', methods=['GET'])
def get_form(form_id):
    """Get specific form details"""
    form = ProcessedForm.query.get_or_404(form_id)
    return jsonify(form.to_dict())

@app.route('/api/form/<int:form_id>/approve', methods=['POST'])
def approve_form(form_id):
    """Approve a form"""
    form = ProcessedForm.query.get_or_404(form_id)
    data = request.get_json()
    
    form.status = 'approved'
    form.reviewed_at = datetime.utcnow()
    form.reviewed_by = data.get('reviewer', 'admin')
    
    # Add audit log
    audit = AuditLog(
        form_id=form.id,
        action='approved',
        user=data.get('reviewer', 'admin'),
        details='Form manually approved'
    )
    db.session.add(audit)
    
    db.session.commit()
    
    return jsonify({'success': True, 'form': form.to_dict()})

@app.route('/api/form/<int:form_id>/reject', methods=['POST'])
def reject_form(form_id):
    """Reject a form"""
    form = ProcessedForm.query.get_or_404(form_id)
    data = request.get_json()
    
    form.status = 'rejected'
    form.reviewed_at = datetime.utcnow()
    form.reviewed_by = data.get('reviewer', 'admin')
    
    # Add audit log
    audit = AuditLog(
        form_id=form.id,
        action='rejected',
        user=data.get('reviewer', 'admin'),
        details=data.get('reason', 'Form manually rejected')
    )
    db.session.add(audit)
    
    db.session.commit()
    
    return jsonify({'success': True, 'form': form.to_dict()})

@app.route('/api/export', methods=['POST'])
def export_data():
    """Export data in various formats"""
    data = request.get_json()
    export_format = data.get('format', 'csv')
    status_filter = data.get('status', None)
    
    # Get forms
    query = ProcessedForm.query
    if status_filter:
        query = query.filter_by(status=status_filter)
    
    forms = query.all()
    forms_data = [form.to_dict() for form in forms]
    
    try:
        if export_format == 'csv':
            filepath = export_service.export_to_csv(forms_data)
        elif export_format == 'json':
            filepath = export_service.export_to_json(forms_data)
        elif export_format == 'excel':
            filepath = export_service.export_to_excel(forms_data)
        else:
            return jsonify({'error': 'Invalid export format'}), 400
        
        return send_file(filepath, as_attachment=True)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    forms = ProcessedForm.query.all()
    forms_data = [form.to_dict() for form in forms]
    
    stats = export_service.generate_summary_report(forms_data)
    
    return jsonify(stats)

@app.route('/api/dashboard', methods=['GET'])
def get_dashboard_stats():
    """Get detailed dashboard statistics"""
    forms = ProcessedForm.query.all()
    forms_data = [form.to_dict() for form in forms]
    
    stats = export_service.get_dashboard_stats(forms_data)
    
    return jsonify(stats)

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get model information"""
    if not ocr_service:
        return jsonify({'error': 'Model not loaded'}), 503
    
    model_info = ocr_service.digit_model.get_model_info()
    return jsonify(model_info)

@app.route('/api/ocr-status', methods=['GET'])
def get_ocr_status():
    """Get OCR engine status — which engines are available"""
    if not ocr_service:
        return jsonify({'error': 'OCR service not initialized'}), 503
    
    return jsonify(ocr_service.get_engine_status())

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Initialize services
    init_services()
    
    # Run app
    print("\n" + "="*60)
    print(" BANKFORM-AI SERVER STARTING...")
    print("="*60)
    print(f"Server: http://localhost:5050")
    print(f"Status: http://localhost:5050/")
    print(f"API Docs: See docs/API.md")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5050)
