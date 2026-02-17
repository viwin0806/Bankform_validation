"""
Banking Form API Routes
REST API endpoints for form processing
"""

from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import io
import json
from datetime import datetime

# Assuming these are available in the main app
from process_forms import BankingFormProcessor
from database.form_store import FormDataStore
from config import get_config

# Create blueprint
form_api = Blueprint('form_api', __name__, url_prefix='/api/forms')

# Initialize components
config = get_config()
processor = BankingFormProcessor(config.MODEL_PATH)
store = FormDataStore()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
UPLOAD_FOLDER = 'uploads'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@form_api.route('/upload', methods=['POST'])
def upload_form():
    """
    Upload a banking form image and process it
    
    POST /api/forms/upload
    Content-Type: multipart/form-data
    
    Parameters:
    - image: Form image file (required)
    - form_type: Type of form - withdrawal, deposit, transfer (optional, default: withdrawal)
    - save_to_db: Save results to database (optional, default: true)
    """
    try:
        # Check if image in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image part in request'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Get form type from request
        form_type = request.form.get('form_type', 'withdrawal')
        if form_type not in ['withdrawal', 'deposit', 'transfer']:
            form_type = 'withdrawal'
        
        save_to_db = request.form.get('save_to_db', 'true').lower() == 'true'
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, saved_filename)
        file.save(filepath)
        
        # Process form
        result = processor.process_form_image(filepath, form_type=form_type)
        
        # Save to database if requested
        if save_to_db and result.get('success'):
            form_id = store.save_form(result)
            result['form_id'] = form_id
        
        return jsonify(result), 200 if result.get('success') else 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@form_api.route('/process', methods=['POST'])
def process_form_endpoint():
    """
    Process a form image from URL or base64
    
    POST /api/forms/process
    Content-Type: application/json
    
    Body:
    {
        "image_path": "path/to/image.jpg",
        "form_type": "withdrawal",
        "save_to_db": true
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'image_path' not in data:
            return jsonify({'error': 'image_path required in request body'}), 400
        
        image_path = data['image_path']
        form_type = data.get('form_type', 'withdrawal')
        save_to_db = data.get('save_to_db', True)
        
        # Validate file exists
        if not os.path.exists(image_path):
            return jsonify({'error': f'Image not found: {image_path}'}), 404
        
        # Process form
        result = processor.process_form_image(image_path, form_type=form_type)
        
        # Save to database if requested
        if save_to_db and result.get('success'):
            form_id = store.save_form(result)
            result['form_id'] = form_id
        
        return jsonify(result), 200 if result.get('success') else 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@form_api.route('/<int:form_id>', methods=['GET'])
def get_form(form_id):
    """
    Retrieve a previously processed form
    
    GET /api/forms/<form_id>
    """
    try:
        form = store.get_form(form_id)
        
        if not form:
            return jsonify({'error': 'Form not found'}), 404
        
        return jsonify(form), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@form_api.route('/type/<form_type>', methods=['GET'])
def get_forms_by_type(form_type):
    """
    Get recent forms of a specific type
    
    GET /api/forms/type/<form_type>?limit=10
    """
    try:
        limit = request.args.get('limit', 10, type=int)
        
        forms = store.get_forms_by_type(form_type, limit=limit)
        
        return jsonify({
            'form_type': form_type,
            'count': len(forms),
            'forms': forms
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@form_api.route('/all', methods=['GET'])
def get_all_forms():
    """
    Get all processed forms
    
    GET /api/forms/all?limit=50
    """
    try:
        limit = request.args.get('limit', 50, type=int)
        
        forms = store.get_all_forms(limit=limit)
        
        return jsonify({
            'count': len(forms),
            'forms': forms
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@form_api.route('/statistics', methods=['GET'])
def get_statistics():
    """
    Get form processing statistics
    
    GET /api/forms/statistics
    """
    try:
        stats = store.get_statistics()
        return jsonify(stats), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@form_api.route('/export/csv', methods=['GET'])
def export_csv():
    """
    Export all forms to CSV
    
    GET /api/forms/export/csv
    """
    try:
        # Export to temporary file
        csv_filename = 'extracted_forms.csv'
        store.export_to_csv(csv_filename)
        
        # Send file
        return send_file(
            csv_filename,
            mimetype='text/csv',
            as_attachment=True,
            download_name=csv_filename
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@form_api.route('/export/json', methods=['GET'])
def export_json():
    """
    Export all forms to JSON
    
    GET /api/forms/export/json
    """
    try:
        forms = store.get_all_forms(limit=1000)
        
        output = io.StringIO()
        json.dump(forms, output, indent=2, default=str)
        
        return output.getvalue(), 200, {
            'Content-Type': 'application/json',
            'Content-Disposition': 'attachment; filename=extracted_forms.json'
        }
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Register blueprint
def register_form_routes(app):
    """Register form routes with Flask app"""
    app.register_blueprint(form_api)
    print("[API] Form processing routes registered")
