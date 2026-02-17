"""
Database Models for BankForm-AI
SQLAlchemy models for storing processed forms and audit trails
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class ProcessedForm(db.Model):
    """Main table for processed banking forms"""
    __tablename__ = 'processed_forms'
    
    id = db.Column(db.Integer, primary_key=True)
    form_type = db.Column(db.String(50), nullable=False)  # deposit, withdrawal, transfer
    image_path = db.Column(db.String(255), nullable=False)
    status = db.Column(db.String(20), default='pending')  # pending, approved, flagged, rejected
    confidence_score = db.Column(db.Float)
    processed_at = db.Column(db.DateTime, default=datetime.utcnow)
    reviewed_at = db.Column(db.DateTime)
    reviewed_by = db.Column(db.String(100))
    
    # Relationships
    fields = db.relationship('FormField', backref='form', lazy=True, cascade='all, delete-orphan')
    audit_logs = db.relationship('AuditLog', backref='form', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'form_type': self.form_type,
            'image_path': self.image_path,
            'status': self.status,
            'confidence_score': self.confidence_score,
            'processed_at': self.processed_at.isoformat() if self.processed_at else None,
            'reviewed_at': self.reviewed_at.isoformat() if self.reviewed_at else None,
            'reviewed_by': self.reviewed_by,
            'fields': [field.to_dict() for field in self.fields]
        }

class FormField(db.Model):
    """Individual field extractions from forms"""
    __tablename__ = 'form_fields'
    
    id = db.Column(db.Integer, primary_key=True)
    form_id = db.Column(db.Integer, db.ForeignKey('processed_forms.id'), nullable=False)
    field_name = db.Column(db.String(100), nullable=False)  # account_number, amount, date, etc.
    field_type = db.Column(db.String(20), nullable=False)  # numeric, date, text
    extracted_value = db.Column(db.String(255))
    original_value = db.Column(db.String(255))  # Store original before corrections
    confidence = db.Column(db.Float)
    is_valid = db.Column(db.Boolean, default=True)
    validation_message = db.Column(db.String(255))
    bbox_x = db.Column(db.Integer)  # Bounding box coordinates
    bbox_y = db.Column(db.Integer)
    bbox_width = db.Column(db.Integer)
    bbox_height = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'field_name': self.field_name,
            'field_type': self.field_type,
            'extracted_value': self.extracted_value,
            'original_value': self.original_value,
            'confidence': self.confidence,
            'is_valid': self.is_valid,
            'validation_message': self.validation_message,
            'bbox': {
                'x': self.bbox_x,
                'y': self.bbox_y,
                'width': self.bbox_width,
                'height': self.bbox_height
            } if self.bbox_x is not None else None
        }

class AuditLog(db.Model):
    """Audit trail for form processing and modifications"""
    __tablename__ = 'audit_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    form_id = db.Column(db.Integer, db.ForeignKey('processed_forms.id'), nullable=False)
    action = db.Column(db.String(50), nullable=False)  # created, updated, approved, rejected
    user = db.Column(db.String(100))
    details = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'form_id': self.form_id,
            'action': self.action,
            'user': self.user,
            'details': self.details,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }

class FormTemplate(db.Model):
    """Template definitions for different form types"""
    __tablename__ = 'form_templates'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    form_type = db.Column(db.String(50), nullable=False)
    bank_name = db.Column(db.String(100))
    template_data = db.Column(db.Text)  # JSON string with field definitions
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, onupdate=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary"""
        import json
        return {
            'id': self.id,
            'name': self.name,
            'form_type': self.form_type,
            'bank_name': self.bank_name,
            'template_data': json.loads(self.template_data) if self.template_data else None,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

def init_db(app):
    """Initialize database with app context"""
    db.init_app(app)
    with app.app_context():
        db.create_all()
        print("[OK] Database initialized successfully")
