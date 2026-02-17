"""
Form Data Storage
Store extracted form data in SQLite database
"""

import sqlite3
from datetime import datetime
from pathlib import Path
import json


class FormDataStore:
    """Store and retrieve banking form data"""
    
    def __init__(self, db_path='form_data.db'):
        """Initialize database connection"""
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create forms table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS forms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                form_type TEXT NOT NULL,
                image_path TEXT,
                extraction_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                overall_confidence REAL,
                success BOOLEAN DEFAULT 1,
                error_message TEXT,
                raw_data JSON
            )
        ''')
        
        # Create form fields table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS form_fields (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                form_id INTEGER NOT NULL,
                field_name TEXT NOT NULL,
                field_value TEXT,
                field_confidence REAL,
                individual_digits TEXT,
                FOREIGN KEY (form_id) REFERENCES forms(id)
            )
        ''')
        
        # Create extraction log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS extraction_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                form_type TEXT,
                status TEXT,
                message TEXT,
                processed_count INTEGER DEFAULT 1
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_form(self, form_result):
        """Save form extraction result to database"""
        if not form_result.get('success'):
            return self._save_error(form_result)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Insert form record
            cursor.execute('''
                INSERT INTO forms (form_type, image_path, overall_confidence, raw_data)
                VALUES (?, ?, ?, ?)
            ''', (
                form_result.get('form_type'),
                form_result.get('image_path'),
                form_result.get('overall_confidence'),
                json.dumps(form_result)
            ))
            
            form_id = cursor.lastrowid
            
            # Insert field records
            for field_detail in form_result.get('field_details', []):
                cursor.execute('''
                    INSERT INTO form_fields 
                    (form_id, field_name, field_value, field_confidence, individual_digits)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    form_id,
                    field_detail['field_name'],
                    form_result['extracted_data'].get(field_detail['field_name']),
                    field_detail.get('confidence'),
                    json.dumps(field_detail.get('individual_digits', []))
                ))
            
            # Log successful extraction
            cursor.execute('''
                INSERT INTO extraction_log (form_type, status, message)
                VALUES (?, ?, ?)
            ''', (
                form_result.get('form_type'),
                'SUCCESS',
                f"Form processed with {len(form_result.get('field_details', []))} fields"
            ))
            
            conn.commit()
            print(f"[DATABASE] Form saved with ID: {form_id}")
            return form_id
            
        except Exception as e:
            print(f"[ERROR] Failed to save form: {e}")
            conn.rollback()
            return None
        finally:
            conn.close()
    
    def _save_error(self, error_result):
        """Log extraction error"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO forms (success, error_message, raw_data)
                VALUES (?, ?, ?)
            ''', (
                False,
                error_result.get('error'),
                json.dumps(error_result)
            ))
            
            cursor.execute('''
                INSERT INTO extraction_log (status, message)
                VALUES (?, ?)
            ''', (
                'ERROR',
                error_result.get('error')
            ))
            
            conn.commit()
        finally:
            conn.close()
    
    def get_form(self, form_id):
        """Retrieve a form by ID"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            # Get form record
            cursor.execute('SELECT * FROM forms WHERE id = ?', (form_id,))
            form = cursor.fetchone()
            
            if not form:
                return None
            
            # Get form fields
            cursor.execute('SELECT * FROM form_fields WHERE form_id = ?', (form_id,))
            fields = cursor.fetchall()
            
            result = dict(form)
            result['fields'] = [dict(field) for field in fields]
            
            return result
        finally:
            conn.close()
    
    def get_forms_by_type(self, form_type, limit=10):
        """Get recent forms of a specific type"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT * FROM forms 
                WHERE form_type = ? AND success = 1
                ORDER BY extraction_timestamp DESC
                LIMIT ?
            ''', (form_type, limit))
            
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()
    
    def get_all_forms(self, limit=50):
        """Get all processed forms"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT * FROM forms
                ORDER BY extraction_timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()
    
    def get_statistics(self):
        """Get extraction statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Total forms
            cursor.execute('SELECT COUNT(*) as total FROM forms')
            total = cursor.fetchone()[0]
            
            # Successful forms
            cursor.execute('SELECT COUNT(*) as successful FROM forms WHERE success = 1')
            successful = cursor.fetchone()[0]
            
            # Failed forms
            failed = total - successful
            
            # By type
            cursor.execute('''
                SELECT form_type, COUNT(*) as count 
                FROM forms WHERE success = 1
                GROUP BY form_type
            ''')
            by_type = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Average confidence
            cursor.execute('''
                SELECT AVG(overall_confidence) as avg_confidence 
                FROM forms WHERE success = 1
            ''')
            avg_confidence = cursor.fetchone()[0] or 0
            
            return {
                'total_forms': total,
                'successful': successful,
                'failed': failed,
                'success_rate': (successful / total * 100) if total > 0 else 0,
                'by_type': by_type,
                'average_confidence': avg_confidence
            }
        finally:
            conn.close()
    
    def export_to_csv(self, output_file='extracted_forms.csv'):
        """Export all extracted forms to CSV"""
        import csv
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT f.id, f.form_type, f.extraction_timestamp, f.overall_confidence,
                       ff.field_name, ff.field_value, ff.field_confidence
                FROM forms f
                LEFT JOIN form_fields ff ON f.id = ff.form_id
                WHERE f.success = 1
                ORDER BY f.id, ff.field_name
            ''')
            
            rows = cursor.fetchall()
            
            if not rows:
                print("[INFO] No data to export")
                return
            
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Form ID', 'Form Type', 'Timestamp', 'Confidence',
                                'Field Name', 'Field Value', 'Field Confidence'])
                
                for row in rows:
                    writer.writerow([
                        row['id'], row['form_type'], row['extraction_timestamp'],
                        f"{row['overall_confidence']:.2%}", row['field_name'],
                        row['field_value'], f"{row['field_confidence']:.2%}" if row['field_confidence'] else ''
                    ])
            
            print(f"[EXPORTED] {output_file}")
        finally:
            conn.close()


def print_statistics(db_path='form_data.db'):
    """Print database statistics"""
    store = FormDataStore(db_path)
    stats = store.get_statistics()
    
    print("\n" + "=" * 60)
    print("FORM DATA STATISTICS")
    print("=" * 60)
    print(f"Total Forms Processed: {stats['total_forms']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Success Rate: {stats['success_rate']:.1f}%")
    print(f"Average Confidence: {stats['average_confidence']:.1%}")
    print("\nBy Form Type:")
    for form_type, count in stats['by_type'].items():
        print(f"  {form_type}: {count}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    print_statistics()
