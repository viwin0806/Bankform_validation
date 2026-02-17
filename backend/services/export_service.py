"""
Export Service - Data export functionality
Export processed form data to various formats
"""

import pandas as pd
import json
from datetime import datetime
from pathlib import Path

class ExportService:
    """Service for exporting processed form data"""
    
    def __init__(self, export_dir='exports'):
        """
        Initialize export service
        
        Args:
            export dir: Directory for exported files
        """
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(exist_ok=True)
    
    def export_to_csv(self, forms_data, filename=None):
        """
        Export data to CSV format
        
        Args:
            forms_data: List of processed form dictionaries
            filename: Output filename (auto-generated if None)
        
        Returns:
            Path to exported file
        """
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'export_{timestamp}.csv'
        
        filepath = self.export_dir / filename
        
        # Convert to flat structure for CSV
        rows = []
        for form in forms_data:
            row = {
                'form_id': form.get('id', ''),
                'form_type': form.get('form_type', ''),
                'processed_at': form.get('processed_at', ''),
                'status': form.get('status', ''),
                'confidence_score': form.get('confidence_score', 0)
            }
            
            # Add field values
            for field in form.get('fields', []):
                field_name = field.get('field_name', 'unknown')
                row[field_name] = field.get('extracted_value', '')
                row[f'{field_name}_confidence'] = field.get('confidence', 0)
            
            rows.append(row)
        
        # Create DataFrame and export
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        
        return str(filepath)
    
    def export_to_json(self, forms_data, filename=None):
        """
        Export data to JSON format
        
        Args:
            forms_data: List of processed form dictionaries
            filename: Output filename (auto-generated if None)
        
        Returns:
            Path to exported file
        """
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'export_{timestamp}.json'
        
        filepath = self.export_dir / filename
        
        # Prepare data
        export_data = {
            'export_date': datetime.now().isoformat(),
            'total_forms': len(forms_data),
            'forms': forms_data
        }
        
        # Write JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def export_to_excel(self, forms_data, filename=None):
        """
        Export data to Excel format with multiple sheets
        
        Args:
            forms_data: List of processed form dictionaries
            filename: Output filename (auto-generated if None)
        
        Returns:
            Path to exported file
        """
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'export_{timestamp}.xlsx'
        
        filepath = self.export_dir / filename
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Summary sheet
            summary_rows = []
            for form in forms_data:
                summary_rows.append({
                    'Form ID': form.get('id', ''),
                    'Type': form.get('form_type', ''),
                    'Status': form.get('status', ''),
                    'Confidence': form.get('confidence_score', 0),
                    'Processed At': form.get('processed_at', '')
                })
            
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Detailed data sheet
            detail_rows = []
            for form in forms_data:
                row = {
                    'form_id': form.get('id', ''),
                    'form_type': form.get('form_type', ''),
                    'processed_at': form.get('processed_at', ''),
                    'status': form.get('status', ''),
                    'confidence_score': form.get('confidence_score', 0)
                }
                
                for field in form.get('fields', []):
                    field_name = field.get('field_name', 'unknown')
                    row[field_name] = field.get('extracted_value', '')
                    row[f'{field_name}_confidence'] = field.get('confidence', 0)
                    row[f'{field_name}_valid'] = field.get('is_valid', True)
                
                detail_rows.append(row)
            
            detail_df = pd.DataFrame(detail_rows)
            detail_df.to_excel(writer, sheet_name='Detailed Data', index=False)
            
            # Flagged items sheet (low confidence)
            flagged = [form for form in forms_data 
                      if form.get('confidence_score', 1.0) < 0.70]
            
            if flagged:
                flagged_rows = []
                for form in flagged:
                    for field in form.get('fields', []):
                        if field.get('confidence', 1.0) < 0.70:
                            flagged_rows.append({
                                'Form ID': form.get('id', ''),
                                'Field Name': field.get('field_name', ''),
                                'Extracted Value': field.get('extracted_value', ''),
                                'Confidence': field.get('confidence', 0),
                                'Reason': 'Low confidence - Manual review required'
                            })
                
                if flagged_rows:
                    flagged_df = pd.DataFrame(flagged_rows)
                    flagged_df.to_excel(writer, sheet_name='Flagged Items', index=False)
        
        return str(filepath)
    
    def export_for_banking_system(self, forms_data, system_type='generic'):
        """
        Export in banking system-specific format
        
        Args:
            forms_data: List of processed form dictionaries
            system_type: Type of banking system (generic, sbi, hdfc, etc.)
        
        Returns:
            Path to exported file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{system_type}_import_{timestamp}.csv'
        filepath = self.export_dir / filename
        
        # Banking system format (simplified example)
        rows = []
        for form in forms_data:
            # Extract standard banking fields
            fields_dict = {f.get('field_name'): f.get('extracted_value') 
                          for f in form.get('fields', [])}
            
            row = {
                'TRANSACTION_ID': form.get('id', ''),
                'ACCOUNT_NUMBER': fields_dict.get('account_number', ''),
                'AMOUNT': fields_dict.get('amount', ''),
                'DATE': fields_dict.get('date', ''),
                'TYPE': form.get('form_type', '').upper(),
                'STATUS': 'PENDING' if form.get('confidence_score', 0) < 0.90 else 'APPROVED'
            }
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        
        return str(filepath)
    
    def generate_summary_report(self, forms_data):
        """
        Generate summary statistics
        
        Args:
            forms_data: List of processed form dictionaries
        
        Returns:
            Summary dictionary
        """
        total = len(forms_data)
        
        if total == 0:
            return {'total_forms': 0}
        
        high_confidence = sum(1 for f in forms_data 
                            if f.get('confidence_score', 0) >= 0.90)
        medium_confidence = sum(1 for f in forms_data 
                               if 0.70 <= f.get('confidence_score', 0) < 0.90)
        low_confidence = sum(1 for f in forms_data 
                            if f.get('confidence_score', 0) < 0.70)
        
        by_type = {}
        for form in forms_data:
            form_type = form.get('form_type', 'unknown')
            by_type[form_type] = by_type.get(form_type, 0) + 1
        
        by_status = {}
        for form in forms_data:
            status = form.get('status', 'unknown')
            by_status[status] = by_status.get(status, 0) + 1
        
        avg_confidence = sum(f.get('confidence_score', 0) for f in forms_data) / total
        
        return {
            'total_forms': total,
            'high_confidence': high_confidence,
            'medium_confidence': medium_confidence,
            'low_confidence': low_confidence,
            'by_type': by_type,
            'by_status': by_status,
            'average_confidence': round(avg_confidence, 4),
            'auto_approved': sum(1 for f in forms_data 
                               if f.get('status') == 'approved'),
            'flagged_for_review': sum(1 for f in forms_data 
                                     if f.get('status') == 'flagged')
        }

    def get_dashboard_stats(self, forms_data):
        """
        Get detailed dashboard statistics
        """
        summary = self.generate_summary_report(forms_data)
        
        # 1. Recent Transactions (Last 5)
        # Sort by processed_at desc
        sorted_forms = sorted(forms_data, key=lambda x: x.get('processed_at', ''), reverse=True)
        recent_transactions = []
        
        for form in sorted_forms[:5]:
            amount = '0'
            # Find amount field
            for field in form.get('fields', []):
                if field.get('field_name') == 'amount':
                    amount = field.get('extracted_value', '0')
                    break
            
            recent_transactions.append({
                'id': form.get('id'),
                'type': form.get('form_type'),
                'date': form.get('processed_at'),
                'amount': amount,
                'status': form.get('status'),
                'confidence': form.get('confidence_score')
            })
            
        # 2. Daily Volume (Last 7 days)
        from collections import defaultdict
        daily_volume = defaultdict(int)
        today = datetime.now().date()
        
        for form in forms_data:
            try:
                # Parse date assuming ISO format or similar
                processed_at = form.get('processed_at')
                if isinstance(processed_at, str):
                    dt = datetime.fromisoformat(processed_at)
                else:
                    dt = processed_at
                
                date_str = dt.strftime('%Y-%m-%d')
                daily_volume[date_str] += 1
            except:
                pass
            
        # Ensure last 7 days are present even if 0
        dates = []
        counts = []
        for i in range(6, -1, -1):
            date = (datetime.now() - pd.Timedelta(days=i)).strftime('%Y-%m-%d')
            dates.append(date)
            counts.append(daily_volume.get(date, 0))
            
        # 3. Total Processed Amount
        total_amount = 0.0
        for form in forms_data:
            for field in form.get('fields', []):
                if field.get('field_name') == 'amount':
                    try:
                        val = float(str(field.get('extracted_value', '0')).replace(',', ''))
                        total_amount += val
                    except:
                        pass
                        
        summary['recent_transactions'] = recent_transactions
        summary['daily_trend'] = {'dates': dates, 'counts': counts}
        summary['total_processed_amount'] = total_amount
        
        return summary
