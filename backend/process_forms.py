"""
Real Banking Form Processor (Hybrid OCR)
Process actual Indian banking challans (withdrawal, deposit, transfer)
Uses EasyOCR + Tesseract + CNN pipeline for maximum accuracy
"""

import cv2
import numpy as np
import os
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from services.ocr_service import OCRService
from config import get_config


class BankingFormProcessor:
    """Process real banking forms using hybrid OCR"""
    
    def __init__(self, config=None):
        """Initialize with hybrid OCR service"""
        self.config = config or get_config()
        
        self.ocr_service = OCRService(
            model_path=self.config.MODEL_PATH,
            template_dir=Path(__file__).parent / 'templates',
            config=self.config
        )
    
    def process_form_image(self, image_path, form_type='deposit', use_template=False):
        """
        Process a banking form image and extract all fields
        
        Args:
            image_path: Path to form image
            form_type: Type of form (deposit, withdrawal, transfer)
            use_template: Whether to use template-based detection
        
        Returns:
            Dictionary with extracted form data
        """
        print(f"\n{'='*60}")
        print(f"[PROCESSING] {os.path.basename(image_path)}")
        print(f"Form Type: {form_type}")
        print(f"Template: {'Yes' if use_template else 'No (auto-detect)'}")
        print(f"{'='*60}")
        
        try:
            # Get template file if using templates
            template_file = None
            if use_template:
                template_dir = Path(__file__).parent / 'templates'
                for name in [f'{form_type}.json', f'{form_type}_form.json', 
                           f'{form_type}_slip.json', f'withdrawal_form.json',
                           f'deposit_slip.json', f'fund_transfer.json']:
                    candidate = template_dir / name
                    if candidate.exists():
                        template_file = str(candidate)
                        break
            
            # Process with hybrid OCR
            result = self.ocr_service.process_form(
                image_path,
                form_type=form_type,
                template_file=template_file
            )
            
            # Add metadata
            result['image_path'] = str(image_path)
            result['timestamp'] = datetime.now().isoformat()
            
            # Build extracted_data dict (for backward compatibility)
            result['extracted_data'] = {}
            for field in result.get('fields', []):
                name = field['field_name']
                value = field.get('corrected_value', field.get('extracted_value', ''))
                result['extracted_data'][name] = value
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def save_results(self, result, output_dir='extracted_forms'):
        """Save extraction results to JSON"""
        os.makedirs(output_dir, exist_ok=True)
        
        if not result.get('success'):
            print(f"[ERROR] {result.get('error', 'Unknown error')}")
            return None
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{result.get('form_type', 'form')}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Remove non-serializable data
        save_data = {}
        for key, value in result.items():
            if key in ('all_detections',):
                continue  # Skip large data
            try:
                json.dumps(value)
                save_data[key] = value
            except (TypeError, ValueError):
                save_data[key] = str(value)
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"[SAVED] {filepath}")
        return filepath
    
    def print_results(self, result):
        """Print extraction results in readable format"""
        if not result.get('success'):
            print(f"\n[ERROR] Failed to process form: {result.get('error', 'Unknown')}")
            return
        
        print(f"\n{'='*60}")
        print(f"  FORM PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"  Form Type      : {result.get('form_type', 'Unknown')}")
        print(f"  OCR Engine     : {result.get('ocr_engine_used', 'Unknown')}")
        print(f"  Confidence     : {result.get('overall_confidence', 0):.1%}")
        print(f"  Fields Found   : {len(result.get('fields', []))}")
        
        if result.get('raw_text'):
            print(f"\n  Raw Text Detected:")
            print(f"  {'-'*50}")
            # Show first 200 chars of raw text
            raw = result['raw_text'][:200]
            print(f"  {raw}{'...' if len(result['raw_text']) > 200 else ''}")
        
        print(f"\n  Extracted Fields:")
        print(f"  {'-'*50}")
        
        for field in result.get('fields', []):
            name = field.get('field_name', 'Unknown')
            value = field.get('extracted_value', 'N/A')
            conf = field.get('confidence', 0)
            valid = field.get('is_valid', None)
            
            status = ""
            if valid is True:
                status = "[OK]"
            elif valid is False:
                status = "[!!]"
            else:
                status = "[--]"
            
            label = field.get('label_text', '')
            label_info = f" (label: '{label}')" if label else ''
            
            print(f"  {status} {name:<25}: {value:<25} ({conf:.0%}){label_info}")
        
        print(f"{'='*60}\n")


if __name__ == '__main__':
    config = get_config()
    processor = BankingFormProcessor(config)
    
    # Test with a form image
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        form_type = sys.argv[2] if len(sys.argv) > 2 else 'deposit'
        use_template = '--template' in sys.argv
        
        result = processor.process_form_image(image_path, form_type, use_template)
        processor.print_results(result)
        processor.save_results(result)
    else:
        # Try sample forms
        sample_dir = Path(__file__).parent / 'sample_forms'
        if sample_dir.exists():
            for img_file in sample_dir.glob('*'):
                if img_file.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tiff', '.bmp'):
                    result = processor.process_form_image(str(img_file))
                    processor.print_results(result)
        else:
            print("\n[INFO] Usage:")
            print("  python process_forms.py <image_path> [form_type] [--template]")
            print("\n  Examples:")
            print("  python process_forms.py challan.jpg deposit")
            print("  python process_forms.py challan.jpg deposit --template")
            print("  python process_forms.py withdrawal.png withdrawal")
