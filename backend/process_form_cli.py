"""
Form Processing CLI Tool
Command-line interface for processing banking forms
Usage: python process_form_cli.py <image_path> [form_type] [--save] [--export]
"""

import sys
import os
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from process_forms import BankingFormProcessor
from database.form_store import FormDataStore
from config import get_config


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Process banking form images and extract field data'
    )
    
    parser.add_argument(
        'image',
        help='Path to form image file'
    )
    
    parser.add_argument(
        '--type', '-t',
        default='withdrawal',
        choices=['withdrawal', 'deposit', 'transfer'],
        help='Form type (default: withdrawal)'
    )
    
    parser.add_argument(
        '--save', '-s',
        action='store_true',
        help='Save results to database'
    )
    
    parser.add_argument(
        '--export', '-e',
        action='store_true',
        help='Export to CSV after processing'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='extracted_forms',
        help='Output directory for JSON results'
    )
    
    args = parser.parse_args()
    
    # Validate image file
    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        return 1
    
    print("\n" + "=" * 70)
    print("BANKING FORM PROCESSOR")
    print("=" * 70)
    
    try:
        # Initialize processor
        config = get_config()
        processor = BankingFormProcessor(config.MODEL_PATH)
        
        # Process form
        result = processor.process_form_image(args.image, form_type=args.type)
        
        # Print results
        processor.print_results(result)
        
        # Save to JSON
        json_path = processor.save_results(result, args.output)
        
        # Save to database
        if args.save:
            store = FormDataStore()
            form_id = store.save_form(result)
            if form_id:
                print(f"[DATABASE] Form saved with ID: {form_id}")
        
        # Export to CSV
        if args.export:
            store = FormDataStore()
            store.export_to_csv('extracted_forms.csv')
        
        return 0
    
    except Exception as e:
        print(f"[ERROR] Processing failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
