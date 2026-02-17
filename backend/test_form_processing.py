"""
Test Form Processing with Real Banking Forms
Complete testing suite for form detection and field extraction
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from process_forms import BankingFormProcessor
from database.form_store import FormDataStore, print_statistics
from config import get_config


def test_form_processing():
    """Test complete form processing pipeline"""
    print("\n" + "=" * 70)
    print("BANKING FORM PROCESSOR TEST SUITE")
    print("=" * 70)
    
    config = get_config()
    processor = BankingFormProcessor(config.MODEL_PATH)
    store = FormDataStore()
    
    # Look for sample forms
    sample_forms_dir = 'sample_forms'
    test_images = []
    
    if os.path.exists(sample_forms_dir):
        for file in os.listdir(sample_forms_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                test_images.append(os.path.join(sample_forms_dir, file))
    
    if not test_images:
        print("\n[INFO] No sample forms found in 'sample_forms' directory")
        print("[INFO] Place banking form images in: backend/sample_forms/")
        print("[INFO] Supported formats: JPG, PNG, BMP, GIF")
        return
    
    print(f"\n[INFO] Found {len(test_images)} form images to process")
    print("-" * 70)
    
    results = []
    successful = 0
    failed = 0
    
    # Process each image
    for image_path in test_images:
        filename = os.path.basename(image_path)
        print(f"\n[PROCESSING] {filename}")
        
        # Try to determine form type from filename
        form_type = 'withdrawal'
        if 'deposit' in filename.lower():
            form_type = 'deposit'
        elif 'transfer' in filename.lower():
            form_type = 'transfer'
        
        # Process form
        result = processor.process_form_image(image_path, form_type=form_type)
        
        if result['success']:
            successful += 1
            results.append(result)
            
            # Print results
            processor.print_results(result)
            
            # Save to JSON
            json_path = processor.save_results(result)
            
            # Save to database
            form_id = store.save_form(result)
            print(f"[DATABASE] Saved with ID: {form_id}")
        else:
            failed += 1
            print(f"[ERROR] {result['error']}")
        
        print("-" * 70)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total Forms Processed: {len(test_images)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {successful/len(test_images)*100:.1f}%")
    print("=" * 70)
    
    # Print statistics
    print_statistics()
    
    # Export to CSV
    print("\n[EXPORT] Creating CSV export...")
    store.export_to_csv('test_results.csv')
    print("[EXPORT] CSV export complete: test_results.csv")


def test_database():
    """Test database operations"""
    print("\n" + "=" * 70)
    print("DATABASE TEST")
    print("=" * 70)
    
    store = FormDataStore()
    
    # Get statistics
    stats = store.get_statistics()
    
    print(f"Total Forms: {stats['total_forms']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Success Rate: {stats['success_rate']:.1f}%")
    print(f"Average Confidence: {stats['average_confidence']:.1%}")
    print("\nBy Form Type:")
    for form_type, count in stats['by_type'].items():
        print(f"  {form_type}: {count} forms")
    
    # Get recent forms
    print("\nRecent Forms:")
    recent = store.get_all_forms(limit=5)
    for form in recent:
        print(f"  ID {form['id']}: {form['form_type']} ({form['extraction_timestamp']})")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '--stats':
            test_database()
        elif sys.argv[1] == '--help':
            print("Usage: python test_form_processing.py [--stats] [--help]")
            print("\nOptions:")
            print("  --stats     Show database statistics only")
            print("  --help      Show this help message")
    else:
        test_form_processing()
