"""
Smart Form Field Detection for Indian Bank Challans
Uses EasyOCR text detection + keyword matching to identify fields.
Falls back to template-based detection when templates are available.
"""

import cv2
import numpy as np
from PIL import Image
import json
import re
import math


class FormDetector:
    """Detect and extract fields from Indian banking challans/forms"""
    
    # Default field keywords for Indian bank challans
    DEFAULT_FIELD_KEYWORDS = {
        'account_number': ['account', 'a/c', 'acct', 'acc no', 'account no', 'account number', 'a/c no', 'ac no'],
        'amount': ['amount', 'amt', 'rs', 'rupees', 'total', 'sum', 'rs.', 'amount rs', 'total amount'],
        'date': ['date', 'dt', 'dated', 'dd/mm/yyyy', 'dd-mm-yyyy'],
        'name': ['name', 'depositor', 'account holder', 'a/c holder', 'customer name', 'applicant', 'deposited by'],
        'branch': ['branch', 'br', 'branch name', 'branch code'],
        'ifsc': ['ifsc', 'ifsc code', 'ifs code', 'micr'],
        'cheque_number': ['cheque', 'chq', 'check', 'cheque no', 'chq no', 'instrument no', 'instrument'],
        'reference_number': ['reference', 'ref', 'ref no', 'transaction', 'txn', 'slip no'],
        'pan': ['pan', 'pan no', 'pan number', 'pan card'],
        'mobile': ['mobile', 'phone', 'contact', 'mob', 'tel', 'cell'],
        'bank_name': ['bank', 'bank name'],
    }
    
    def __init__(self, template_file=None, field_keywords=None):
        """
        Initialize form detector
        
        Args:
            template_file: Path to template JSON file
            field_keywords: Custom keyword mapping for fields
        """
        self.template = None
        self.field_keywords = field_keywords or self.DEFAULT_FIELD_KEYWORDS
        
        if template_file:
            self.load_template(template_file)
    
    def load_template(self, template_file):
        """Load form template from JSON file"""
        try:
            with open(template_file, 'r') as f:
                self.template = json.load(f)
            print(f"[OK] Template loaded: {self.template.get('name', 'Unknown')}")
            return True
        except Exception as e:
            print(f"[ERROR] Error loading template: {e}")
            return False
    
    # ─── Image Preprocessing ────────────────────────────────────────────
    
    def preprocess_form(self, image_path):
        """
        Preprocess form image for improved OCR accuracy.
        Designed for scanned/photographed Indian bank challans.
        
        Args:
            image_path: Path to form image
        
        Returns:
            Tuple of (original_color, grayscale, preprocessed_binary)
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Resize if image is too large (speeds up processing)
        h, w = image.shape[:2]
        max_dim = 2000
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # 2. CLAHE for contrast enhancement (good for faded challans)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # 3. Deskew
        enhanced = self._deskew_image(enhanced)
        
        # 4. Adaptive threshold for binary version
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        return image, enhanced, binary
    
    def _deskew_image(self, image):
        """Correct image skew/rotation"""
        coords = np.column_stack(np.where(image > 0))
        if len(coords) < 100:
            return image
        
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        if abs(angle) > 0.5:
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                image, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            return rotated
        return image
    
    # ─── Template-based Field Detection ─────────────────────────────────
    
    def detect_fields_by_template(self, image_path):
        """
        Detect fields using template coordinates.
        
        Args:
            image_path: Path to form image
        
        Returns:
            List of field dicts with 'image', 'bbox', 'field_name', 'field_type'
        """
        if not self.template:
            raise ValueError("No template loaded. Load a template first.")
        
        original, gray, processed = self.preprocess_form(image_path)
        h, w = gray.shape
        
        fields = []
        for field_def in self.template.get('fields', []):
            bbox = field_def.get('bbox', {})
            x = int(bbox.get('x', 0) * w)
            y = int(bbox.get('y', 0) * h)
            width = int(bbox.get('width', 0) * w)
            height = int(bbox.get('height', 0) * h)
            
            # Clamp to image bounds
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            width = min(width, w - x)
            height = min(height, h - y)
            
            if width > 0 and height > 0:
                field_image = gray[y:y+height, x:x+width]
                fields.append({
                    'field_name': field_def.get('id'),
                    'field_type': field_def.get('type'),
                    'bbox': {'x': x, 'y': y, 'width': width, 'height': height},
                    'image': field_image,
                    'validation': field_def.get('validation')
                })
        
        return fields
    
    # ─── Smart OCR-based Field Detection ────────────────────────────────
    
    def detect_fields_from_ocr_results(self, ocr_results, image_shape):
        """
        Given raw OCR text detections, intelligently group them into 
        labeled form fields using keyword matching.
        
        This is the key improvement: instead of blind contour detection,
        we use the OCR text itself to find field labels (e.g. "Account No")
        and then capture the adjacent text as the field value.
        
        Args:
            ocr_results: List of dicts with keys:
                - 'text': recognized text
                - 'bbox': [x, y, w, h] bounding box
                - 'confidence': float 0-1
            image_shape: (height, width) of the source image
        
        Returns:
            List of structured field dicts
        """
        if not ocr_results:
            return []
        
        h, w = image_shape[:2]
        
        # Step 1: Classify each OCR detection as a "label" or "value"
        labels = []   # Things like "Account No:", "Amount", "Date"
        values = []   # Things like "1234567890", "50000", "15/02/2026"
        
        for det in ocr_results:
            text = det['text'].strip()
            if not text:
                continue
            
            matched_field = self._match_keyword(text)
            if matched_field:
                labels.append({
                    **det,
                    'matched_field': matched_field
                })
            else:
                values.append(det)
        
        # Step 2: For each label, find the closest value to its right or below
        fields = []
        used_values = set()
        
        for label in labels:
            lx, ly, lw, lh = label['bbox']
            label_center_y = ly + lh / 2
            label_right_x = lx + lw
            
            best_value = None
            best_score = float('inf')
            best_idx = -1
            
            for idx, val in enumerate(values):
                if idx in used_values:
                    continue
                
                vx, vy, vw, vh = val['bbox']
                val_center_y = vy + vh / 2
                
                # Check: value should be to the RIGHT of the label, 
                # or BELOW the label (common in Indian challans)
                
                # Right of label (same row)
                if vx > label_right_x - 10 and abs(val_center_y - label_center_y) < max(lh, vh) * 1.5:
                    dist = vx - label_right_x
                    score = dist  # prefer closer
                    
                # Below label (next row)
                elif vy > ly + lh * 0.3 and vy < ly + lh * 4:
                    # Must be roughly aligned horizontally
                    horizontal_overlap = min(lx + lw, vx + vw) - max(lx, vx)
                    if horizontal_overlap > -lw * 0.5:
                        dist = vy - (ly + lh)
                        score = dist + 500  # penalize "below" matches slightly
                    else:
                        continue
                else:
                    continue
                
                if score < best_score:
                    best_score = score
                    best_value = val
                    best_idx = idx
            
            if best_value:
                used_values.add(best_idx)
                
                # Determine field type
                field_type = self._infer_field_type(label['matched_field'], best_value['text'])
                
                fields.append({
                    'field_name': label['matched_field'],
                    'field_type': field_type,
                    'extracted_value': best_value['text'],
                    'confidence': best_value['confidence'],
                    'bbox': {
                        'x': best_value['bbox'][0],
                        'y': best_value['bbox'][1],
                        'width': best_value['bbox'][2],
                        'height': best_value['bbox'][3]
                    },
                    'label_text': label['text'],
                    'label_bbox': {
                        'x': label['bbox'][0],
                        'y': label['bbox'][1],
                        'width': label['bbox'][2],
                        'height': label['bbox'][3]
                    },
                    'image': None,  # Will be filled by OCR service if needed
                    'validation': self._get_validation_rule(label['matched_field'])
                })
        
        # Step 3: Also capture any remaining unmatched text as generic fields
        for idx, val in enumerate(values):
            if idx not in used_values and val['confidence'] > 0.3:
                text = val['text'].strip()
                if len(text) >= 2:  # Skip single chars
                    field_type = self._guess_field_type(text)
                    fields.append({
                        'field_name': f'detected_{len(fields)}',
                        'field_type': field_type,
                        'extracted_value': text,
                        'confidence': val['confidence'],
                        'bbox': {
                            'x': val['bbox'][0],
                            'y': val['bbox'][1],
                            'width': val['bbox'][2],
                            'height': val['bbox'][3]
                        },
                        'label_text': None,
                        'label_bbox': None,
                        'image': None,
                        'validation': None
                    })
        
        # Sort by position (top to bottom, left to right)
        fields.sort(key=lambda f: (f['bbox']['y'], f['bbox']['x']))
        
        return fields
    
    def detect_fields_auto(self, image_path):
        """
        Legacy auto-detect method. Now improved with adaptive contour detection.
        Used as a fallback when OCR-based detection is not available.
        """
        original, gray, processed = self.preprocess_form(image_path)
        inverted = cv2.bitwise_not(processed)
        
        contours, _ = cv2.findContours(
            inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        fields = []
        h, w = gray.shape
        
        for i, contour in enumerate(contours):
            x, y, bw, bh = cv2.boundingRect(contour)
            area = bw * bh
            aspect_ratio = bw / bh if bh > 0 else 0
            
            # Better filtering for form fields
            min_area = max(200, (h * w) * 0.001)   # At least 0.1% of image
            max_area = (h * w) * 0.5                # At most 50% of image
            
            if min_area < area < max_area and 0.2 < aspect_ratio < 15:
                field_image = gray[y:y+bh, x:x+bw]
                fields.append({
                    'field_name': f'field_{i}',
                    'field_type': 'numeric',
                    'bbox': {'x': x, 'y': y, 'width': bw, 'height': bh},
                    'image': field_image,
                    'validation': None
                })
        
        fields.sort(key=lambda f: (f['bbox']['y'], f['bbox']['x']))
        return fields
    
    # ─── Digit Segmentation ─────────────────────────────────────────────
    
    def extract_digit_regions(self, field_image, min_width=3, min_height=8):
        """
        Extract individual digit regions from a field image.
        Used when CNN-based digit recognition is needed for handwritten digits.
        
        Args:
            field_image: Grayscale image of numeric field
            min_width: Minimum digit width
            min_height: Minimum digit height
        
        Returns:
            List of dicts with 'image' and 'bbox'
        """
        if field_image is None or field_image.size == 0:
            return []
        
        # Upscale small images
        if field_image.shape[0] < 24:
            scale = max(2.0, 28.0 / field_image.shape[0])
            field_image = cv2.resize(field_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            min_width = int(min_width * scale)
            min_height = int(min_height * scale)
        
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            field_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphological close to connect broken digit parts
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= min_width and h >= min_height:
                boxes.append((x, y, w, h))
        
        boxes.sort(key=lambda b: b[0])
        
        # Merge overlapping boxes
        merged_boxes = self._merge_close_boxes(boxes)
        
        digit_regions = []
        for (x, y, w, h) in merged_boxes:
            pad = 2
            y1 = max(0, y - pad)
            y2 = min(field_image.shape[0], y + h + pad)
            x1 = max(0, x - pad)
            x2 = min(field_image.shape[1], x + w + pad)
            
            digit_img = field_image[y1:y2, x1:x2]
            if digit_img.size > 0:
                digit_regions.append({
                    'image': digit_img,
                    'bbox': {'x': x, 'y': y, 'width': w, 'height': h}
                })
        
        return digit_regions
    
    def _merge_close_boxes(self, boxes, gap_threshold=5):
        """Merge bounding boxes that are horizontally close"""
        if not boxes:
            return []
        
        merged = []
        current = list(boxes[0])
        
        for next_box in boxes[1:]:
            cx, cy, cw, ch = current
            nx, ny, nw, nh = next_box
            
            # Check horizontal proximity
            gap = nx - (cx + cw)
            
            # Check vertical overlap
            intersect_y = min(cy + ch, ny + nh) - max(cy, ny)
            min_h = min(ch, nh)
            vertical_overlap = intersect_y / min_h if min_h > 0 else 0
            
            if gap < gap_threshold and vertical_overlap > 0.3:
                # Merge
                new_x = min(cx, nx)
                new_y = min(cy, ny)
                new_w = max(cx + cw, nx + nw) - new_x
                new_h = max(cy + ch, ny + nh) - new_y
                current = [new_x, new_y, new_w, new_h]
            else:
                merged.append(tuple(current))
                current = list(next_box)
        
        merged.append(tuple(current))
        return merged
    
    # ─── Keyword Matching Helpers ───────────────────────────────────────
    
    def _match_keyword(self, text):
        """
        Check if text matches any field label keyword.
        Returns matched field name or None.
        """
        text_lower = text.lower().strip()
        # Remove common trailing chars like :, -, .
        text_clean = re.sub(r'[:\-.\s]+$', '', text_lower)
        
        for field_name, keywords in self.field_keywords.items():
            for kw in keywords:
                if kw in text_clean or text_clean in kw:
                    return field_name
                # Also check if text starts with keyword
                if text_clean.startswith(kw):
                    return field_name
        
        return None
    
    def _infer_field_type(self, field_name, value_text):
        """Infer field type from the field name and value"""
        if field_name in ('date',):
            return 'date'
        elif field_name in ('name', 'branch', 'bank_name', 'deposit_type'):
            return 'text'
        elif field_name in ('account_number', 'amount', 'cheque_number', 
                          'reference_number', 'mobile', 'pan'):
            return 'numeric'
        elif field_name in ('ifsc',):
            return 'alphanumeric'
        
        # Guess from value content
        return self._guess_field_type(value_text)
    
    def _guess_field_type(self, text):
        """Guess field type from value content"""
        text = text.strip()
        
        # Date pattern
        if re.match(r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}', text):
            return 'date'
        
        # Purely numeric
        if re.match(r'^[\d,.\s]+$', text):
            return 'numeric'
        
        # Alphanumeric (like IFSC codes)
        if re.match(r'^[A-Z]{4}0[A-Z0-9]{6}$', text, re.IGNORECASE):
            return 'alphanumeric'
        
        return 'text'
    
    def _get_validation_rule(self, field_name):
        """Get the appropriate validation rule for a field"""
        rules = {
            'account_number': 'account_number',
            'amount': 'positive_amount',
            'date': 'date',
            'ifsc': 'ifsc',
            'cheque_number': 'numeric',
            'reference_number': 'numeric',
            'mobile': 'numeric',
            'pan': 'alphanumeric',
        }
        return rules.get(field_name)
    
    # ─── Visualization ──────────────────────────────────────────────────
    
    def visualize_fields(self, image_path, fields, output_path=None):
        """
        Visualize detected fields on image
        
        Args:
            image_path: Path to original image
            fields: List of detected fields
            output_path: Path to save visualization
        
        Returns:
            Annotated image
        """
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        for field in fields:
            bbox = field['bbox']
            x, y = bbox['x'], bbox['y']
            w, h = bbox['width'], bbox['height']
            
            # Color by confidence
            confidence = field.get('confidence', 0)
            if confidence > 0.8:
                color = (0, 255, 0)    # Green
            elif confidence > 0.5:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)    # Red
            
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
            label = field.get('field_name', 'unknown')
            value = field.get('extracted_value', '')
            display = f"{label}: {value}"
            
            cv2.putText(
                image, display, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )
        
        if output_path:
            cv2.imwrite(str(output_path), image)
        
        return image
