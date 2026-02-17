"""
Hybrid OCR Service for Indian Bank Challans (v2 - Region-Based)
================================================================
Improved pipeline using REGION-BASED extraction:
  1. Detect rectangular field boxes on the form (contour detection)
  2. Within each box, find the label (printed text) and value area
  3. Use EasyOCR to read each region separately
  4. Match labels to field names using keyword matching
  5. Post-process and validate extracted values

This approach is much more accurate than full-page OCR because:
  - It avoids cross-contamination between fields
  - It correctly associates labels with their values
  - It can apply different strategies for printed vs handwritten text
"""

import os
import re
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

from models.digit_model import DigitRecognitionModel
from models.form_detector import FormDetector


class OCRService:
    """Hybrid OCR service using region-based extraction for bank challans"""
    
    def __init__(self, model_path, template_dir=None, config=None):
        self.config = config or {}
        self.template_dir = template_dir
        
        # Engine flags
        self.easyocr_available = False
        self.tesseract_available = False
        self.cnn_available = False
        
        # Initialize engines
        self._init_easyocr()
        self._init_tesseract()
        self._init_cnn(model_path)
        
        # Initialize Form Detector
        field_keywords = getattr(config, 'FIELD_KEYWORDS', None) if config else None
        self.form_detector = FormDetector(field_keywords=field_keywords)
        
        self._print_status()
    
    # ─── Engine Initialization ──────────────────────────────────────────
    
    def _init_easyocr(self):
        try:
            import easyocr
            languages = getattr(self.config, 'EASYOCR_LANGUAGES', ['en'])
            use_gpu = getattr(self.config, 'EASYOCR_GPU', False)
            
            print("[INIT] Loading EasyOCR model...")
            self.easyocr_reader = easyocr.Reader(languages, gpu=use_gpu, verbose=False)
            self.easyocr_available = True
            print("[OK] EasyOCR initialized successfully")
        except ImportError:
            print("[WARN] EasyOCR not installed. pip install easyocr")
            self.easyocr_reader = None
        except Exception as e:
            print(f"[WARN] EasyOCR init failed: {e}")
            self.easyocr_reader = None
    
    def _init_tesseract(self):
        try:
            import pytesseract
            tesseract_cmd = getattr(self.config, 'TESSERACT_CMD', None)
            if tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            pytesseract.get_tesseract_version()
            self.pytesseract = pytesseract
            self.tesseract_available = True
            self.tesseract_lang = getattr(self.config, 'TESSERACT_LANG', 'eng')
            print("[OK] Tesseract OCR initialized successfully")
        except ImportError:
            print("[WARN] pytesseract not installed")
            self.pytesseract = None
        except Exception as e:
            print(f"[WARN] Tesseract not available: {e}")
            self.pytesseract = None
    
    def _init_cnn(self, model_path):
        try:
            if model_path and os.path.exists(model_path):
                self.digit_model = DigitRecognitionModel(model_path)
                self.cnn_available = True
                print("[OK] CNN digit model loaded successfully")
            else:
                print(f"[WARN] CNN model not found at: {model_path}")
                self.digit_model = DigitRecognitionModel()
        except Exception as e:
            print(f"[WARN] CNN model init failed: {e}")
            self.digit_model = DigitRecognitionModel()
    
    def _print_status(self):
        print("\n" + "=" * 50)
        print("  OCR ENGINE STATUS")
        print("=" * 50)
        print(f"  EasyOCR  (primary)  : {'[OK] Ready' if self.easyocr_available else '[--] Unavailable'}")
        print(f"  Tesseract (fallback) : {'[OK] Ready' if self.tesseract_available else '[--] Unavailable'}")
        print(f"  CNN Model (digits)   : {'[OK] Ready' if self.cnn_available else '[--] Unavailable'}")
        if not self.easyocr_available and not self.tesseract_available:
            print("\n  [!!] WARNING: No OCR engine available!")
        print("=" * 50 + "\n")
    
    # ═══════════════════════════════════════════════════════════════════
    #  MAIN PROCESSING PIPELINE
    # ═══════════════════════════════════════════════════════════════════
    
    def process_form(self, image_path, form_type=None, template_file=None):
        """
        Process a banking form image.
        
        Pipeline:
        1. Preprocess image (denoise, enhance contrast, deskew)
        2. Detect rectangular field regions (boxes on the form)
        3. For each box: find label text + value text
        4. Match labels to known field names
        5. Post-process values (clean, format)
        """
        result = {
            'success': False,
            'form_type': form_type or 'generic',
            'fields': [],
            'raw_text': '',
            'overall_confidence': 0.0,
            'ocr_engine_used': 'none',
            'errors': []
        }
        
        try:
            # Template-based processing
            if template_file and os.path.exists(str(template_file)):
                return self._process_with_template(image_path, template_file, result)
            
            # Region-based processing (main approach)
            return self._process_region_based(image_path, result)
            
        except Exception as e:
            result['errors'].append(f"Processing error: {str(e)}")
            import traceback
            traceback.print_exc()
            return result
    
    def _process_region_based(self, image_path, result):
        """
        Region-based form processing:
        1. Read image
        2. Detect rectangular field boxes
        3. For each box, read label + value
        4. Smart field matching
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            result['errors'].append(f"Could not read image: {image_path}")
            return result
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        print(f"\n[PROCESSING] Image: {os.path.basename(str(image_path))} ({w}x{h})")
        
        # Step 1: Run full-page OCR to get ALL text with positions
        all_detections = self._run_ocr_on_image(image)
        result['raw_text'] = ' '.join([d['text'] for d in all_detections])
        
        print(f"[OCR] Detected {len(all_detections)} text regions")
        for det in all_detections:
            print(f"  >> '{det['text']}' (conf={det['confidence']:.2f}) at {det['bbox']}")
        
        # Step 2: Detect rectangular field boxes
        field_boxes = self._detect_field_boxes(gray)
        print(f"[BOXES] Found {len(field_boxes)} field regions")
        
        # Step 3: For each box, find which OCR detections fall inside it
        fields = []
        
        if field_boxes:
            fields = self._extract_fields_from_boxes(field_boxes, all_detections, gray)
        else:
            # Fallback: no boxes found, use keyword-based matching on flat OCR
            print("[FALLBACK] No boxes detected, using keyword matching")
            fields = self.form_detector.detect_fields_from_ocr_results(
                all_detections, image.shape
            )
        
        # Step 4: Post-process fields
        processed_fields = []
        total_confidence = 0.0
        valid_count = 0
        
        for field in fields:
            field = self._post_process_field(field)
            processed_fields.append(field)
            conf = field.get('confidence', 0)
            if conf > 0:
                total_confidence += conf
                valid_count += 1
        
        result['fields'] = processed_fields
        result['overall_confidence'] = total_confidence / valid_count if valid_count > 0 else 0
        result['success'] = True
        result['ocr_engine_used'] = 'easyocr' if self.easyocr_available else (
            'tesseract' if self.tesseract_available else 'cnn')
        
        # Print results summary
        print(f"\n[RESULTS] {len(processed_fields)} fields extracted:")
        for f in processed_fields:
            print(f"  {f['field_name']}: '{f.get('extracted_value', '')}' ({f.get('confidence', 0):.0%})")
        
        return result
    
    # ═══════════════════════════════════════════════════════════════════
    #  STEP 1: RECTANGULAR BOX DETECTION
    # ═══════════════════════════════════════════════════════════════════
    
    def _detect_field_boxes(self, gray):
        """
        Detect rectangular field boxes on the form.
        Bank challans typically have clear bordered rectangles for each field.
        
        Returns: List of (x, y, w, h) tuples sorted top-to-bottom
        """
        h, w = gray.shape
        
        # Adaptive threshold to get binary image
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5
        )
        
        # Morphological operations to enhance horizontal and vertical lines
        # Detect horizontal lines
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 8, 30), 1))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
        
        # Detect vertical lines
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 8, 30)))
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
        
        # Combine lines
        lines_combined = cv2.add(h_lines, v_lines)
        
        # Dilate to close gaps
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        lines_combined = cv2.dilate(lines_combined, dilate_kernel, iterations=2)
        
        # Find contours of the combined lines
        contours, hierarchy = cv2.findContours(
            lines_combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter rectangular contours
        boxes = []
        min_area = (h * w) * 0.005   # At least 0.5% of image
        max_area = (h * w) * 0.6     # At most 60% of image
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            
            # Approximate contour to polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # Should be roughly rectangular (4 sides)
            if len(approx) >= 4:
                x, y, bw, bh = cv2.boundingRect(contour)
                aspect = bw / bh if bh > 0 else 0
                
                # Field boxes are usually wider than tall
                if 0.5 < aspect < 15 and bw > 30 and bh > 20:
                    boxes.append((x, y, bw, bh))
        
        # Remove duplicate/overlapping boxes
        boxes = self._remove_overlapping_boxes(boxes)
        
        # Sort top-to-bottom
        boxes.sort(key=lambda b: b[1])
        
        return boxes
    
    def _remove_overlapping_boxes(self, boxes, iou_threshold=0.5):
        """Remove overlapping boxes, keeping the larger ones"""
        if not boxes:
            return []
        
        # Sort by area (largest first)
        boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
        
        keep = []
        for box in boxes:
            is_overlapping = False
            for kept in keep:
                if self._iou(box, kept) > iou_threshold:
                    is_overlapping = True
                    break
            if not is_overlapping:
                keep.append(box)
        
        return keep
    
    def _iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        xi = max(x1, x2)
        yi = max(y1, y2)
        xf = min(x1 + w1, x2 + w2)
        yf = min(y1 + h1, y2 + h2)
        
        if xi >= xf or yi >= yf:
            return 0.0
        
        intersection = (xf - xi) * (yf - yi)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    # ═══════════════════════════════════════════════════════════════════
    #  STEP 2: EXTRACT FIELDS FROM DETECTED BOXES
    # ═══════════════════════════════════════════════════════════════════
    
    def _extract_fields_from_boxes(self, boxes, all_detections, gray):
        """
        For each detected box, find which OCR text falls inside it.
        Separate label text (field name) from value text (user's input).
        """
        fields = []
        h, w = gray.shape
        
        for box_idx, (bx, by, bw, bh) in enumerate(boxes):
            # Find all OCR detections that overlap with this box
            texts_in_box = []
            
            for det in all_detections:
                dx, dy, dw, dh = det['bbox']
                det_cx = dx + dw / 2
                det_cy = dy + dh / 2
                
                # Check if center of detection is inside the box (with some margin)
                margin = 10
                if (bx - margin <= det_cx <= bx + bw + margin and 
                    by - margin <= det_cy <= by + bh + margin):
                    texts_in_box.append(det)
            
            if not texts_in_box:
                # No OCR text found in this box - try reading it directly
                box_img = gray[by:by+bh, bx:bx+bw]
                direct_text = self._read_region(box_img)
                if direct_text:
                    texts_in_box = [{
                        'text': direct_text,
                        'bbox': [bx, by, bw, bh],
                        'confidence': 0.5
                    }]
            
            if not texts_in_box:
                continue
            
            # Separate labels from values
            label_text, value_text, confidence = self._separate_label_value(
                texts_in_box, by, bh
            )
            
            # Match label to known field name
            field_name = self.form_detector._match_keyword(label_text) if label_text else None
            
            if not field_name:
                field_name = f'field_{box_idx}'
            
            # Determine field type
            field_type = self._infer_type(field_name, value_text)
            
            fields.append({
                'field_name': field_name,
                'field_type': field_type,
                'extracted_value': value_text,
                'confidence': confidence,
                'label_text': label_text,
                'bbox': {'x': bx, 'y': by, 'width': bw, 'height': bh},
                'validation': self._get_validation_rule(field_name),
                'individual_digits': []
            })
        
        return fields
    
    def _separate_label_value(self, detections, box_y, box_h):
        """
        Separate label text from value text within a box.
        
        Strategies:
        1. Single detection with "Label: Value" format -> split on separator
        2. Single detection with keyword at start -> extract value after keyword
        3. Multiple detections -> separate by position + keyword analysis
        """
        if len(detections) == 1:
            text = detections[0]['text'].strip()
            conf = detections[0]['confidence']
            
            # Strategy 1: Split on common separators (: - )
            # Try colon first
            for separator_pattern in [r':\s*', r';\s*', r'\-\s+']:
                parts = re.split(separator_pattern, text, maxsplit=1)
                if len(parts) == 2 and len(parts[0]) > 2:
                    label_part = parts[0].strip()
                    value_part = parts[1].strip()
                    
                    # Verify the label part actually contains a keyword
                    if self.form_detector._match_keyword(label_part):
                        return label_part, value_part, conf
            
            # Strategy 2: Find keyword at the beginning and extract the rest
            matched = self.form_detector._match_keyword(text)
            if matched:
                # Try to find where the keyword ends and value begins
                # Look for transition from letters to digits
                m = re.search(r'([A-Za-z\s:.]+?)\s*[:\-]?\s*(\d[\d/\-.,\s]*)', text)
                if m:
                    return m.group(1).strip(), m.group(2).strip(), conf
                
                # No numeric value found - it's just a label
                return text, '', conf
            
            # No keyword found - it's probably a value
            return '', text, conf
        
        # Multiple detections: sort by vertical position
        sorted_dets = sorted(detections, key=lambda d: d['bbox'][1])
        
        # Find which detections are labels vs values
        label_parts = []
        value_parts = []
        
        for det in sorted_dets:
            text = det['text'].strip()
            
            # Check if this text is a field label keyword
            is_label_keyword = self.form_detector._match_keyword(text) is not None
            
            # Check if text looks like a value: 
            # - Pure digits/punctuation
            # - OR mostly digits (>60%) allowing for OCR misreads like 'g' for '8'
            digit_count = sum(1 for c in text if c.isdigit() or c in '.,/-')
            total_chars = len(text.replace(' ', ''))
            is_mostly_digits = total_chars > 0 and (digit_count / total_chars) > 0.6
            looks_like_value = bool(re.match(r'^[\d,./\-\s]+$', text)) or is_mostly_digits
            
            if is_label_keyword:
                # If it's "Label: Value" combined, split it
                for sep_pattern in [r':\s*', r';\s*']:
                    parts = re.split(sep_pattern, text, maxsplit=1)
                    if len(parts) == 2 and parts[1].strip():
                        label_parts.append(parts[0].strip())
                        value_parts.append(parts[1].strip())
                        break
                else:
                    label_parts.append(text)
            elif looks_like_value:
                # Clearly a value (digits, dates, numbers)
                value_parts.append(text)
            elif not is_label_keyword and len(label_parts) > 0:
                # Already have a label, so this is likely a value
                value_parts.append(text)
            else:
                label_parts.append(text)
        
        label = ' '.join(label_parts)
        value = ' '.join(value_parts)
        
        # Average confidence
        avg_conf = sum(d['confidence'] for d in sorted_dets) / len(sorted_dets)
        
        return label, value, avg_conf
    
    # ═══════════════════════════════════════════════════════════════════
    #  OCR ENGINE CALLS
    # ═══════════════════════════════════════════════════════════════════
    
    def _run_ocr_on_image(self, image):
        """
        Run OCR on an image (BGR numpy array or file path).
        Returns list of {'text', 'bbox': [x,y,w,h], 'confidence'}
        """
        if isinstance(image, str) or isinstance(image, Path):
            image = cv2.imread(str(image))
        
        if image is None:
            return []
        
        # Try EasyOCR first
        if self.easyocr_available:
            return self._run_easyocr(image)
        
        # Fallback to Tesseract
        if self.tesseract_available:
            return self._run_tesseract(image)
        
        return []
    
    def _run_easyocr(self, image):
        """Run EasyOCR on a BGR image array"""
        try:
            # Convert to RGB for EasyOCR
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            results = self.easyocr_reader.readtext(
                rgb,
                detail=1,
                paragraph=False,
                min_size=10,
                text_threshold=0.5,
                low_text=0.3,
                contrast_ths=0.2,
                width_ths=0.5,
            )
            
            detections = []
            for (bbox_points, text, confidence) in results:
                text = text.strip()
                if not text:
                    continue
                
                pts = np.array(bbox_points)
                x = int(pts[:, 0].min())
                y = int(pts[:, 1].min())
                w = int(pts[:, 0].max() - x)
                h = int(pts[:, 1].max() - y)
                
                detections.append({
                    'text': text,
                    'bbox': [x, y, w, h],
                    'confidence': float(confidence),
                    'engine': 'easyocr'
                })
            
            return detections
        except Exception as e:
            print(f"[EasyOCR ERROR] {e}")
            return []
    
    def _run_tesseract(self, image):
        """Run Tesseract on a BGR image array"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Enhance
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            data = self.pytesseract.image_to_data(
                enhanced, lang=self.tesseract_lang,
                output_type=self.pytesseract.Output.DICT,
                config='--psm 6'
            )
            
            detections = []
            n_boxes = len(data['level'])
            
            for i in range(n_boxes):
                text = data['text'][i].strip()
                conf = int(data['conf'][i])
                
                if text and conf > 20:
                    detections.append({
                        'text': text,
                        'bbox': [data['left'][i], data['top'][i], 
                                data['width'][i], data['height'][i]],
                        'confidence': conf / 100.0,
                        'engine': 'tesseract'
                    })
            
            # Merge nearby words
            return self._merge_nearby_words(detections)
        except Exception as e:
            print(f"[Tesseract ERROR] {e}")
            return []
    
    def _read_region(self, region_img):
        """Read text from a small image region (grayscale)"""
        if region_img is None or region_img.size == 0:
            return ''
        
        try:
            # Convert to BGR for OCR
            if len(region_img.shape) == 2:
                bgr = cv2.cvtColor(region_img, cv2.COLOR_GRAY2BGR)
            else:
                bgr = region_img
            
            if self.easyocr_available:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                results = self.easyocr_reader.readtext(rgb, detail=0, paragraph=True)
                return ' '.join(results).strip() if results else ''
            
            if self.tesseract_available:
                text = self.pytesseract.image_to_string(region_img, lang=self.tesseract_lang)
                return text.strip()
        except:
            pass
        
        return ''
    
    def _merge_nearby_words(self, detections, gap_ratio=1.5):
        """Merge Tesseract word-level detections into line-level"""
        if not detections:
            return []
        
        sorted_dets = sorted(detections, key=lambda d: (d['bbox'][1], d['bbox'][0]))
        merged = []
        current = dict(sorted_dets[0])
        
        for det in sorted_dets[1:]:
            cx, cy, cw, ch = current['bbox']
            nx, ny, nw, nh = det['bbox']
            
            c_cy = cy + ch / 2
            n_cy = ny + nh / 2
            avg_h = (ch + nh) / 2
            same_line = abs(c_cy - n_cy) < avg_h * 0.6
            gap = nx - (cx + cw)
            close = gap < avg_h * gap_ratio
            
            if same_line and close:
                new_x = min(cx, nx)
                new_y = min(cy, ny)
                new_w = max(cx + cw, nx + nw) - new_x
                new_h = max(cy + ch, ny + nh) - new_y
                current['text'] = current['text'] + ' ' + det['text']
                current['bbox'] = [new_x, new_y, new_w, new_h]
                current['confidence'] = (current['confidence'] + det['confidence']) / 2
            else:
                merged.append(current)
                current = dict(det)
        
        merged.append(current)
        return merged
    
    # ═══════════════════════════════════════════════════════════════════
    #  POST-PROCESSING
    # ═══════════════════════════════════════════════════════════════════
    
    def _post_process_field(self, field):
        """Clean and validate extracted field values with OCR error correction"""
        value = field.get('extracted_value', '') or ''
        field_name = field.get('field_name', '')
        field_type = field.get('field_type', 'text')
        
        if not value:
            return field
        
        # Step 1: Fix common OCR character misreadings
        if field_type in ('numeric', 'date') or field_name in (
            'account_number', 'amount', 'cheque_number', 'reference_number', 'mobile', 'date'
        ):
            value = self._fix_ocr_chars(value, context='numeric')
        
        # Step 2: Type-specific processing
        if field_type == 'numeric' or field_name in ('account_number', 'amount', 
                                                       'cheque_number', 'reference_number',
                                                       'mobile'):
            # Keep only digits, dots, and commas
            cleaned = re.sub(r'[^\d.,]', '', value)
            field['extracted_value'] = cleaned
        elif field_type == 'date' or field_name == 'date':
            field['extracted_value'] = self._format_date(value)
        elif field_name == 'ifsc':
            value = self._fix_ocr_chars(value, context='alphanumeric')
            cleaned = re.sub(r'[^A-Z0-9]', '', value.upper())
            field['extracted_value'] = cleaned
        else:
            field['extracted_value'] = value
        
        return field
    
    def _fix_ocr_chars(self, text, context='numeric'):
        """
        Fix common OCR character misreadings.
        
        EasyOCR frequently confuses:
          o/O -> 0, l/I/| -> 1, g -> 8, S/s -> 5, 
          z/Z -> 2, b -> 6, B -> 8, q -> 9, 
          [ or ( -> /, ] or ) -> /
        """
        if context == 'numeric':
            # In numeric context, letters should be digits
            char_map = {
                'o': '0', 'O': '0', 'D': '0',
                'l': '1', 'I': '1', '|': '1', 'i': '1',
                'z': '2', 'Z': '2',
                'E': '3',
                'A': '4', 'h': '4',
                's': '5', 'S': '5',
                'b': '6', 'G': '6',
                'T': '7',
                'g': '8', 'B': '8',
                'q': '9',
                '[': '/', ']': '/',
                '(': '/', ')': '/',
                '{': '/', '}': '/',
                '\\': '/',
            }
        elif context == 'alphanumeric':
            # In alphanumeric, be more conservative
            char_map = {
                '|': '1',
                '[': '/',
                ']': '/',
            }
        else:
            return text
        
        result = []
        for ch in text:
            result.append(char_map.get(ch, ch))
        
        return ''.join(result)
    
    def _format_date(self, raw_text):
        """Try to format extracted text as DD/MM/YYYY"""
        if not raw_text:
            return raw_text
        
        text = raw_text.strip()
        
        # Already formatted
        if re.match(r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}', text):
            return text
        
        # Extract digits
        digits = re.sub(r'[^\d]', '', text)
        
        if len(digits) >= 8:
            return f"{digits[0:2]}/{digits[2:4]}/{digits[4:8]}"
        elif len(digits) >= 6:
            return f"{digits[0:2]}/{digits[2:4]}/{digits[4:6]}"
        
        return text
    
    def _infer_type(self, field_name, value_text):
        """Infer field type from name and value"""
        type_map = {
            'date': 'date',
            'name': 'text',
            'branch': 'text',
            'bank_name': 'text',
            'account_number': 'numeric',
            'amount': 'numeric',
            'cheque_number': 'numeric',
            'reference_number': 'numeric',
            'mobile': 'numeric',
            'ifsc': 'alphanumeric',
            'pan': 'alphanumeric',
        }
        
        if field_name in type_map:
            return type_map[field_name]
        
        # Guess from value
        if value_text:
            if re.match(r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}', value_text):
                return 'date'
            if re.match(r'^[\d,.\s]+$', value_text):
                return 'numeric'
        
        return 'text'
    
    def _get_validation_rule(self, field_name):
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
    
    # ═══════════════════════════════════════════════════════════════════
    #  TEMPLATE-BASED PROCESSING
    # ═══════════════════════════════════════════════════════════════════
    
    def _process_with_template(self, image_path, template_file, result):
        """Process form using template-defined field locations"""
        self.form_detector.load_template(str(template_file))
        fields = self.form_detector.detect_fields_by_template(image_path)
        
        image = cv2.imread(str(image_path))
        
        processed_fields = []
        total_confidence = 0.0
        valid_count = 0
        
        for field in fields:
            field_image = field.get('image')
            if field_image is not None and field_image.size > 0:
                # Read text from this field region
                if len(field_image.shape) == 2:
                    bgr = cv2.cvtColor(field_image, cv2.COLOR_GRAY2BGR)
                else:
                    bgr = field_image
                
                detections = self._run_ocr_on_image(bgr)
                text = ' '.join([d['text'] for d in detections])
                confidence = np.mean([d['confidence'] for d in detections]) if detections else 0
                
                field['extracted_value'] = text
                field['confidence'] = confidence
                field['individual_digits'] = []
            
            processed_fields.append(field)
            conf = field.get('confidence', 0)
            if conf > 0:
                total_confidence += conf
                valid_count += 1
        
        result['fields'] = processed_fields
        result['overall_confidence'] = total_confidence / valid_count if valid_count > 0 else 0
        result['success'] = True
        result['ocr_engine_used'] = 'template+ocr'
        
        return result
    
    # ═══════════════════════════════════════════════════════════════════
    #  UTILITY / API
    # ═══════════════════════════════════════════════════════════════════
    
    def get_engine_status(self):
        return {
            'easyocr': {'available': self.easyocr_available, 'role': 'primary'},
            'tesseract': {'available': self.tesseract_available, 'role': 'fallback'},
            'cnn': {'available': self.cnn_available, 'role': 'handwritten_digits'}
        }
    
    def process_batch(self, image_paths, form_type=None):
        return [self.process_form(path, form_type) for path in image_paths]
