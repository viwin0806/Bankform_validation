import json, urllib.request

data = json.dumps({
    'filepath': 'uploads/20260216_105238_deposit_slip.jpg',
    'form_type': 'deposit',
    'use_template': False
}).encode()

req = urllib.request.Request(
    'http://localhost:5050/api/process',
    data=data,
    headers={'Content-Type': 'application/json'}
)

resp = urllib.request.urlopen(req)
result = json.loads(resp.read())

print('=== EXTRACTION RESULTS ===')
print(f'Engine: {result["ocr_engine_used"]}')
print(f'Overall Confidence: {result["overall_confidence"]:.0%}')
print(f'Raw OCR text: {result["raw_text"]}')
print()

for field in result.get('fields', []):
    name = field['field_name']
    label = field.get('label_text', '')
    value = field.get('extracted_value', '') or field.get('corrected_value', '')
    conf = field.get('confidence', 0)
    msg = field.get('validation_message', '')
    print(f'  {name:20s} = "{value}" (conf={conf:.0%}) label="{label}"')
    if msg:
        print(f'                       validation: {msg}')
