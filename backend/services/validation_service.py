"""
Validation Service - Banking data validation
Validates extracted data according to banking rules
"""

import re
from datetime import datetime

class ValidationService:
    """Service for validating banking form data"""
    
    def __init__(self, config=None):
        """
        Initialize validation service
        
        Args:
            config: Configuration object with validation rules
        """
        self.config = config or {}
        self.account_min_length = self.config.get('ACCOUNT_NUMBER_MIN_LENGTH', 9)
        self.account_max_length = self.config.get('ACCOUNT_NUMBER_MAX_LENGTH', 18)
        self.max_amount = self.config.get('MAX_AMOUNT', 10000000)
    
    def validate_field(self, field_name, value, field_type, validation_rule=None):
        """
        Validate a single field
        
        Args:
            field_name: Name of field
            value: Extracted value
            field_type: Type of field (numeric, date, text)
            validation_rule: Specific validation rule
        
        Returns:
            Validation result dictionary
        """
        result = {
            'is_valid': True,
            'message': '',
            'corrected_value': value
        }
        
        if not value or value == '':
            result['is_valid'] = False
            result['message'] = 'Field is empty'
            return result
        
        # Apply type-specific validation
        if field_type == 'numeric':
            if validation_rule == 'account_number':
                return self.validate_account_number(value)
            elif validation_rule == 'amount' or validation_rule == 'positive_amount':
                return self.validate_amount(value)
            elif validation_rule == 'check_digit':
                return self.validate_check_digit(value)
            else:
                return self.validate_numeric(value)
        
        elif field_type == 'date':
            return self.validate_date(value)
        
        elif field_type == 'text':
            return self.validate_text(value)
        
        return result
    
    def validate_account_number(self, account_number):
        """Validate bank account number"""
        result = {
            'is_valid': True,
            'message': '',
            'corrected_value': account_number
        }
        
        # Remove any spaces or special characters
        cleaned = re.sub(r'[^0-9]', '', str(account_number))
        result['corrected_value'] = cleaned
        
        # Check length
        if len(cleaned) < self.account_min_length:
            result['is_valid'] = False
            result['message'] = f'Account number too short (min {self.account_min_length} digits)'
        elif len(cleaned) > self.account_max_length:
            result['is_valid'] = False
            result['message'] = f'Account number too long (max {self.account_max_length} digits)'
        
        # Check if all digits
        if not cleaned.isdigit():
            result['is_valid'] = False
            result['message'] = 'Account number must contain only digits'
        
        return result
    
    def validate_amount(self, amount):
        """Validate monetary amount"""
        result = {
            'is_valid': True,
            'message': '',
            'corrected_value': amount
        }
        
        try:
            # Convert to float
            amount_value = float(str(amount).replace(',', ''))
            
            # Check if positive
            if amount_value <= 0:
                result['is_valid'] = False
                result['message'] = 'Amount must be positive'
            
            # Check maximum limit
            elif amount_value > self.max_amount:
                result['is_valid'] = False
                result['message'] = f'Amount exceeds maximum limit of ₹{self.max_amount:,.2f}'
            
            # Format amount
            result['corrected_value'] = f'{amount_value:.2f}'
        
        except ValueError:
            result['is_valid'] = False
            result['message'] = 'Invalid amount format'
        
        return result
    
    def validate_date(self, date_str):
        """Validate date field"""
        result = {
            'is_valid': True,
            'message': '',
            'corrected_value': date_str
        }
        
        # Try different date formats
        formats = ['%d/%m/%Y', '%d-%m-%Y', '%d/%m/%y', '%d-%m-%y']
        
        parsed_date = None
        for fmt in formats:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                break
            except ValueError:
                continue
        
        if parsed_date is None:
            result['is_valid'] = False
            result['message'] = 'Invalid date format (expected DD/MM/YYYY)'
            return result
        
        # Check if date is not in future
        if parsed_date > datetime.now():
            result['is_valid'] = False
            result['message'] = 'Date cannot be in the future'
        
        # Check if date is not too old (e.g., more than 1 year ago)
        days_old = (datetime.now() - parsed_date).days
        if days_old > 365:
            result['is_valid'] = False
            result['message'] = 'Date is more than 1 year old'
        
        # Format date consistently
        result['corrected_value'] = parsed_date.strftime('%d/%m/%Y')
        
        return result
    
    def validate_numeric(self, value):
        """Validate generic numeric field"""
        result = {
            'is_valid': True,
            'message': '',
            'corrected_value': value
        }
        
        # Check if numeric
        cleaned = re.sub(r'[^0-9.]', '', str(value))
        
        if not cleaned:
            result['is_valid'] = False
            result['message'] = 'Field must be numeric'
        
        result['corrected_value'] = cleaned
        return result
    
    def validate_text(self, value):
        """Validate text field"""
        result = {
            'is_valid': True,
            'message': '',
            'corrected_value': value
        }
        
        # Basic text validation (non-empty, reasonable length)
        if len(str(value)) < 2:
            result['is_valid'] = False
            result['message'] = 'Text too short'
        elif len(str(value)) > 200:
            result['is_valid'] = False
            result['message'] = 'Text too long'
        
        return result
    
    def validate_check_digit(self, account_number):
        """
        Validate account number using check digit algorithm
        
        This is a simplified version. Real implementation would depend
        on the specific bank's check digit algorithm.
        
        Args:
            account_number: Account number to validate
        
        Returns:
            Validation result
        """
        result = {
            'is_valid': True,
            'message': '',
            'corrected_value': account_number
        }
        
        # Remove non-digits
        cleaned = re.sub(r'[^0-9]', '', str(account_number))
        
        if len(cleaned) < 10:
            result['is_valid'] = False
            result['message'] = 'Account number too short for check digit validation'
            return result
        
        # Simple modulo 11 check digit validation (example)
        # Real banks use different algorithms
        try:
            digits = [int(d) for d in cleaned[:-1]]
            check_digit = int(cleaned[-1])
            
            # Calculate check digit
            weights = list(range(2, len(digits) + 2))
            weighted_sum = sum(d * w for d, w in zip(reversed(digits), weights))
            calculated_check = (11 - (weighted_sum % 11)) % 11
            
            if calculated_check != check_digit:
                result['is_valid'] = False
                result['message'] = 'Check digit validation failed'
        
        except (ValueError, IndexError):
            result['is_valid'] = False
            result['message'] = 'Error in check digit validation'
        
        return result
    
    def validate_ifsc_code(self, ifsc):
        """
        Validate IFSC code format
        
        Args:
            ifsc: IFSC code to validate
        
        Returns:
            Validation result
        """
        result = {
            'is_valid': True,
            'message': '',
            'corrected_value': ifsc.upper() if ifsc else ''
        }
        
        # IFSC format: 4 letters + 7 alphanumeric (first char is always 0)
        pattern = r'^[A-Za-z]{4}0[A-Za-z0-9]{6}$'
        
        if not re.match(pattern, str(ifsc)):
            result['is_valid'] = False
            result['message'] = 'Invalid IFSC code format (expected: ABCD0123456)'
        
        return result
    
    def validate_form(self, form_data):
        """
        Validate entire form
        
        Args:
            form_data: Dictionary of form fields
        
        Returns:
            Validation results for all fields
        """
        results = {}
        
        for field_name, field_info in form_data.items():
            value = field_info.get('value', '')
            field_type = field_info.get('type', 'text')
            validation_rule = field_info.get('validation')
            
            results[field_name] = self.validate_field(
                field_name, value, field_type, validation_rule
            )
        
        return results
    
    def get_confidence_flag(self, confidence):
        """
        Get confidence flag based on threshold
        
        Args:
            confidence: Confidence score (0-1)
        
        Returns:
            Flag color and message
        """
        conf_high = self.config.get('CONFIDENCE_THRESHOLD_HIGH', 0.90)
        conf_low = self.config.get('CONFIDENCE_THRESHOLD_LOW', 0.70)
        
        if confidence >= conf_high:
            return {
                'flag': 'green',
                'status': 'auto_approve',
                'message': 'High confidence - Auto-approved'
            }
        elif confidence >= conf_low:
            return {
                'flag': 'yellow',
                'status': 'review_recommended',
                'message': 'Medium confidence - Review recommended'
            }
        else:
            return {
                'flag': 'red',
                'status': 'manual_review',
                'message': 'Low confidence - Manual review required'
            }
