# financial_analyzer.py

from typing import Dict, Any, Optional
from dataclasses import dataclass
import json
import logging
import time
from groq import Groq
from document_processor import DocumentProcessor, TextAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class FinancialMetrics:
    """Data class to hold financial metrics"""
    revenue: Optional[float] = None
    ebit: Optional[float] = None
    ebitda: Optional[float] = None
    net_income: Optional[float] = None
    employees: Optional[int] = None
    depreciation: Optional[float] = None
    amortization: Optional[float] = None
    currency: str = "EUR"
    units: str = "actuals"
    confidence_score: float = 0.0

class FinancialAnalyzer:
    """Financial document analysis using LLM"""
    
    def __init__(self, groq_api_key: str):
        self.groq_client = Groq(api_key=groq_api_key)
        self.document_processor = DocumentProcessor()
        self.validation_rules = {
            'min_revenue': 0,
            'min_confidence': 70,
            'required_fields': ['revenue', 'ebit', 'net_income'],
            'retry_attempts': 3,
            'retry_delay': 1
        }

    def analyze_document(self, file_path: str) -> Dict[str, Any]:
        """Main method to analyze a financial document"""
        try:
            # Process document
            doc_result = self.document_processor.process_document(file_path)
            if not doc_result['success']:
                return {
                    'error': doc_result.get('error', 'Document processing failed'),
                    'success': False
                }

            # Analyze text
            results = self.analyze_with_llm(doc_result['text'], doc_result['language'])
            
            # Add metadata
            results['language'] = doc_result['language']
            results['validations'] = self.validate_financial_data(results)
            results['accuracy'] = self._calculate_accuracy(results)
            
            return {
                'success': True,
                'data': results
            }

        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            return {
                'error': str(e),
                'success': False
            }

    def _clean_json_response(self, response: str) -> str:
        """Clean and validate JSON response"""
        try:
            # Find first { and last }
            start = response.find('{')
            end = response.rfind('}')
            
            if start != -1 and end != -1:
                potential_json = response[start:end + 1]
                # Validate it's parseable
                json.loads(potential_json)
                return potential_json
            
            raise ValueError("No valid JSON object found in response")
        except Exception as e:
            logger.error(f"Error cleaning JSON response: {e}")
            raise

    def analyze_with_llm(self, text: str, language: str) -> Dict[str, Any]:
        """Use LLM for financial analysis with retries"""
        retry_count = 0
        
        while retry_count < self.validation_rules['retry_attempts']:
            try:
                prompt = self._create_analysis_prompt(text, language)
                response = self._make_llm_request(prompt)
                
                # Clean and parse response
                cleaned_response = self._clean_json_response(response)
                result = self._parse_llm_response(cleaned_response)
                
                if self._validate_llm_result(result):
                    return result
                
            except Exception as e:
                logger.error(f"LLM analysis error (attempt {retry_count + 1}): {e}")
            
            retry_count += 1
            if retry_count < self.validation_rules['retry_attempts']:
                time.sleep(self.validation_rules['retry_delay'])
        
        return self._create_empty_result()

    def _create_analysis_prompt(self, text: str, language: str) -> str:
        """Create analysis prompt for LLM"""
        return f"""
You are a multilingual financial expert. Analyze this financial document and return ONLY a valid JSON object.

Find these values in the {language} text:
1. REVENUE (Sales/Turnover)
2. EBIT (Operating Result)
3. EBITDA (EBIT + Depreciation + Amortization)
4. NET INCOME (Final profit/loss)
5. DEPRECIATION
6. AMORTIZATION
7. EMPLOYEE COUNT

FORMAT YOUR RESPONSE EXACTLY LIKE THIS, replacing values with NULL if not found:
{{
    "revenue": 1000000,
    "ebit": -50000,
    "ebitda": -45000,
    "net_income": -55000,
    "depreciation": 5000,
    "amortization": null,
    "employees": 100,
    "currency": "EUR",
    "units": "actuals",
    "confidence_score": 90
}}

CRITICAL RULES:
1. Response MUST be a valid JSON object
2. Respond with ONLY the JSON object, no other text
3. Use exact numbers found in document
4. Keep negative signs (-)
5. Use null for missing values
6. Units must be one of: "millions", "thousands", "actuals"

Input text:
{text}
"""

    def _make_llm_request(self, prompt: str) -> str:
        """Make request to LLM API"""
        completion = self.groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a multilingual financial expert. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        return completion.choices[0].message.content.strip()

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse and normalize LLM response"""
        result = json.loads(response)
        
        # Normalize all numerical values
        for key in ['revenue', 'ebit', 'ebitda', 'net_income', 'depreciation', 'amortization']:
            result[key] = TextAnalyzer.normalize_number(result.get(key))
        
        # Handle employees separately
        if result.get('employees') is not None:
            try:
                result['employees'] = int(TextAnalyzer.normalize_number(result['employees']))
            except (TypeError, ValueError):
                result['employees'] = None
        
        return result

    def _validate_llm_result(self, result: Dict[str, Any]) -> bool:
        """Validate LLM result"""
        try:
            # Check required fields
            required_fields = ['revenue', 'ebit', 'net_income', 'currency', 'units', 'confidence_score']
            if not all(key in result for key in required_fields):
                return False
                
            # Validate currency
            if not isinstance(result.get('currency'), str):
                return False
                
            # Validate units
            if result.get('units', '').lower() not in ['millions', 'thousands', 'actuals']:
                return False
                
            # Validate confidence score
            confidence = result.get('confidence_score')
            if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 100:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating LLM result: {e}")
            return False

    def _create_empty_result(self) -> Dict[str, Any]:
        """Create empty result structure"""
        return FinancialMetrics().__dict__

    def validate_financial_data(self, data: Dict[str, Any]) -> Dict[str, bool]:
        """Validate extracted financial data with null checks"""
        revenue = data.get('revenue')
        return {
            'has_revenue': bool(revenue and revenue > self.validation_rules['min_revenue']),
            'has_required_metrics': all(data.get(key) is not None for key in self.validation_rules['required_fields']),
            'confidence_sufficient': bool(data.get('confidence_score', 0) >= self.validation_rules['min_confidence']),
            'values_consistent': self._check_value_consistency(data),
            'ebitda_consistent': self._check_ebitda_consistency(data)
        }

    def _check_value_consistency(self, data: Dict[str, Any]) -> bool:
            """Check if financial values are logically consistent"""
            try:
                revenue = data.get('revenue')
                ebit = data.get('ebit')
                net_income = data.get('net_income')
                ebitda = data.get('ebitda')
                depreciation = data.get('depreciation')
                amortization = data.get('amortization')
                
                # Only check relationships when both values are present
                if revenue is not None and ebit is not None:
                    # EBIT should not be greater than revenue (in most normal cases)
                    if ebit > revenue:
                        logger.warning("EBIT greater than revenue")
                        return False
                
                if ebit is not None and net_income is not None:
                    # Net income should not be greater than EBIT (in most normal cases)
                    if net_income > ebit:
                        logger.warning("Net income greater than EBIT")
                        return False
                
                if ebitda is not None and ebit is not None:
                    # EBITDA should be greater than or equal to EBIT
                    # (since we add back depreciation and amortization)
                    if ebitda < ebit:
                        logger.warning("EBITDA less than EBIT")
                        return False
                
                # Check if EBITDA calculation is consistent when all components are present
                if all(v is not None for v in [ebit, depreciation, amortization, ebitda]):
                    calculated_ebitda = ebit + depreciation + amortization
                    # Allow for small rounding differences (0.1% tolerance)
                    if abs(calculated_ebitda - ebitda) > abs(ebitda * 0.001):
                        logger.warning(f"EBITDA calculation inconsistent: calculated={calculated_ebitda}, reported={ebitda}")
                        return False
                
                # Check for logical sign consistency
                if revenue is not None and net_income is not None:
                    if revenue > 0 and net_income > revenue:
                        logger.warning("Net income greater than positive revenue")
                        return False
                    if revenue < 0 and net_income > 0:
                        logger.warning("Positive net income with negative revenue")
                        return False
                
                # Check for reasonable depreciation/amortization values
                if revenue is not None and depreciation is not None:
                    if depreciation > revenue:
                        logger.warning("Depreciation greater than revenue")
                        return False
                
                if revenue is not None and amortization is not None:
                    if amortization > revenue:
                        logger.warning("Amortization greater than revenue")
                        return False
                
                return True
                
            except Exception as e:
                logger.error(f"Error in value consistency check: {e}")
                return False

    def _check_ebitda_consistency(self, data: Dict[str, Any]) -> bool:
                """Check EBITDA consistency with components"""
                try:
                    # Get all components needed for EBITDA calculation
                    ebit = data.get('ebit')
                    depreciation = data.get('depreciation')
                    amortization = data.get('amortization')
                    reported_ebitda = data.get('ebitda')
                    
                    # Only check if we have all components and reported EBITDA
                    if all(v is not None for v in [ebit, depreciation, amortization, reported_ebitda]):
                        calculated_ebitda = ebit + depreciation + amortization
                        
                        # Allow for small rounding differences (0.1% tolerance)
                        tolerance = abs(reported_ebitda * 0.001)
                        if abs(calculated_ebitda - reported_ebitda) <= tolerance:
                            return True
                        else:
                            logger.warning(
                                f"EBITDA inconsistency - Calculated: {calculated_ebitda}, "
                                f"Reported: {reported_ebitda}, "
                                f"Difference: {abs(calculated_ebitda - reported_ebitda)}"
                            )
                            return False
                    
                    # If we don't have all components, consider it consistent
                    # (we can't verify what we don't have)
                    return True
                    
                except Exception as e:
                    logger.error(f"Error in EBITDA consistency check: {e}")
                    return False        
                

    def _calculate_accuracy(self, data: Dict[str, Any]) -> float:
        """Calculate overall accuracy score based on validations and confidence"""
        try:
            # Get validation results
            validations = self.validate_financial_data(data)
            
            # Define weights for different components
            weights = {
                'has_revenue': 0.2,
                'has_required_metrics': 0.3,
                'values_consistent': 0.25,
                'ebitda_consistent': 0.15,
                'confidence': 0.1
            }
            
            # Calculate weighted score
            score = 0.0
            
            # Add validation components
            for key, weight in weights.items():
                if key == 'confidence':
                    # Handle confidence score separately
                    confidence = data.get('confidence_score', 0)
                    score += (confidence / 100.0) * weight
                else:
                    # Handle boolean validation results
                    score += float(validations.get(key, False)) * weight
            
            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, score)) * 100
            
        except Exception as e:
            logger.error(f"Error calculating accuracy: {e}")
            return 0.0

        