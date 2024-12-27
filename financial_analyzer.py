import google.generativeai as genai
import logging
from typing import Dict, Any, Optional
import json
from dataclasses import dataclass
from typing import List

@dataclass
class CurrencyInfo:
    code: str
    unit: str
    year: str

logger = logging.getLogger(__name__)

class CurrencyHandler:
    """Handle currency detection and validation"""
    
    KNOWN_CURRENCIES = {
        'EUR': {'symbols': ['€', 'EUR', 'euro', 'euros']},
        'USD': {'symbols': ['$', 'USD', 'dollar', 'dollars']},
        'GBP': {'symbols': ['£', 'GBP', 'pound', 'pounds']},
        'CNY': {'symbols': ['¥', 'CNY', 'yuan']},
        'JPY': {'symbols': ['¥', 'JPY', 'yen']},
        'RUB': {'symbols': ['₽', 'RUB', 'ruble', 'rubles']},
        'SEK': {'symbols': ['kr', 'SEK', 'krona', 'kronor']},
    }
    
    KNOWN_UNITS = ['thousands', 'millions', 'billions', 'M', 'B', 'K', 'Thousand', 'Million', 'Billion']
    
    @classmethod
    def detect_currency(cls, text: str, year: str) -> Optional[CurrencyInfo]:
        """Detect currency and unit from text for a specific year"""
        # Special case for Latvia as per SOP section 9
        if "Latvia" in text and int(year) >= 2014:
            return CurrencyInfo(code='EUR', unit='actuals', year=year)
            
        # Look for currency symbols and codes
        detected_currency = None
        for curr_code, curr_info in cls.KNOWN_CURRENCIES.items():
            if any(symbol.lower() in text.lower() for symbol in curr_info['symbols']):
                detected_currency = curr_code
                break
                
        # Look for units
        detected_unit = 'actuals'  # default
        for unit in cls.KNOWN_UNITS:
            if unit.lower() in text.lower():
                # Special case for Colombian companies as per SOP section 8
                if unit == 'M' and 'Colombian' in text:
                    detected_unit = 'thousands'
                else:
                    detected_unit = unit.lower()
                break
                
        if detected_currency:
            return CurrencyInfo(
                code=detected_currency,
                unit=detected_unit,
                year=year
            )
        return None

    @staticmethod
    def validate_currency(currency_info: CurrencyInfo) -> bool:
        """Validate currency information as per Hard Stops"""
        if not currency_info:
            return False
        return all([
            currency_info.code is not None,
            currency_info.code in CurrencyHandler.KNOWN_CURRENCIES,
            currency_info.unit is not None,
            currency_info.year is not None
        ])

class FinancialAnalyzer:
    """Financial document analysis using only Gemini with enhanced extraction"""
    
    def __init__(self, gemini_api_key: str):
        if not gemini_api_key:
            raise ValueError("Gemini API key is required")
            
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.currency_handler = CurrencyHandler()

    async def analyze_document(self, text: str, year: str = None) -> Dict[str, Any]:
        """Main method to analyze a financial document"""
        try:
            # Extract and analyze in a single step
            results = await self._extract_and_analyze(text, year)
            if not results['success']:
                return results

            # Validate results
            validation = self.validate_financial_data(results['data'])
            results['validation'] = validation
            results['accuracy_score'] = self._calculate_accuracy(results['data'])

            return results

        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            return {
                'error': str(e),
                'success': False
            }

    async def _extract_and_analyze(self, text: str, year: str = None) -> Dict[str, Any]:
        """Extract metrics and perform analysis using Gemini"""
        try:
            # Detect currency information
            currency_info = self.currency_handler.detect_currency(text, year or "2024")
            
            prompt = f"""
You are a financial expert analyzing a financial document. Extract and calculate key metrics according to these guidelines:

1. Extract these metrics (use null if not found):
   - Revenue (Liikevaihto/Omsättning/Sales/Turnover)
   - EBIT (Operating Profit/Liikevoitto/Rörelseresultat)
   - Net Income (Tilikauden tulos/Årets resultat/Profit after tax)
   - Depreciation (Poistot/Avskrivningar)
   - Amortization
   - Number of Employees (Henkilöstö/Personal)

2. Special cases:
   - For banks/financial institutions: Revenue = Net Interest Income + Fees + Trading Income
   - For insurance companies: Revenue = Net Earned Premium + Reinsurance Revenue
   - EBIT calculation if not directly provided: Profit Before Tax + Interest Expense - Interest Income
   - EBITDA = EBIT + Depreciation + Amortization

3. Calculate additional metrics:
   - EBITDA margin
   - Net profit margin
   - Operating margin
   - Financial health score (0-100)

Currency Information:
- Detected Currency: {currency_info.code if currency_info else 'Unknown'}
- Detected Unit: {currency_info.unit if currency_info else 'Unknown'}
- Year: {currency_info.year if currency_info else 'Unknown'}

Input Text:
{text}

Return a JSON object in this exact format (replace null with null value and number with actual numbers):

{
    "metrics": {
        "revenue": "number or null",
        "ebit": "number or null",
        "ebitda": "number or null",
        "net_income": "number or null",
        "depreciation": "number or null",
        "amortization": "number or null",
        "employees": "number or null"
    },
    "calculated_metrics": {
        "ebitda_margin": "number or null",
        "net_profit_margin": "number or null",
        "operating_margin": "number or null"
    },
    "analysis": {
        "currency": "detected_currency_code",
        "financial_health_score": "number",
        "confidence_score": "number",
        "company_type": "company_type_string",
        "data_completeness": "number"
    },
    "warnings": []
}"""

            safety_settings = [
                {
                    "category": "HARM_CATEGORY_DANGEROUS",
                    "threshold": "BLOCK_NONE",
                }
            ]
            
            response = self.model.generate_content(
                prompt,
                safety_settings=safety_settings,
                generation_config={
                    "temperature": 0.1,
                    "top_p": 1,
                    "top_k": 1,
                    "max_output_tokens": 2048,
                }
            )

            if not response or not response.text:
                return {
                    'success': False,
                    'error': 'No response from Gemini'
                }

            try:
                # Clean the response text to ensure it contains only the JSON part
                response_text = response.text.strip()
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    data = json.loads(json_str)
                    
                    # Validate the response structure
                    required_keys = ['metrics', 'calculated_metrics', 'analysis']
                    if not all(key in data for key in required_keys):
                        return {
                            'success': False,
                            'error': 'Invalid response structure: missing required keys'
                        }
                    
                    return {
                        'success': True,
                        'data': data
                    }
                else:
                    return {
                        'success': False,
                        'error': 'No valid JSON object found in response'
                    }
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {str(e)}")
                return {
                    'success': False,
                    'error': f'Invalid JSON response from Gemini: {str(e)}'
                }
            except Exception as e:
                logger.error(f"Unexpected error processing response: {str(e)}")
                return {
                    'success': False,
                    'error': f'Error processing response: {str(e)}'
                }

        except Exception as e:
            logger.error(f"Extraction error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def validate_financial_data(self, data: Dict[str, Any]) -> Dict[str, bool]:
        """Validate extracted financial data"""
        metrics = data.get('metrics', {})
        calculated = data.get('calculated_metrics', {})
        analysis = data.get('analysis', {})
        
        return {
            'has_core_metrics': all(metrics.get(key) is not None for key in ['revenue', 'net_income']),
            'has_operational_metrics': all(metrics.get(key) is not None for key in ['ebit', 'ebitda']),
            'has_calculated_ratios': all(calculated.get(key) is not None for key in ['ebitda_margin', 'net_profit_margin']),
            'has_valid_scores': all(0 <= analysis.get(key, -1) <= 100 for key in ['financial_health_score', 'confidence_score'])
        }

    def _calculate_accuracy(self, data: Dict[str, Any]) -> float:
        """Calculate accuracy score based on data completeness and validity"""
        try:
            metrics = data.get('metrics', {})
            calculated = data.get('calculated_metrics', {})
            
            # Weight different components
            weights = {
                'core_metrics': 0.4,  # revenue, net_income
                'operational_metrics': 0.3,  # ebit, ebitda
                'calculated_ratios': 0.2,  # margins
                'metadata': 0.1  # currency, scores
            }
            
            scores = {
                'core_metrics': sum(1 for k in ['revenue', 'net_income'] if metrics.get(k) is not None) / 2,
                'operational_metrics': sum(1 for k in ['ebit', 'ebitda'] if metrics.get(k) is not None) / 2,
                'calculated_ratios': sum(1 for k in calculated if calculated.get(k) is not None) / len(calculated),
                'metadata': 1.0 if data.get('analysis', {}).get('currency') else 0.0
            }
            
            accuracy = sum(weights[k] * scores[k] for k in weights)
            return round(accuracy * 100, 2)
            
        except Exception as e:
            logger.error(f"Error calculating accuracy: {e}")
            return 0.0