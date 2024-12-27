from typing import Dict, Any, Optional
import logging
from document_processor import EnhancedDocumentProcessor, PDFExtractor, TextAnalyzer
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Structure for multilingual analysis results"""
    metrics: Dict[str, Optional[float]]
    calculated_metrics: Dict[str, Optional[float]]
    analysis: Dict[str, Any]
    currency_info: Dict[str, str]
    language: str
    success: bool
    error: Optional[str] = None
    raw_metrics: Optional[Dict[str, Any]] = None

class MultilingualFinancialAnalyzer:
    """Handles multilingual financial document analysis with Gemini integration"""
    
    def __init__(self, gemini_api_key: str):
        """Initialize analyzer with Gemini API key"""
        if not gemini_api_key:
            raise ValueError("Gemini API key is required")
            
        self.processor = EnhancedDocumentProcessor(gemini_api_key)
        self.pdf_extractor = PDFExtractor()
        self.text_analyzer = TextAnalyzer()

    async def analyze_document(self, file_path: str, year: Optional[str] = None) -> Dict[str, Any]:
        """Extract and analyze financial data from PDF"""
        try:
            # Extract text from PDF
            text = self.pdf_extractor.extract_text(file_path)
            if not text:
                logger.error("Failed to extract text from PDF")
                return self._create_error_response('Could not extract text from PDF file')

            # Detect language
            language = self.text_analyzer.detect_language(text)
            logger.info(f"Detected language: {language}")
            
            if language == 'unknown':
                logger.warning("Could not detect document language, defaulting to English")
                language = 'en'

            # Process and analyze the document
            result = await self.processor.process_and_analyze(text, year)
            
            if not result.success:
                logger.error(f"Analysis failed: {result.error}")
                return self._create_error_response(result.error or 'Analysis failed')

            # Format the response
            response = {
                'success': True,
                'data': {
                    'revenue': result.metrics.get('revenue'),
                    'ebit': result.metrics.get('ebit'),
                    'ebitda': result.metrics.get('ebitda'),
                    'net_income': result.metrics.get('net_income'),
                    'depreciation': result.metrics.get('depreciation'),
                    'amortization': result.metrics.get('amortization'),
                    'employees': result.metrics.get('employees'),
                    'profit_margin': result.calculated_metrics.get('net_profit_margin'),
                    'ebitda_margin': result.calculated_metrics.get('ebitda_margin'),
                    'operating_margin': result.calculated_metrics.get('operating_margin'),
                    'financial_health_score': result.analysis.get('financial_health_score'),
                    'currency': result.currency_info.get('code'),
                    'units': result.currency_info.get('units'),
                    'confidence_score': result.currency_info.get('confidence'),
                    'detected_language': language
                },
                'raw_metrics': result.metrics  # Include raw metrics for debugging
            }

            # Validate response data
            self._validate_response(response)
            
            return response

        except TypeError as e:
            logger.error(f"Type error during processing: {str(e)}")
            return self._create_error_response('Invalid data type encountered during processing')
            
        except ValueError as e:
            logger.error(f"Value error during processing: {str(e)}")
            return self._create_error_response('Invalid value encountered during processing')
            
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return self._create_error_response(f'Failed to process document: {str(e)}')

    def _validate_response(self, response: Dict[str, Any]) -> None:
        """Validate response data and log warnings for missing metrics"""
        if not response.get('success'):
            return

        data = response.get('data', {})
        core_metrics = ['revenue', 'net_income', 'ebitda']
        for metric in core_metrics:
            if data.get(metric) is None:
                logger.warning(f"Core metric '{metric}' is missing from analysis")

        if not data.get('currency'):
            logger.warning("Currency information is missing")

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            'success': False,
            'error': error_message,
            'data': None
        }

    def _format_number(self, value: Any) -> Optional[float]:
        """Format numeric values with proper handling"""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            logger.warning(f"Could not convert value to float: {value}")
            return None