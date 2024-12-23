# multilingual_analyzer.py

from typing import Dict, Any, Optional
import logging
from financial_analyzer import FinancialAnalyzer
from document_processor import PDFExtractor, TextAnalyzer

logger = logging.getLogger(__name__)

class MultilingualFinancialAnalyzer:
    """Handles multilingual financial document analysis"""
    
    def __init__(self, groq_api_key: str):
        self.analyzer = FinancialAnalyzer(groq_api_key)
        self.pdf_extractor = PDFExtractor()
        self.text_analyzer = TextAnalyzer()

    def extract_from_pdf(self, file_path: str) -> Dict[str, Any]:
        """Extract and analyze financial data from PDF"""
        try:
            # First validate that we can read the PDF
            text = self.pdf_extractor.extract_text(file_path)
            if not text:
                logger.error("Failed to extract text from PDF")
                return {'error': 'Could not extract text from PDF file'}

            # Detect language
            language = self.text_analyzer.detect_language(text)
            if language == 'unknown':
                logger.warning("Could not detect document language")
                language = 'en'  # Default to English

            # Analyze the document
            result = self.analyzer.analyze_document(file_path)
            
            # Handle analysis failure
            if not result.get('success'):
                error_msg = result.get('error', 'Unknown analysis error')
                logger.error(f"Analysis failed: {error_msg}")
                return {'error': error_msg}

            # Add language information
            result['data']['detected_language'] = language
            
            return result

        except TypeError as e:
            logger.error(f"Type error during PDF processing: {str(e)}")
            return {'error': 'Invalid data type encountered during processing'}
            
        except ValueError as e:
            logger.error(f"Value error during PDF processing: {str(e)}")
            return {'error': 'Invalid value encountered during processing'}
            
        except Exception as e:
            logger.error(f"Unexpected error processing PDF: {str(e)}")
            return {'error': f'Failed to process PDF file: {str(e)}'}