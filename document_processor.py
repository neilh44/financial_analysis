# document_processor.py

from typing import Optional, Dict, Any
import PyPDF2
import pdfplumber
import logging
from langdetect import detect
import re

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PDFExtractor:
    """Handles PDF text extraction with multiple fallback methods"""
    
    @staticmethod
    def extract_with_pypdf(file_path: str) -> str:
        """Extract text using PyPDF2"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {e}")
        return text.strip()

    @staticmethod
    def extract_with_pdfplumber(file_path: str) -> str:
        """Extract text using pdfplumber"""
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                text = '\n'.join(page.extract_text() for page in pdf.pages)
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
        return text.strip()

    @classmethod
    def extract_text(cls, file_path: str) -> Optional[str]:
        """Extract text from PDF using multiple methods"""
        methods = [
            ('PyPDF2', cls.extract_with_pypdf),
            ('pdfplumber', cls.extract_with_pdfplumber)
        ]
        
        for method_name, extractor in methods:
            try:
                text = extractor(file_path)
                if text:
                    logger.info(f"Successfully extracted text using {method_name}")
                    return text
            except Exception as e:
                logger.error(f"{method_name} extraction error: {e}")
                continue
        
        return None

class TextAnalyzer:
    """Text analysis and preprocessing utilities"""
    
    @staticmethod
    def detect_language(text: str) -> str:
        """Detect the language of the document"""
        try:
            return detect(text)
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return 'unknown'

    @staticmethod
    def extract_number(text: str) -> Optional[float]:
        """Extract number from text, handling various formats"""
        try:
            # Remove thousands separators and convert decimal separators
            cleaned = text.replace(',', '').replace(' ', '')
            return float(cleaned)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def find_closest_number(text: str, keyword: str, window: int = 100) -> Optional[float]:
        """Find the closest number to a keyword in text"""
        pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
        
        # Find keyword position
        pos = text.lower().find(keyword.lower())
        if pos == -1:
            return None
            
        # Get surrounding context
        start = max(0, pos - window)
        end = min(len(text), pos + window)
        context = text[start:end]
        
        # Find all numbers in context
        numbers = [(m.start(), float(m.group())) 
                  for m in re.finditer(pattern, context)]
        
        if not numbers:
            return None
            
        # Find closest number to keyword
        keyword_pos = pos - start
        closest = min(numbers, key=lambda x: abs(x[0] - keyword_pos))
        return closest[1]

    @staticmethod
    def normalize_number(value: Any) -> Optional[float]:
        """Normalize number from various formats"""
        if value is None:
            return None
            
        try:
            if isinstance(value, (int, float)):
                return float(value)
                
            # Handle string values
            value = str(value).strip()
            
            # Handle parentheses for negative values
            if value.startswith('(') and value.endswith(')'):
                value = f"-{value[1:-1]}"
                
            # Remove thousands separators and other formatting
            value = value.replace(',', '').replace(' ', '')
            
            return float(value)
            
        except (ValueError, TypeError):
            return None

class DocumentProcessor:
    """Main class for document processing and text extraction"""
    
    def __init__(self):
        self.pdf_extractor = PDFExtractor()
        self.text_analyzer = TextAnalyzer()

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process document and extract text with metadata"""
        try:
            # Extract text
            text = self.pdf_extractor.extract_text(file_path)
            if not text:
                raise ValueError("No text could be extracted from the document")

            # Detect language
            language = self.text_analyzer.detect_language(text)
            logger.info(f"Detected language: {language}")

            return {
                'text': text,
                'language': language,
                'success': True
            }

        except Exception as e:
            logger.error(f"Document processing error: {str(e)}")
            return {
                'text': None,
                'language': 'unknown',
                'success': False,
                'error': str(e)
            }

# Usage example
if __name__ == "__main__":
    processor = DocumentProcessor()
    result = processor.process_document("example.pdf")
    
    if result['success']:
        print(f"Language detected: {result['language']}")
        print(f"Text length: {len(result['text'])} characters")
    else:
        print(f"Processing failed: {result.get('error')}")