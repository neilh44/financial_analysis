import logging
import fitz  # PyMuPDF
import os
from typing import Optional

logger = logging.getLogger(__name__)

class PDFConverter:
    """PDF conversion utility using PyMuPDF"""
    
    def __init__(self):
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required dependencies are available"""
        try:
            import fitz
            logger.info("PyMuPDF (fitz) is available")
        except ImportError:
            logger.error("PyMuPDF is not installed. Please install it using: pip install PyMuPDF")
            raise ImportError("PyMuPDF is required but not installed")
    
    def _pdf_to_text(self, pdf_path: str) -> Optional[str]:
        """Extract text from PDF using PyMuPDF"""
        try:
            # Open PDF document
            doc = fitz.open(pdf_path)
            
            # Extract text from all pages
            text = []
            for page in doc:
                text.append(page.get_text())
                
            # Close the document
            doc.close()
            
            return "\n\n".join(text)
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return None
            
    def convert_pdf_to_text(self, file_path: str) -> Optional[str]:
        """Public method to convert PDF to text"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"PDF file not found: {file_path}")
                
            return self._pdf_to_text(file_path)
                
        except Exception as e:
            logger.error(f"Error converting PDF to text: {e}")
            return None

    @staticmethod
    def is_pdf_ready() -> bool:
        """Check if PDF conversion is ready"""
        try:
            import fitz
            return True
        except ImportError:
            return False