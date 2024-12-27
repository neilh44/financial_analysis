# text_analyzer.py

import logging
from typing import Optional, Any
from langdetect import detect
import re

logger = logging.getLogger(__name__)

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
    def normalize_number(value: Any) -> Optional[float]:
        """Normalize number from various formats"""
        if value is None:
            return None
        try:
            if isinstance(value, (int, float)):
                return float(value)

            if not isinstance(value, str):
                value = str(value)

            value = value.replace('€', '').replace('$', '').replace('£', '').strip()
            if value.startswith('(') and value.endswith(')'):
                value = f"-{value[1:-1]}"
            value = value.replace(' ', '').replace(',', '.')
            if '%' in value:
                return float(value.replace('%', '')) / 100

            multipliers = {'K': 1000, 'M': 1e6, 'B': 1e9}
            for suffix, multiplier in multipliers.items():
                if value.upper().endswith(suffix):
                    return float(value[:-1]) * multiplier

            return float(value)
        except (ValueError, TypeError) as e:
            logger.debug(f"Number normalization failed for value '{value}': {str(e)}")
            return None

    @staticmethod
    def extract_number(text: str) -> Optional[float]:
        """Extract first number from text"""
        try:
            numbers = re.findall(r'[-+]?\d*[.,]?\d+', text)
            return TextAnalyzer.normalize_number(numbers[0]) if numbers else None
        except Exception as e:
            logger.error(f"Number extraction error: {e}")
            return None