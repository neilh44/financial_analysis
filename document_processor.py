from typing import Optional, Dict, Any
import logging
import json
import httpx
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    metrics: Dict[str, Optional[float]]
    calculated_metrics: Dict[str, Optional[float]]
    analysis: Dict[str, Any]
    currency_info: Dict[str, str]
    success: bool
    error: Optional[str] = None

class DocumentProcessor:
    def __init__(self, groq_api_key: Optional[str] = None):
        from pdf_converter import PDFConverter
        from text_analyzer import TextAnalyzer
        
        self.pdf_converter = PDFConverter()
        self.text_analyzer = TextAnalyzer()
        self.groq_api_key = groq_api_key
        if not self.groq_api_key:
            raise ValueError("Groq API key is required")

    def _convert_to_markdown(self, text: str) -> str:
        """Convert plain text to markdown format"""
        lines = text.split('\n')
        markdown_lines = []
        current_table = []
        in_table = False

        for line in lines:
            line = line.strip()
            if not line:
                if in_table:
                    # End current table
                    if current_table:
                        markdown_lines.extend(self._format_table(current_table))
                        current_table = []
                    in_table = False
                markdown_lines.append('')
                continue

            # Detect table rows (lines with multiple numbers or financial data)
            if sum(1 for c in line if c.isdigit() or c in '.,¥$€£') > 3:
                in_table = True
                current_table.append(line)
                continue

            # Headers (typically short lines with year or category)
            if len(line) < 50 and any(char.isdigit() for char in line):
                if in_table:
                    if current_table:
                        markdown_lines.extend(self._format_table(current_table))
                        current_table = []
                    in_table = False
                markdown_lines.append(f'### {line}')
            else:
                if in_table:
                    if current_table:
                        markdown_lines.extend(self._format_table(current_table))
                        current_table = []
                    in_table = False
                markdown_lines.append(line)

        # Handle any remaining table
        if current_table:
            markdown_lines.extend(self._format_table(current_table))

        return '\n'.join(markdown_lines)

    def _format_table(self, table_lines: list) -> list:
        """Format detected table lines into markdown table"""
        if not table_lines:
            return []

        # Create header
        header = ['Item'] + ['Value'] * (len(table_lines[0].split()) - 1)
        markdown_table = [
            '| ' + ' | '.join(header) + ' |',
            '| ' + ' | '.join(['---'] * len(header)) + ' |'
        ]

        # Add data rows
        for line in table_lines:
            cells = line.split()
            markdown_table.append('| ' + ' | '.join(cells) + ' |')

        return markdown_table

    async def process_and_analyze(self, file_path: str, year: Optional[str] = None) -> AnalysisResult:
        """Process and analyze a PDF document"""
        try:
            # Extract text from PDF
            text = self.pdf_converter._pdf_to_text(file_path)
            if not text:
                raise ValueError("Could not extract text from the document")

            # Convert to markdown
            markdown_text = self._convert_to_markdown(text)
            logger.info("Converted text to markdown format")

            # Detect language
            language = self.text_analyzer.detect_language(text)
            logger.info(f"Detected language: {language}")

            # Analyze with Groq
            analysis_result = await self._analyze_with_groq(markdown_text, language, year)
            if not analysis_result['success']:
                return AnalysisResult(
                    metrics={}, 
                    calculated_metrics={}, 
                    analysis={}, 
                    currency_info={}, 
                    success=False, 
                    error=analysis_result['error']
                )

            enhanced_result = self._enhance_analysis(analysis_result['data'])
            return AnalysisResult(
                metrics=enhanced_result['metrics'],
                calculated_metrics=enhanced_result['calculated_metrics'],
                analysis=enhanced_result['analysis'],
                currency_info=enhanced_result['currency_info'],
                success=True
            )
        except Exception as e:
            logger.error(f"Document processing error: {str(e)}")
            return AnalysisResult(
                metrics={}, 
                calculated_metrics={}, 
                analysis={}, 
                currency_info={}, 
                success=False, 
                error=str(e)
            )

    async def _analyze_with_groq(self, markdown_text: str, language: str, year: Optional[str] = None) -> Dict[str, Any]:
        """Analyze markdown document using Groq API"""
        try:
            # Create chunked markdown if too long
            chunked_markdown = self._chunk_markdown(markdown_text)
            prompt = self._create_analysis_prompt(chunked_markdown, language, year)
            
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "llama3-8b-8192",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a financial expert analyzing markdown-formatted financial documents. Always respond in valid JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 2048
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=data, headers=headers)
                response.raise_for_status()
                
                result = response.json()
                if not result or 'choices' not in result or not result['choices']:
                    return {'success': False, 'error': 'No response from Groq'}
                
                response_text = result['choices'][0]['message']['content'].strip()
                
                try:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = response_text[json_start:json_end]
                        data = json.loads(json_str)
                        
                        if not all(key in data for key in ['metrics', 'calculated_metrics', 'analysis']):
                            return {
                                'success': False,
                                'error': 'Invalid response structure: missing required keys'
                            }
                        
                        return {'success': True, 'data': data}
                    else:
                        return {'success': False, 'error': 'No valid JSON object found in response'}

                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error: {e}")
                    return {'success': False, 'error': f'Invalid JSON response from Groq: {str(e)}'}

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from Groq API: {e}")
            return {'success': False, 'error': f'Groq API error: {str(e)}'}
        except Exception as e:
            logger.error(f"Groq analysis error: {e}")
            return {'success': False, 'error': str(e)}

    def _chunk_markdown(self, markdown_text: str, max_length: int = 5000) -> str:
        """Chunk markdown text to avoid token limits while preserving structure"""
        if len(markdown_text) <= max_length:
            return markdown_text
        
        # Split text into lines to work with markdown structure
        lines = markdown_text.split('\n')
        
        # Track headers and tables
        headers = []
        tables = []
        current_section = []
        in_table = False
        
        # First pass: identify structure
        for i, line in enumerate(lines):
            if line.startswith('#'):  # Header
                headers.append((i, line))
            elif line.startswith('|'):  # Table
                if not in_table:
                    in_table = True
                    tables.append([i])  # Start new table
                if i < len(lines) - 1 and not lines[i + 1].startswith('|'):
                    in_table = False
                    tables[-1].append(i)  # End table
        
        # Calculate sections to keep
        start_length = int(max_length * 0.6)
        end_length = int(max_length * 0.4)
        
        # Get important parts from beginning
        start_text = []
        current_length = 0
        
        for line in lines:
            line_length = len(line) + 1  # +1 for newline
            if current_length + line_length > start_length:
                break
            start_text.append(line)
            current_length += line_length
        
        # Get important parts from end
        end_text = []
        current_length = 0
        
        for line in reversed(lines):
            line_length = len(line) + 1
            if current_length + line_length > end_length:
                break
            end_text.insert(0, line)
            current_length += line_length
        
        # Ensure we're not breaking in the middle of a table or section
        while start_text and start_text[-1].startswith('|'):
            start_text.pop()
        while end_text and end_text[0].startswith('|'):
            if not any(line.startswith('|') for line in end_text[1:]):
                break
            end_text.pop(0)
        
        # Add a clear separator
        separator = "\n\n... [Content truncated for length] ...\n\n"
        
        # Combine sections
        chunked_text = (
            '\n'.join(start_text) +
            separator +
            '\n'.join(end_text)
        )
        
        # Ensure we haven't exceeded max length
        if len(chunked_text) > max_length:
            # If still too long, do a simple truncation
            available_length = max_length - len(separator)
            start_portion = int(available_length * 0.6)
            end_portion = available_length - start_portion
            chunked_text = (
                chunked_text[:start_portion] +
                separator +
                chunked_text[-end_portion:]
            )
        
        return chunked_text

    def _create_analysis_prompt(self, text: str, language: str, year: Optional[str]) -> str:
        """Create analysis prompt for Groq"""
        return f"""
Analyze this financial document and extract key metrics. The document is in {language}.
Year of analysis: {year if year else 'current'}

Extract and return the following information in JSON format:

1. Key metrics:
   - Revenue (Liikevaihto/Omsättning/Sales/Turnover)
   - EBIT (Operating Profit/Liikevoitto/Rörelseresultat)
   - Net Income (Tilikauden tulos/Årets resultat/Profit after tax)
   - Depreciation (Poistot/Avskrivningar)
   - Amortization
   - Number of Employees (Henkilöstö/Personal)

2. Calculate:
   - EBITDA = EBIT + Depreciation + Amortization
   - EBITDA margin = (EBITDA / Revenue) * 100
   - Net profit margin = (Net Income / Revenue) * 100
   - Operating margin = (EBIT / Revenue) * 100

Document text:
{text}

Return a JSON object in this exact format (replace null with null value and number with actual numbers):

{{
    "metrics": {{
        "revenue": "number or null",
        "ebit": "number or null",
        "ebitda": "number or null",
        "net_income": "number or null",
        "depreciation": "number or null",
        "amortization": "number or null",
        "employees": "number or null"
    }},
    "calculated_metrics": {{
        "ebitda_margin": "number or null",
        "net_profit_margin": "number or null",
        "operating_margin": "number or null"
    }},
    "analysis": {{
        "currency": "detected_currency_code",
        "financial_health_score": "number",
        "confidence_score": "number",
        "company_type": "company_type_string",
        "data_completeness": "number"
    }},
    "warnings": []
}}"""

    def _enhance_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance analysis results with additional calculations"""
        metrics = data.get('metrics', {})
        calculated = data.get('calculated_metrics', {})
        analysis = data.get('analysis', {})

        # Calculate EBITDA if not provided
        if metrics.get('ebit') and metrics.get('depreciation') and metrics.get('amortization'):
            metrics['ebitda'] = metrics['ebit'] + (metrics['depreciation'] or 0) + (metrics['amortization'] or 0)

        # Calculate margins if revenue exists
        if metrics.get('revenue'):
            if metrics.get('net_income'):
                calculated['net_profit_margin'] = (metrics['net_income'] / metrics['revenue']) * 100
            if metrics.get('ebitda'):
                calculated['ebitda_margin'] = (metrics['ebitda'] / metrics['revenue']) * 100
            if metrics.get('ebit'):
                calculated['operating_margin'] = (metrics['ebit'] / metrics['revenue']) * 100

        return {
            'metrics': metrics,
            'calculated_metrics': calculated,
            'analysis': analysis,
            'currency_info': {
                'code': analysis.get('currency', 'Unknown'),
                'units': analysis.get('units', 'actuals'),
                'confidence': analysis.get('confidence_score', 0)
            }
        }