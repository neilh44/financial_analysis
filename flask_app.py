from flask import Flask, render_template, request, jsonify
from document_processor import DocumentProcessor
from text_analyzer import TextAnalyzer
import os
from dotenv import load_dotenv
import tempfile
import logging
from werkzeug.utils import secure_filename
import asyncio
from functools import wraps
from datetime import datetime


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Validate API key
if not GROQ_API_KEY:
    raise ValueError("GROQ API key not found. Please set GROQ_API_KEY in your .env file.")

def async_route(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapped

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit
document_processor = DocumentProcessor(GROQ_API_KEY)
text_analyzer = TextAnalyzer()

app.dir_path = os.path.dirname(os.path.realpath(__file__))
app.template_folder = os.path.join(app.dir_path, 'templates')
app.static_folder = os.path.join(app.dir_path, 'static')
app.temp_folder = os.path.join(app.dir_path, 'temp')

# Ensure temp folder exists
os.makedirs(app.temp_folder, exist_ok=True)

def is_valid_pdf(file_stream):
    """Validate PDF file before processing"""
    try:
        pdf_signature = file_stream.read(4)
        file_stream.seek(0)
        return pdf_signature == b'%PDF'
    except Exception as e:
        logger.error(f"PDF validation error: {str(e)}")
        return False

def cleanup_temp_files():
    """Clean up old temporary files"""
    try:
        current_time = datetime.now()
        for filename in os.listdir(app.temp_folder):
            file_path = os.path.join(app.temp_folder, filename)
            file_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
            if (current_time - file_modified).total_seconds() > 3600:  # 1 hour
                os.unlink(file_path)
    except Exception as e:
        logger.error(f"Error cleaning up temp files: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
@async_route
async def analyze():
    """Process and analyze uploaded PDF document"""
    try:
        # Validate file upload
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not is_valid_pdf(file):
            return jsonify({'error': 'Invalid PDF file'}), 400

        # Get optional parameters
        year = request.form.get('year')
        
        # Create a unique temporary file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_filename = f"temp_{timestamp}_{secure_filename(file.filename)}"
        temp_path = os.path.join(app.temp_folder, temp_filename)
        
        try:
            # Save file
            file.save(temp_path)
            logger.info(f"File saved to: {temp_path}")

            # Process and analyze document
            logger.info("Starting document analysis...")
            result = await document_processor.process_and_analyze(temp_path, year)
            
            if not result.success:
                return jsonify({
                    'error': result.error or 'Analysis failed'
                }), 500

            # Updated response structure to match frontend expectations
            formatted_response = {
                'metrics': {
                    'revenue': result.metrics.get('revenue'),
                    'ebit': result.metrics.get('ebit'),
                    'ebitda': result.metrics.get('ebitda'),
                    'netIncome': result.metrics.get('net_income'),  # Note the camelCase
                    'depreciation': result.metrics.get('depreciation'),
                    'amortization': result.metrics.get('amortization'),
                    'employees': result.metrics.get('employees')
                },
                'calculated_metrics': {  # Additional metrics if needed
                    'ebitdaMargin': result.calculated_metrics.get('ebitda_margin'),
                    'netProfitMargin': result.calculated_metrics.get('net_profit_margin'),
                    'operatingMargin': result.calculated_metrics.get('operating_margin')
                },
                'metadata': {
                    'currency': result.currency_info.get('code', 'USD'),
                    'confidenceScore': result.analysis.get('confidence_score', 0),
                    'units': result.currency_info.get('unit', 'actuals'),
                    'language': getattr(result, 'language', 'unknown')
                }
            }
            
            return jsonify(formatted_response)
            
        finally:
            # Cleanup temporary file
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception as e:
                logger.error(f"Error removing temporary file: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500
    
@app.route('/health')
def health_check():
    """Simple health check endpoint"""
    cleanup_temp_files()  # Clean up old temp files during health checks
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 10MB'}), 413

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'error': 'An internal server error occurred',
        'details': str(error) if app.debug else None
    }), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    return jsonify({
        'error': 'An unexpected error occurred',
        'details': str(e) if app.debug else None
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)