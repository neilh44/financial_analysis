from flask import Flask, render_template, request, jsonify, current_app
from multilingual_analyzer import MultilingualFinancialAnalyzer
import os
from dotenv import load_dotenv
import tempfile
import logging
from werkzeug.utils import secure_filename
import traceback

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

if not GROQ_API_KEY:
    raise ValueError("GROQ API key not found. Please set GROQ_API_KEY in your .env file.")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit
analyzer = MultilingualFinancialAnalyzer(GROQ_API_KEY)
app.dir_path = os.path.dirname(os.path.realpath(__file__))
app.template_folder = os.path.join(app.dir_path, 'templates')
app.static_folder = os.path.join(app.dir_path, 'static')
app.temp_folder = os.path.join(app.dir_path, 'temp')

def is_valid_pdf(file_stream):
    """Validate PDF file before processing"""
    try:
        pdf_signature = file_stream.read(4)
        file_stream.seek(0)
        return pdf_signature == b'%PDF'
    except Exception as e:
        logger.error(f"PDF validation error: {str(e)}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name

            try:
                results = analyzer.extract_from_pdf(temp_path)
                if results.get('success') and results.get('data'):
                    data = results['data']
                    formatted_response = {
                        'revenue': data['revenue'],
                        'ebit': data['ebit'], 
                        'ebitda': data['ebitda'],
                        'netIncome': data['net_income'],
                        'depreciation': data['depreciation'],
                        'amortization': data['amortization'],
                        'employees': data['employees'],
                        'accuracy': data['accuracy'],
                        'currency': data['currency'],
                        'detectedLanguage': data['detected_language'],
                        'validations': data['validations']
                    }
                    return jsonify(formatted_response)
                return jsonify({'error': 'Analysis failed'}), 500
            finally:
                os.unlink(temp_path)
                
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500
    
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 10MB'}), 413
    
if __name__ == '__main__':
    app.run(debug=True)