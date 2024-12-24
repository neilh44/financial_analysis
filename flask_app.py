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
    logger.info("Received analyze request")
    
    try:
        if 'file' not in request.files:
            logger.error("No file in request")
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        if file.filename == '':
            logger.error("Empty filename")
            return jsonify({'error': 'No file selected'}), 400
            
        if not file.filename.lower().endswith('.pdf'):
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Only PDF files are supported'}), 400

        if not is_valid_pdf(file):
            logger.error("Invalid PDF format")
            return jsonify({'error': 'Invalid PDF format'}), 400

        # Create temp directory if it doesn't exist
        os.makedirs(app.temp_folder, exist_ok=True)

        # Use tempfile for secure temporary file creation
        with tempfile.NamedTemporaryFile(delete=False, dir=app.temp_folder, suffix='.pdf') as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name

            try:
                logger.info(f"Processing file: {file.filename}")
                results = analyzer.extract_from_pdf(temp_path)
                logger.info(f"Analysis results: {results}")
                return jsonify(results)
                
            except Exception as e:
                logger.error(f"Processing error: {str(e)}", exc_info=True)
                return jsonify({'error': 'Error processing document'}), 500
                
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 10MB'}), 413
    
if __name__ == '__main__':
    app.run(debug=True)