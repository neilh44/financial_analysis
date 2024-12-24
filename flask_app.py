# flask_app.py
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
analyzer = MultilingualFinancialAnalyzer(GROQ_API_KEY)
app.dir_path = os.path.dirname(os.path.realpath(__file__))
app.template_folder = os.path.join(app.dir_path, 'templates')
app.static_folder = os.path.join(app.dir_path, 'static')

def is_valid_pdf(file_stream):
    """Validate PDF file before processing"""
    try:
        # Read the first few bytes to check PDF signature
        pdf_signature = file_stream.read(4)
        file_stream.seek(0)  # Reset file pointer
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
            
        if not file.filename.endswith('.pdf'):
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Only PDF files are supported'}), 400

        # Log the file info
        logger.info(f"Processing file: {file.filename}, Size: {len(file.read())} bytes")
        file.seek(0)  # Reset file pointer

        try:
            # Save and process file
            temp_path = os.path.join(app.dir_path, 'temp', secure_filename(file.filename))
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            file.save(temp_path)
            
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
    
if __name__ == '__main__':
    app.run(debug=True)