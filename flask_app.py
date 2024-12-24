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
    try:
        if 'file' not in request.files:
            logger.warning("No file part in request")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.warning("No selected file")
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.pdf'):
            logger.warning(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Only PDF files are supported'}), 400

        # Validate PDF before processing
        if not is_valid_pdf(file):
            logger.error(f"Invalid PDF structure: {file.filename}")
            return jsonify({'error': 'Invalid or corrupted PDF file'}), 400

        # Create temp directory if it doesn't exist
        temp_dir = os.path.join(current_app.root_path, 'temp')
        os.makedirs(temp_dir, exist_ok=True)

        try:
            # Save uploaded file with secure filename
            temp_path = os.path.join(temp_dir, secure_filename(file.filename))
            file.save(temp_path)
            
            # Process the file
            logger.info(f"Processing file: {file.filename}")
            results = analyzer.extract_from_pdf(temp_path)
            
            # Validate results
            if not results:
                logger.error("Empty results from analyzer")
                return jsonify({'error': 'Failed to analyze document'}), 500

            if 'error' in results:
                logger.error(f"Analyzer error: {results['error']}")
                return jsonify({'error': results['error']}), 500

            logger.info(f"Successfully analyzed file: {file.filename}")
            return jsonify(results)

        except Exception as e:
            logger.error(f"Processing error: {str(e)}\n{traceback.format_exc()}")
            return jsonify({'error': 'An error occurred while processing the document'}), 500
        
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception as e:
                logger.error(f"Error cleaning up temp file: {str(e)}")

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True)