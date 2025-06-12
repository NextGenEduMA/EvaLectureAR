import os
import logging
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from flask_migrate import Migrate
from werkzeug.utils import secure_filename
import tempfile
from datetime import datetime
import json

# Import our services
from models.database import db, Student, Assessment, Text, PronunciationError, AudioFeedback
from services.speech_recognition import ArabicSpeechRecognizer
from services.ai_assessment import ArabicAssessmentEngine
from services.feedback_generation import ArabicFeedbackGenerator
from services.learning_management import LearningManagementSystem

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///arabic_assessment.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Initialize extensions
db.init_app(app)
migrate = Migrate(app, db)
CORS(app)

# Initialize services
speech_recognizer = ArabicSpeechRecognizer()
assessment_engine = ArabicAssessmentEngine()
feedback_generator = ArabicFeedbackGenerator()
learning_system = LearningManagementSystem()

# Create upload directories
UPLOAD_FOLDER = 'uploads'
AUDIO_CACHE_FOLDER = 'audio_cache'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_CACHE_FOLDER, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'ogg', 'flac'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve the Arabic test interface"""
    return render_template('arabic_test.html')

@app.route('/test')
def test_english():
    """Serve the English test interface"""
    return render_template('test.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'services': {
            'database': 'connected',
            'speech_recognition': 'initialized',
            'ai_assessment': 'initialized',
            'feedback_generation': 'initialized'
        }
    })

@app.route('/api/students', methods=['POST'])
def create_student():
    """Create a new student profile"""
    try:
        data = request.get_json()
        
        if not data or 'name' not in data or 'email' not in data:
            return jsonify({
                'success': False,
                'message': 'Name and email are required'
            }), 400
        
        result = learning_system.create_student_profile(
            name=data['name'],
            email=data['email'],
            grade_level=data.get('grade_level', 1),
            difficulty_level=data.get('difficulty_level', 'easy')
        )
        
        return jsonify(result), 201 if result['success'] else 400
        
    except Exception as e:
        logger.error(f"Error creating student: {e}")
        return jsonify({
            'success': False,
            'message': 'Internal server error'
        }), 500

@app.route('/api/students/<int:student_id>', methods=['GET'])
def get_student(student_id):
    """Get student profile and progress"""
    try:
        result = learning_system.get_student_profile(student_id)
        return jsonify(result), 200 if result['success'] else 404
        
    except Exception as e:
        logger.error(f"Error getting student: {e}")
        return jsonify({
            'success': False,
            'message': 'Internal server error'
        }), 500

@app.route('/api/students/<int:student_id>/recommendations', methods=['GET'])
def get_recommendations(student_id):
    """Get recommended texts for student"""
    try:
        limit = request.args.get('limit', 5, type=int)
        recommendations = learning_system.get_recommended_texts(student_id, limit)
        
        return jsonify({
            'success': True,
            'recommendations': recommendations
        })
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        return jsonify({
            'success': False,
            'message': 'Internal server error'
        }), 500

@app.route('/api/texts', methods=['POST'])
def create_text():
    """Create a new text for assessment"""
    try:
        data = request.get_json()
        
        if not data or 'title' not in data or 'content' not in data:
            return jsonify({
                'success': False,
                'message': 'Title and content are required'
            }), 400
        
        # Create new text
        text = Text(
            title=data['title'],
            content=data['content'],
            content_with_diacritics=data.get('content_with_diacritics'),
            grade_level=data.get('grade_level', 1),
            difficulty_level=data.get('difficulty_level', 'easy'),
            category=data.get('category', 'عام'),
            word_count=len(data['content'].split())
        )
        
        db.session.add(text)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Text created successfully',
            'text_id': text.id,
            'text': {
                'id': text.id,
                'title': text.title,
                'content': text.content,
                'difficulty_level': text.difficulty_level,
                'category': text.category,
                'word_count': text.word_count
            }
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating text: {e}")
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': 'Internal server error'
        }), 500

@app.route('/api/texts', methods=['GET'])
def get_texts():
    """Get all available texts"""
    try:
        difficulty = request.args.get('difficulty')
        category = request.args.get('category')
        
        query = Text.query
        
        if difficulty:
            query = query.filter_by(difficulty_level=difficulty)
        if category:
            query = query.filter_by(category=category)
        
        texts = query.all()
        
        return jsonify({
            'success': True,
            'texts': [{
                'id': text.id,
                'title': text.title,
                'content': text.content[:200] + '...' if len(text.content) > 200 else text.content,
                'difficulty_level': text.difficulty_level,
                'category': text.category,
                'word_count': text.word_count,
                'created_at': text.created_at.isoformat()
            } for text in texts]
        })
        
    except Exception as e:
        logger.error(f"Error getting texts: {e}")
        return jsonify({
            'success': False,
            'message': 'Internal server error'
        }), 500

# Import and register assessment routes
from routes.assessment_routes import assessment_bp, init_services

# Initialize services for assessment routes
init_services(speech_recognizer, assessment_engine, feedback_generator, learning_system)

# Register blueprints
app.register_blueprint(assessment_bp)

@app.route('/api/students/<int:student_id>/progress', methods=['GET'])
def get_student_progress(student_id):
    """Get detailed progress report for student"""
    try:
        days = request.args.get('days', 30, type=int)
        result = learning_system.generate_progress_report(student_id, days)
        return jsonify(result), 200 if result['success'] else 404
        
    except Exception as e:
        logger.error(f"Error getting student progress: {e}")
        return jsonify({
            'success': False,
            'message': 'Internal server error'
        }), 500

@app.route('/api/demo/quick-test', methods=['POST'])
def quick_test():
    """Quick test endpoint for development"""
    try:
        # Create a sample student and text for testing
        student = Student.query.filter_by(email='test@example.com').first()
        if not student:
            student = Student(
                name='طالب تجريبي',
                email='test@example.com',
                grade_level=1,
                difficulty_level='easy'
            )
            db.session.add(student)
        
        text = Text.query.filter_by(title='نص عربي تجريبي').first()
        if not text:
            text = Text(
                title='نص عربي تجريبي',
                content='السلام عليكم ورحمة الله وبركاته. هذا نص تجريبي للقراءة.',
                content_with_diacritics='السَّلامُ عَلَيْكُمْ وَرَحْمَةُ اللهِ وَبَرَكاتُهُ. هَذا نَصٌّ تَجْرِيبِيٌّ لِلْقِراءَةِ.',
                grade_level=1,
                difficulty_level='easy',
                category='تحية',
                word_count=8
            )
            db.session.add(text)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Test data created',
            'student_id': student.id,
            'text_id': text.id,
            'instructions': {
                'upload_audio': f'POST /api/assessments with student_id={student.id} and text_id={text.id}',
                'get_student': f'GET /api/students/{student.id}',
                'get_recommendations': f'GET /api/students/{student.id}/recommendations'
            }
        })
        
    except Exception as e:
        logger.error(f"Error in quick test: {e}")
        return jsonify({
            'success': False,
            'message': 'Internal server error'
        }), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        logger.info("Database tables created successfully")
        logger.info("Arabic Reading Assessment Platform starting...")
        logger.info("Available endpoints:")
        logger.info("  - POST /api/demo/quick-test (create test data)")
        logger.info("  - POST /api/students (create student)")
        logger.info("  - GET /api/students/<id> (get student profile)")
        logger.info("  - POST /api/assessments (upload audio for assessment)")
        logger.info("  - GET /api/assessments/<id> (get assessment results)")
        logger.info("  - POST /api/texts (create text)")
        logger.info("  - GET /api/texts (list texts)")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
