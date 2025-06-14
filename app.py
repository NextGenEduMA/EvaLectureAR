import os
import logging
import warnings
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from flask_migrate import Migrate
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import tempfile
from datetime import datetime
import json
import base64
import io

# Suppress common warnings that don't affect functionality
warnings.filterwarnings("ignore", message="Passing `gradient_checkpointing` to a config initialization is deprecated")
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be .* leaked semaphore objects")
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")

# Handle multiprocessing properly for web apps
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

# Import our services
from models.database import db, Student, Assessment, Text, PronunciationError, AudioFeedback
from services.speech_recognition import ArabicSpeechRecognizer
from services.ai_assessment import ArabicAssessmentEngine
from services.feedback_generation import ArabicFeedbackGenerator
from services.learning_management import LearningManagementSystem
# RAG disabled for now - using database texts only

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
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize services
speech_recognizer = ArabicSpeechRecognizer()
assessment_engine = ArabicAssessmentEngine()
feedback_generator = ArabicFeedbackGenerator()
learning_system = LearningManagementSystem()

# Initialize real-time processor
from services.realtime_processor import RealTimeAudioProcessor
realtime_processor = RealTimeAudioProcessor(
    speech_recognizer, assessment_engine, feedback_generator, learning_system)

# RAG system disabled - using database texts for performance

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
    """Serve the student registration page"""
    return render_template('student_registration.html')

@app.route('/evaluation')
def evaluation_page():
    """Serve the evaluation page"""
    return render_template('evaluation.html')

@app.route('/original')
def original_interface():
    """Serve the original complex interface"""
    return render_template('simple_arabic.html')

@app.route('/test')
def test_english():
    """Serve the English test interface"""
    return render_template('test.html')

@app.route('/advanced')
def advanced_interface():
    """Serve the advanced Arabic interface"""
    return render_template('arabic_test.html')

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

        status_code = 201 if result['success'] else 400
        return jsonify(result), status_code

    except Exception as e:
        logger.error(f"Error creating student: {e}")
        return jsonify({
            'success': False,
            'message': f'Internal server error: {str(e)}'
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

@app.route('/api/texts/random', methods=['GET'])
def get_random_text():
    """Get a random text based on student's grade level and difficulty - DATABASE ONLY"""
    try:
        grade_level = request.args.get('grade_level', 1, type=int)
        difficulty_level = request.args.get('difficulty_level', 'easy')

        logger.info(f"Fetching random text: grade={grade_level}, difficulty={difficulty_level}")

        # Query database for matching texts
        texts = Text.query.filter_by(
            grade_level=grade_level,
            difficulty_level=difficulty_level
        ).all()

        if not texts:
            # Try to find texts with just grade level if no difficulty match
            texts = Text.query.filter_by(grade_level=grade_level).all()

        if not texts:
            return jsonify({
                'success': False,
                'message': 'لا توجد نصوص متاحة لهذا المستوى',
                'suggestion': 'Run python populate_texts.py to generate texts'
            }), 404

        # Select random text
        import random
        selected_text = random.choice(texts)

        return jsonify({
            'success': True,
            'text': {
                'id': selected_text.id,
                'title': selected_text.title,
                'content': selected_text.content,
                'content_with_diacritics': selected_text.content_with_diacritics or selected_text.content,
                'difficulty_level': selected_text.difficulty_level,
                'category': selected_text.category,
                'grade_level': selected_text.grade_level,
                'word_count': selected_text.word_count
            },
            'source': 'database',
            'available_count': len(texts),
            'message': f'تم اختيار نص من {len(texts)} نص متاح'
        })

    except Exception as e:
        logger.error(f"Error getting random text: {e}")
        return jsonify({
            'success': False,
            'message': f'خطأ في جلب النص: {str(e)}'
        }), 500

# Import and register assessment routes
from routes.assessment_routes import assessment_bp, init_services

# Initialize services for assessment routes
init_services(speech_recognizer, assessment_engine, feedback_generator, learning_system)

# RAG routes disabled for now to prevent hanging
# from routes.rag_routes import rag_bp

# Register blueprints
app.register_blueprint(assessment_bp)
@app.route('/api/generate-text', methods=['POST'])
def generate_arabic_text():
    """Generate Arabic text - now redirects to database instead of AI generation"""
    try:
        data = request.get_json()
        grade_level = data.get('grade_level', 1)
        difficulty_level = data.get('difficulty_level', 'easy')
        topic = data.get('topic', None)

        logger.info(f"Text generation request: grade={grade_level}, difficulty={difficulty_level}, topic={topic}")

        # Redirect to database-based text selection
        # This is much faster and more reliable than RAG generation

        # Validate parameters
        if grade_level not in range(1, 7):
            return jsonify({
                'success': False,
                'message': 'Grade level must be between 1 and 6'
            }), 400

        if difficulty_level not in ['easy', 'medium', 'hard']:
            return jsonify({
                'success': False,
                'message': 'Difficulty must be easy, medium, or hard'
            }), 400

        # Build query for database texts
        query = Text.query.filter_by(
            grade_level=grade_level,
            difficulty_level=difficulty_level
        )

        # Add topic filter if specified
        if topic and topic != 'عام':
            query = query.filter(Text.category.ilike(f'%{topic}%'))

        # Get all matching texts
        available_texts = query.all()

        if not available_texts:
            # Fallback: try without topic filter
            available_texts = Text.query.filter_by(
                grade_level=grade_level,
                difficulty_level=difficulty_level
            ).all()

        if not available_texts:
            return jsonify({
                'success': False,
                'message': 'لا توجد نصوص متاحة للمعايير المحددة',
                'suggestion': 'Run python populate_texts.py to generate educational texts',
                'fallback_available': True
            }), 404

        # Select a random text from available options
        import random
        selected_text = random.choice(available_texts)

        return jsonify({
            'success': True,
            'text': {
                'id': selected_text.id,
                'title': selected_text.title,
                'content': selected_text.content,
                'content_with_diacritics': selected_text.content,  # Could be enhanced
                'grade_level': selected_text.grade_level,
                'difficulty_level': selected_text.difficulty_level,
                'category': selected_text.category,
                'word_count': selected_text.word_count
            },
            'source': 'database',
            'generation_method': 'pre_generated',
            'available_alternatives': len(available_texts),
            'message': f'تم اختيار نص من {len(available_texts)} نص متاح'
        })

    except Exception as e:
        logger.error(f"Error in text generation: {e}")
        return jsonify({
            'success': False,
            'message': f'خطأ في توليد النص: {str(e)}'
        }), 500

def get_fallback_text(grade_level: int, difficulty_level: str) -> str:
    """Get a pre-defined fallback text based on grade level and difficulty"""
    texts = {
        1: {  # Grade 1
            'easy': 'القط الصغير يلعب في الحديقة. يحب القط اللون الأبيض. هو سعيد جداً.',
            'medium': 'ذهب أحمد إلى المدرسة صباحاً. حمل حقيبته الجديدة. قابل أصدقاءه في الفصل.',
            'hard': 'في يوم جميل، خرجت العصافير من أعشاشها. طارت في السماء الزرقاء. غنت أجمل الألحان.'
        },
        2: {  # Grade 2
            'easy': 'الأرنب يأكل الجزر. يقفز في البستان. يحب اللعب مع أصدقائه.',
            'medium': 'زرع محمد شجرة في الحديقة. سقاها كل يوم بالماء. نمت الشجرة وأصبحت كبيرة.',
            'hard': 'في الغابة الجميلة، عاش أسد شجاع. كان يحمي الحيوانات الصغيرة. ساعد كل من يحتاج المساعدة.'
        }
    }

    # Default to grade 1 if grade level not found
    grade_texts = texts.get(grade_level, texts[1])
    # Default to 'easy' if difficulty not found
    return grade_texts.get(difficulty_level, grade_texts['easy'])

def add_diacritics_to_text(text: str) -> str:
    """Add diacritics to Arabic text"""
    diacritics_map = {
        'القط': 'الْقِطُّ',
        'الصغير': 'الصَّغِيرُ',
        'يلعب': 'يَلْعَبُ',
        'في': 'فِي',
        'الحديقة': 'الْحَدِيقَةِ',
        'يحب': 'يُحِبُّ',
        'اللون': 'اللَّوْنَ',
        'الأبيض': 'الْأَبْيَضَ',
        'هو': 'هُوَ',
        'سعيد': 'سَعِيدٌ',
        'جداً': 'جِدّاً',
        'ذهب': 'ذَهَبَ',
        'أحمد': 'أَحْمَدُ',
        'إلى': 'إِلَى',
        'المدرسة': 'الْمَدْرَسَةِ',
        'صباحاً': 'صَبَاحاً',
        'حمل': 'حَمَلَ',
        'حقيبته': 'حَقِيبَتَهُ',
        'الجديدة': 'الْجَدِيدَةَ',
        'قابل': 'قَابَلَ',
        'أصدقاءه': 'أَصْدِقَاءَهُ',
        'الفصل': 'الْفَصْلِ'
    }

    words = text.split()
    diacritized_words = []

    for word in words:
        diacritized_words.append(diacritics_map.get(word, word))

    return ' '.join(diacritized_words)

@app.route('/api/text-to-speech', methods=['POST'])
def text_to_speech():
    """Convert Arabic text to speech for listening"""
    try:
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({
                'success': False,
                'message': 'النص مطلوب'
            }), 400

        # Use Azure TTS to generate audio
        azure_key = os.getenv('AZURE_SPEECH_KEY')
        azure_region = os.getenv('AZURE_SPEECH_REGION')

        if not azure_key or not azure_region:
            return jsonify({
                'success': False,
                'message': 'Azure Speech Services not configured'
            }), 500

        import azure.cognitiveservices.speech as speechsdk

        # Configure Azure Speech
        speech_config = speechsdk.SpeechConfig(subscription=azure_key, region=azure_region)
        speech_config.speech_synthesis_voice_name = "ar-SA-ZariyahNeural"

        # Create temporary file for audio
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        audio_config = speechsdk.audio.AudioOutputConfig(filename=temp_file.name)

        # Create synthesizer
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config,
            audio_config=audio_config
        )

        # Generate speech
        result = synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            # Return audio file
            return send_file(temp_file.name, as_attachment=True, download_name='arabic_text.wav')
        else:
            return jsonify({
                'success': False,
                'message': 'فشل في توليد الصوت'
            }), 500

    except Exception as e:
        logger.error(f"Error in text-to-speech: {e}")
        return jsonify({
            'success': False,
            'message': f'خطأ في تحويل النص إلى صوت: {str(e)}'
        }), 500

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

# ===== WEBSOCKET HANDLERS FOR REAL-TIME AUDIO =====

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'message': 'متصل بنجاح مع الخادم'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")
    # Cleanup any active sessions for this client
    # In a production app, you'd track session_id to client_id mapping

@socketio.on('start_recording')
def handle_start_recording(data):
    """Start a new recording session"""
    try:
        session_id = request.sid  # Use socket ID as session ID
        student_id = data.get('student_id')
        text_id = data.get('text_id')

        if not student_id or not text_id:
            emit('recording_error', {
                'message': 'يرجى تحديد الطالب والنص'
            })
            return

        result = realtime_processor.start_session(session_id, student_id, text_id)

        if result['success']:
            emit('recording_started', {
                'message': result['message'],
                'session_info': result['session_info']
            })
        else:
            emit('recording_error', {
                'message': result['message']
            })

    except Exception as e:
        logger.error(f"Error starting recording: {e}")
        emit('recording_error', {
            'message': 'خطأ في بدء التسجيل'
        })

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    """Process incoming audio chunk"""
    try:
        session_id = request.sid
        audio_data = data.get('audio')

        if not audio_data:
            return

        result = realtime_processor.process_audio_chunk(session_id, audio_data)

        if result['success']:
            # Send acknowledgment
            emit('chunk_processed', {
                'chunk_count': result['chunk_received']
            })

            # Send real-time feedback if available
            if 'realtime_feedback' in result:
                emit('realtime_feedback', result['realtime_feedback'])
        else:
            emit('processing_error', {
                'message': result['message']
            })

    except Exception as e:
        logger.error(f"Error processing audio chunk: {e}")
        emit('processing_error', {
            'message': 'خطأ في معالجة الصوت'
        })

@socketio.on('stop_recording')
def handle_stop_recording():
    """Stop recording and process final assessment"""
    try:
        session_id = request.sid

        # Notify client that processing has started
        emit('processing_started', {
            'message': 'جاري معالجة التسجيل وإنشاء التقييم...'
        })

        result = realtime_processor.finish_session(session_id)

        if result['success']:
            emit('assessment_complete', {
                'message': result['message'],
                'assessment_id': result['assessment_id'],
                'result': result['result']
            })
        else:
            emit('assessment_error', {
                'message': result['message']
            })

    except Exception as e:
        logger.error(f"Error stopping recording: {e}")
        emit('assessment_error', {
            'message': 'خطأ في إنهاء التسجيل'
        })

@app.route('/api/generate_preview_audio', methods=['POST'])
def generate_preview_audio():
    """Generate TTS preview audio with usage tracking"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        speed = data.get('speed', 'normal')
        session_id = data.get('session_id', 'demo')

        if not text:
            return jsonify({
                'success': False,
                'error': 'لا يوجد نص للتحويل إلى صوت'
            }), 400

        # Check usage limits
        usage_check = feedback_generator.track_preview_usage(session_id, text_preview=True)
        if not usage_check['allowed']:
            return jsonify({
                'success': False,
                'error': usage_check['message']
            }), 429

        # Generate TTS audio
        feedback_generator.cleanup_old_usage_files()  # Clean up periodically
        result = feedback_generator.generate_text_preview_audio(text, speed)

        if result['success']:
            # Create URL for the audio file
            audio_filename = os.path.basename(result['audio_path'])
            audio_url = f'/api/preview_audio/{audio_filename}'

            return jsonify({
                'success': True,
                'audio_path': audio_url,
                'duration': result['duration'],
                'speed': result['speed'],
                'remaining_uses': usage_check['remaining'] - 1,
                'message': usage_check['message']
            })
        else:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 500

    except Exception as e:
        logger.error(f"Error generating preview audio: {e}")
        return jsonify({
            'success': False,
            'error': 'حدث خطأ في تحميل الصوت'
        }), 500

@app.route('/api/preview_audio/<filename>')
def serve_preview_audio(filename):
    """Serve generated preview audio files"""
    try:
        # Security check - only allow wav files
        if not filename.endswith('.wav'):
            return jsonify({'error': 'نوع الملف غير مدعوم'}), 400

        # Look for file in temporary directory
        import tempfile
        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, filename)

        if os.path.exists(audio_path):
            return send_file(audio_path, as_attachment=False, mimetype='audio/wav')
        else:
            return jsonify({'error': 'الملف الصوتي غير موجود'}), 404

    except Exception as e:
        logger.error(f"Error serving preview audio: {e}")
        return jsonify({'error': 'خطأ في تحميل الملف الصوتي'}), 500

@app.route('/api/test/system-status', methods=['GET'])
def test_system_status():
    """Test endpoint to check if all system components are working"""
    try:
        status = {
            'speech_recognition': {
                'wav2vec2': False,
                'azure': False
            },
            'ai_assessment': {
                'gemini': False
            },
            'tts': {
                'azure': False
            },
            'database': False
        }

        # Test Wav2Vec2
        try:
            if speech_recognizer.wav2vec2_model is not None:
                status['speech_recognition']['wav2vec2'] = True
        except:
            pass

        # Test Azure Speech
        try:
            if speech_recognizer.azure_speech_config is not None:
                status['speech_recognition']['azure'] = True
        except:
            pass

        # Test Gemini
        try:
            if assessment_engine.gemini_model is not None:
                status['ai_assessment']['gemini'] = True
        except:
            pass

        # Test Azure TTS
        try:
            if feedback_generator.azure_speech_config is not None:
                status['tts']['azure'] = True
        except:
            pass

        # Test Database
        try:
            from models.database import db
            result = db.session.execute(db.text('SELECT 1')).scalar()
            if result == 1:
                status['database'] = True
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
            pass

        overall_status = all([
            status['speech_recognition']['wav2vec2'],
            status['ai_assessment']['gemini'],
            status['database']
        ])

        return jsonify({
            'success': True,
            'overall_healthy': overall_status,
            'components': status,
            'message': 'النظام يعمل بشكل صحيح' if overall_status else 'بعض المكونات لا تعمل بشكل صحيح'
        })

    except Exception as e:
        logger.error(f"Error in system status check: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'خطأ في فحص حالة النظام'
        }), 500

# Check critical environment variables on startup
def check_environment_variables():
    """Check if critical environment variables are set"""
    critical_vars = ['GOOGLE_API_KEY', 'AZURE_SPEECH_KEY', 'AZURE_SPEECH_REGION']
    missing_vars = []

    for var in critical_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
        logger.warning("Some features may not work properly")
    else:
        logger.info("All critical environment variables are set")

if __name__ == '__main__':
    # Check environment variables before starting the app
    check_environment_variables()

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
        logger.info("Real-time WebSocket events:")
        logger.info("  - start_recording (begin real-time recording)")
        logger.info("  - audio_chunk (stream audio data)")
        logger.info("  - stop_recording (finish and assess)")

    # Use SocketIO run instead of app.run for WebSocket support
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)

@app.route('/api/texts/generate-from-db', methods=['POST'])
def generate_text_from_database():
    """Get a random educational text from the database instead of RAG generation"""
    try:
        data = request.get_json()

        # Extract parameters
        grade_level = data.get('grade_level', 1)
        difficulty_level = data.get('difficulty_level', 'easy')
        topic = data.get('topic', None)  # Optional topic filter

        logger.info(f"Fetching text from database: grade={grade_level}, difficulty={difficulty_level}, topic={topic}")

        # Validate parameters
        if grade_level not in range(1, 7):
            return jsonify({
                'success': False,
                'error': 'Grade level must be between 1 and 6'
            }), 400

        if difficulty_level not in ['easy', 'medium', 'hard']:
            return jsonify({
                'success': False,
                'error': 'Difficulty must be easy, medium, or hard'
            }), 400

        # Build query
        query = Text.query.filter_by(
            grade_level=grade_level,
            difficulty_level=difficulty_level
        )

        # Add topic filter if specified
        if topic:
            query = query.filter(Text.category.ilike(f'%{topic}%'))

        # Get all matching texts
        available_texts = query.all()

        if not available_texts:
            return jsonify({
                'success': False,
                'error': 'No texts found for the specified criteria',
                'suggestion': 'Try different grade level or difficulty, or run populate_texts.py to generate texts'
            }), 404

        # Select a random text
        import random
        selected_text = random.choice(available_texts)

        return jsonify({
            'success': True,
            'text': selected_text.content,
            'text_id': selected_text.id,
            'title': selected_text.title,
            'word_count': selected_text.word_count,
            'category': selected_text.category,
            'grade_level': selected_text.grade_level,
            'difficulty_level': selected_text.difficulty_level,
            'source': 'database',
            'available_texts_count': len(available_texts),
            'message': f'تم اختيار نص من {len(available_texts)} نص متاح'
        })

    except Exception as e:
        logger.error(f"Error fetching text from database: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'خطأ في جلب النص من قاعدة البيانات'
        }), 500

@app.route('/api/texts/stats', methods=['GET'])
def get_text_statistics():
    """Get statistics about available texts in the database"""
    try:
        stats = {
            'total_texts': Text.query.count(),
            'by_grade': {},
            'by_difficulty': {},
            'by_category': {}
        }

        # Statistics by grade level
        for grade in range(1, 7):
            grade_count = Text.query.filter_by(grade_level=grade).count()
            stats['by_grade'][f'grade_{grade}'] = grade_count

        # Statistics by difficulty
        for difficulty in ['easy', 'medium', 'hard']:
            diff_count = Text.query.filter_by(difficulty_level=difficulty).count()
            stats['by_difficulty'][difficulty] = diff_count

        # Statistics by category (simplified)
        all_texts = Text.query.all()
        category_counts = {}
        for text in all_texts:
            if text.category:
                category_counts[text.category] = category_counts.get(text.category, 0) + 1

        # Get top 10 categories
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        stats['by_category'] = dict(sorted_categories)

        return jsonify({
            'success': True,
            'statistics': stats,
            'message': f'قاعدة البيانات تحتوي على {stats["total_texts"]} نص تعليمي'
        })

    except Exception as e:
        logger.error(f"Error getting text statistics: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
