import os
import logging
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
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize services
speech_recognizer = ArabicSpeechRecognizer()
assessment_engine = ArabicAssessmentEngine()
feedback_generator = ArabicFeedbackGenerator()
learning_system = LearningManagementSystem()

# Initialize real-time processor
from services.realtime_processor import RealTimeAudioProcessor
realtime_processor = RealTimeAudioProcessor(
    speech_recognizer, assessment_engine, feedback_generator, learning_system
)

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
    """Get a random text based on student's grade level and difficulty"""
    try:
        grade_level = request.args.get('grade_level', 1, type=int)
        difficulty_level = request.args.get('difficulty_level', 'easy')

        # Try to find a matching text from database
        text = Text.query.filter_by(
            grade_level=grade_level,
            difficulty_level=difficulty_level
        ).first()

        if text:
            return jsonify({
                'success': True,
                'text': {
                    'id': text.id,
                    'title': text.title,
                    'content': text.content,
                    'content_with_diacritics': text.content_with_diacritics,
                    'difficulty_level': text.difficulty_level,
                    'category': text.category
                }
            })
        else:
            # If no text found, create a demo text based on grade level
            demo_texts = {
                1: {
                    'easy': 'هَذَا بَيْتٌ جَمِيلٌ. فِي الْبَيْتِ أُسْرَةٌ سَعِيدَةٌ. الْأَبُ يَعْمَلُ فِي الْمَكْتَبِ. الْأُمُّ تَطْبُخُ الطَّعَامَ.',
                    'medium': 'كَانَ يَا مَا كَانَ فِي قَدِيمِ الزَّمَانِ وَلَدٌ صَغِيرٌ يُحِبُّ الْقِرَاءَةَ. كُلَّ يَوْمٍ يَقْرَأُ كِتَابًا جَدِيدًا وَيَتَعَلَّمُ أَشْيَاءَ مُفِيدَةً.',
                    'hard': 'الْعِلْمُ نُورٌ يُضِيءُ طَرِيقَ الْحَيَاةِ. مَنْ طَلَبَ الْعِلْمَ مِنَ الْمَهْدِ إِلَى اللَّحْدِ وَجَدَ السَّعَادَةَ وَالنَّجَاحَ فِي دُنْيَاهُ وَآخِرَتِهِ.'
                },
                2: {
                    'easy': 'فِي الْحَدِيقَةِ أَزْهَارٌ جَمِيلَةٌ وَأَشْجَارٌ خَضْرَاءُ. الطُّيُورُ تُغَرِّدُ فِي الصَّبَاحِ. الشَّمْسُ تُشْرِقُ كُلَّ يَوْمٍ.',
                    'medium': 'يُحِبُّ أَحْمَدُ الذَّهَابَ إِلَى الْمَدْرَسَةِ. يَلْعَبُ مَعَ أَصْدِقَائِهِ فِي الْفُسْحَةِ وَيَدْرُسُ بِجِدٍّ فِي الْفَصْلِ. مُعَلِّمُهُ يُحِبُّهُ كَثِيرًا.',
                    'hard': 'الصَّدَاقَةُ كَنْزٌ ثَمِينٌ لَا يُقَدَّرُ بِثَمَنٍ. الصَّدِيقُ الْوَفِيُّ يَقِفُ بِجَانِبِكَ فِي السَّرَّاءِ وَالضَّرَّاءِ وَيُسَاعِدُكَ فِي الْمُلِمَّاتِ.'
                },
                3: {
                    'easy': 'الْمَاءُ مُهِمٌّ لِلْحَيَاةِ. نَشْرَبُ الْمَاءَ كُلَّ يَوْمٍ. النَّبَاتَاتُ تَحْتَاجُ إِلَى الْمَاءِ لِتَنْمُوَ. يَجِبُ أَنْ نُحَافِظَ عَلَى الْمَاءِ.',
                    'medium': 'فِي فَصْلِ الرَّبِيعِ تَتَفَتَّحُ الْأَزْهَارُ وَتَخْضَرُّ الْأَشْجَارُ. الطَّقْسُ يَصْبِحُ مُعْتَدِلًا وَالنَّاسُ يَخْرُجُونَ لِلتَّنَزُّهِ فِي الْحَدَائِقِ.',
                    'hard': 'الْقِرَاءَةُ غِذَاءُ الْعَقْلِ وَالرُّوحِ. تُوَسِّعُ آفَاقَنَا وَتُنَمِّي مَعْرِفَتَنَا. مَنْ يَقْرَأُ كَثِيرًا يَكْتَسِبُ ثَقَافَةً وَاسِعَةً وَيُصْبِحُ أَكْثَرَ حِكْمَةً.'
                }
            }

            # Get appropriate demo text
            grade_texts = demo_texts.get(grade_level, demo_texts[1])
            demo_content = grade_texts.get(difficulty_level, grade_texts['easy'])

            return jsonify({
                'success': True,
                'text': {
                    'id': 'demo',
                    'title': f'نص تجريبي - الصف {grade_level}',
                    'content': demo_content,
                    'content_with_diacritics': demo_content,
                    'difficulty_level': difficulty_level,
                    'category': 'تجريبي'
                }
            })

    except Exception as e:
        logger.error(f"Error getting random text: {e}")
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

@app.route('/api/generate-text', methods=['POST'])
def generate_arabic_text():
    """Generate Arabic text using AI based on grade level and difficulty"""
    try:
        data = request.get_json()
        grade_level = data.get('grade_level', 1)
        difficulty_level = data.get('difficulty_level', 'easy')
        topic = data.get('topic', 'عام')

        # Use Gemini to generate appropriate Arabic text
        import google.generativeai as genai

        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            return jsonify({
                'success': False,
                'message': 'Google API key not configured'
            }), 500

        genai.configure(api_key=api_key)

        # List available models first
        models = genai.list_models()
        gemini_model = None
        for model in models:
            if 'gemini' in model.name.lower():
                gemini_model = model.name
                break

        if not gemini_model:
            # Fallback text if no model is available
            generated_text = get_fallback_text(grade_level, difficulty_level)
            text_with_diacritics = add_diacritics_to_text(generated_text)

            # Save to database
            text_record = Text(
                title=f"نص تجريبي - الصف {grade_level}",
                content=generated_text,
                content_with_diacritics=text_with_diacritics,
                grade_level=grade_level,
                difficulty_level=difficulty_level,
                category=topic,
                word_count=len(generated_text.split())
            )

            db.session.add(text_record)
            db.session.commit()

            return jsonify({
                'success': True,
                'text': {
                    'id': text_record.id,
                    'title': text_record.title,
                    'content': generated_text,
                    'content_with_diacritics': text_with_diacritics,
                    'grade_level': grade_level,
                    'difficulty_level': difficulty_level,
                    'word_count': text_record.word_count
                }
            })

        model = genai.GenerativeModel(gemini_model)

        # Create prompt based on grade level and difficulty
        difficulty_map = {
            'easy': 'سهل جداً',
            'medium': 'متوسط',
            'hard': 'صعب'
        }

        prompt = f"""
        اكتب نصاً عربياً مناسباً للأطفال في الصف {grade_level} الابتدائي.
        مستوى الصعوبة: {difficulty_map.get(difficulty_level, 'سهل')}
        الموضوع: {topic}

        المتطلبات:
        - النص يجب أن يكون من 3-5 جمل فقط
        - استخدم كلمات بسيطة ومناسبة للعمر
        - اجعل النص تعليمياً وممتعاً
        - تجنب الكلمات الصعبة
        - اكتب النص بدون تشكيل أولاً

        أعطني النص فقط بدون أي تفسير إضافي.
        """

        response = model.generate_content(prompt)
        generated_text = response.text.strip()

        # Generate version with diacritics
        diacritics_prompt = f"""
        أضف التشكيل الكامل (الحركات) للنص العربي التالي:
        {generated_text}

        أعطني النص مع التشكيل فقط بدون أي تفسير.
        """

        diacritics_response = model.generate_content(diacritics_prompt)
        text_with_diacritics = diacritics_response.text.strip()

        # Save to database
        text_record = Text(
            title=f"نص مولد - الصف {grade_level}",
            content=generated_text,
            content_with_diacritics=text_with_diacritics,
            grade_level=grade_level,
            difficulty_level=difficulty_level,
            category=topic,
            word_count=len(generated_text.split())
        )

        db.session.add(text_record)
        db.session.commit()

        return jsonify({
            'success': True,
            'text': {
                'id': text_record.id,
                'title': text_record.title,
                'content': generated_text,
                'content_with_diacritics': text_with_diacritics,
                'grade_level': grade_level,
                'difficulty_level': difficulty_level,
                'word_count': text_record.word_count
            }
        })

    except Exception as e:
        logger.error(f"Error generating Arabic text: {e}")
        # Return fallback text on error
        generated_text = get_fallback_text(grade_level, difficulty_level)
        text_with_diacritics = add_diacritics_to_text(generated_text)

        # Save fallback text
        text_record = Text(
            title=f"نص تجريبي - الصف {grade_level}",
            content=generated_text,
            content_with_diacritics=text_with_diacritics,
            grade_level=grade_level,
            difficulty_level=difficulty_level,
            category=topic,
            word_count=len(generated_text.split())
        )

        db.session.add(text_record)
        db.session.commit()

        return jsonify({
            'success': True,
            'text': {
                'id': text_record.id,
                'title': text_record.title,
                'content': generated_text,
                'content_with_diacritics': text_with_diacritics,
                'grade_level': grade_level,
                'difficulty_level': difficulty_level,
                'word_count': text_record.word_count
            }
        })

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
        logger.info("Real-time WebSocket events:")
        logger.info("  - start_recording (begin real-time recording)")
        logger.info("  - audio_chunk (stream audio data)")
        logger.info("  - stop_recording (finish and assess)")

    # Use SocketIO run instead of app.run for WebSocket support
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)
