import os
import logging
from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename
import tempfile
from datetime import datetime
import json

from models.database import db, Student, Assessment, Text, PronunciationError, AudioFeedback
from services.speech_recognition import ArabicSpeechRecognizer
from services.ai_assessment import ArabicAssessmentEngine
from services.feedback_generation import ArabicFeedbackGenerator
from services.learning_management import LearningManagementSystem

logger = logging.getLogger(__name__)

# Create blueprint
assessment_bp = Blueprint('assessment', __name__, url_prefix='/api/assessments')

# Initialize services (these should be passed from main app)
speech_recognizer = None
assessment_engine = None
feedback_generator = None
learning_system = None

def init_services(sr, ae, fg, ls):
    """Initialize services from main app"""
    global speech_recognizer, assessment_engine, feedback_generator, learning_system
    speech_recognizer = sr
    assessment_engine = ae
    feedback_generator = fg
    learning_system = ls

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'ogg', 'flac'}
UPLOAD_FOLDER = 'uploads'
AUDIO_CACHE_FOLDER = 'audio_cache'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@assessment_bp.route('', methods=['POST'])
def create_assessment():
    """Create a new assessment from uploaded audio"""
    try:
        logger.info(f"Assessment request received - Files: {list(request.files.keys())}, Form: {dict(request.form)}")
        
        # Validate request
        if 'audio' not in request.files:
            logger.error("No audio file in request")
            return jsonify({
                'success': False,
                'message': 'No audio file provided'
            }), 400

        audio_file = request.files['audio']
        logger.info(f"Audio file received: {audio_file.filename}, size: {len(audio_file.read())} bytes")
        audio_file.seek(0)  # Reset file pointer after reading for size
        
        if audio_file.filename == '':
            logger.error("Empty audio filename")
            return jsonify({
                'success': False,
                'message': 'No audio file selected'
            }), 400

        if not allowed_file(audio_file.filename):
            logger.error(f"Invalid file format: {audio_file.filename}")
            return jsonify({
                'success': False,
                'message': 'Invalid file format. Supported formats: wav, mp3, m4a, ogg, flac'
            }), 400

        # Get form data
        student_id_raw = request.form.get('student_id')
        text_id_raw = request.form.get('text_id')
        
        logger.info(f"Received student_id: {student_id_raw}, text_id: {text_id_raw}")
        
        # Handle 'demo' values and convert to integers
        try:
            if student_id_raw == 'demo_student':
                # Get the first available student for demo
                student = Student.query.first()
                if not student:
                    return jsonify({
                        'success': False,
                        'message': 'No students available for demo'
                    }), 400
                student_id = student.id
            else:
                student_id = int(student_id_raw)
        except (ValueError, TypeError):
            return jsonify({
                'success': False,
                'message': 'Invalid student ID format'
            }), 400
            
        try:
            if text_id_raw == 'demo':
                # Get the first available text for demo
                text = Text.query.first()
                if not text:
                    return jsonify({
                        'success': False,
                        'message': 'No texts available for demo'
                    }), 400
                text_id = text.id
            else:
                text_id = int(text_id_raw)
        except (ValueError, TypeError):
            return jsonify({
                'success': False,
                'message': 'Invalid text ID format'
            }), 400

        # Validate student and text exist (if not already found in demo handling)
        if student_id_raw != 'demo_student':
            student = Student.query.get(student_id)
            if not student:
                return jsonify({
                    'success': False,
                    'message': 'Student not found'
                }), 404

        if text_id_raw != 'demo':
            text = Text.query.get(text_id)
            if not text:
                return jsonify({
                    'success': False,
                    'message': 'Text not found'
                }), 404

        # Save uploaded audio file
        filename = secure_filename(audio_file.filename)
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        audio_filename = f"{student_id}_{text_id}_{timestamp}_{filename}"
        audio_path = os.path.join(UPLOAD_FOLDER, audio_filename)
        audio_file.save(audio_path)

        # Create assessment record
        assessment = Assessment(
            student_id=student_id,
            text_id=text_id,
            original_audio_path=audio_path,
            status='processing'
        )

        db.session.add(assessment)
        db.session.commit()

        # Process assessment asynchronously (in a real app, use Celery)
        try:
            result = process_assessment_sync(assessment.id, audio_path, text.content)
            return jsonify({
                'success': True,
                'message': 'Assessment completed successfully',
                'assessment_id': assessment.id,
                'result': result
            })
        except Exception as e:
            logger.error(f"Error processing assessment: {e}")
            assessment.status = 'failed'
            db.session.commit()
            return jsonify({
                'success': False,
                'message': 'Assessment processing failed',
                'assessment_id': assessment.id
            }), 500

    except Exception as e:
        logger.error(f"Error creating assessment: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'Internal server error: {str(e)}'
        }), 500

def process_assessment_sync(assessment_id: int, audio_path: str, reference_text: str):
    """Process assessment synchronously"""
    try:
        assessment = Assessment.query.get(assessment_id)
        if not assessment:
            raise Exception("Assessment not found")

        # Step 1: Speech Recognition
        logger.info(f"Starting speech recognition for assessment {assessment_id}")
        transcription_result = speech_recognizer.transcribe_audio(audio_path, reference_text)

        # Debug logging
        logger.info(f"Transcription result: {transcription_result}")
        logger.info(f"Best transcription: {transcription_result.get('best_transcription', {})}")

        # Check if transcription is empty
        best_transcription = transcription_result.get('best_transcription', {})
        transcribed_text = best_transcription.get('transcription', '').strip()
        confidence_score = best_transcription.get('confidence', 0)

        if not transcribed_text:
            logger.warning(f"Empty transcription for assessment {assessment_id}")
            # Still proceed but with warning

        # Update assessment with transcription
        assessment.transcribed_text = transcribed_text
        assessment.confidence_score = confidence_score
        assessment.reading_duration = transcription_result.get('audio_duration', 0)

        # Step 2: AI Assessment
        logger.info(f"Starting AI assessment for assessment {assessment_id}")
        logger.info(f"Reference text: '{reference_text}'")
        logger.info(f"Transcribed text: '{assessment.transcribed_text}'")
        logger.info(f"Reading duration: {assessment.reading_duration}")

        ai_assessment = assessment_engine.generate_comprehensive_assessment(
            original_text=reference_text,
            transcribed_text=assessment.transcribed_text,
            audio_duration=assessment.reading_duration,
            pronunciation_details=transcription_result.get('pronunciation_assessment')
        )

        # Debug AI assessment result
        logger.info(f"AI assessment result: {ai_assessment}")

        # Update assessment with AI results - with fallback values
        assessment.overall_score = ai_assessment.get('overall_score', 0)
        assessment.pronunciation_score = ai_assessment.get('pronunciation_score', 0)
        assessment.fluency_score = ai_assessment.get('fluency_score', 0)
        assessment.accuracy_score = ai_assessment.get('accuracy_score', 0)
        assessment.comprehension_score = ai_assessment.get('comprehension_score', 0)
        assessment.errors_detected = ai_assessment.get('errors', [])
        assessment.feedback_text = ai_assessment.get('feedback', 'تم إكمال التقييم. يرجى المراجعة.')
        assessment.recommendations = ai_assessment.get('recommendations', [])

        # Calculate words per minute
        word_count = len(reference_text.split())
        if assessment.reading_duration > 0:
            assessment.words_per_minute = (word_count / assessment.reading_duration) * 60

        # Step 3: Generate Feedback
        logger.info(f"Generating feedback for assessment {assessment_id}")
        feedback_dir = os.path.join(AUDIO_CACHE_FOLDER, f"assessment_{assessment_id}")
        os.makedirs(feedback_dir, exist_ok=True)

        # Generate comprehensive feedback package
        feedback_package = feedback_generator.generate_comprehensive_feedback_package(
            {
                'original_text': reference_text,
                'errors': assessment.errors_detected,
                'feedback': assessment.feedback_text,
                'recommendations': assessment.recommendations
            },
            feedback_dir
        )

        # Save audio feedback paths
        if feedback_package.get('progressive_reading'):
            audio_feedback = AudioFeedback(
                assessment_id=assessment_id,
                slow_reading_path=feedback_package['progressive_reading'].get('slow'),
                normal_reading_path=feedback_package['progressive_reading'].get('normal'),
                fast_reading_path=feedback_package['progressive_reading'].get('fast'),
                voice_type='arabic_native',
                generation_method='azure_tts'
            )
            db.session.add(audio_feedback)

        # Save pronunciation errors
        for error in assessment.errors_detected:
            pronunciation_error = PronunciationError(
                assessment_id=assessment_id,
                word_position=error.get('position', -1),
                expected_word=error.get('expected', ''),
                actual_word=error.get('actual', ''),
                error_type=error.get('type', 'unknown'),
                severity=error.get('severity', 'medium'),
                feedback_message=error.get('message', '')
            )
            db.session.add(pronunciation_error)

        # Step 4: Update Learning Management
        logger.info(f"Updating student progress for assessment {assessment_id}")
        learning_system.update_student_progress(
            assessment.student_id,
            {
                'overall_score': assessment.overall_score,
                'pronunciation_score': assessment.pronunciation_score,
                'fluency_score': assessment.fluency_score,
                'accuracy_score': assessment.accuracy_score,
                'word_count': word_count,
                'errors': assessment.errors_detected,
                'audio_duration': assessment.reading_duration
            }
        )

        # Mark assessment as completed
        assessment.status = 'completed'
        assessment.completed_at = datetime.utcnow()

        db.session.commit()

        # Return comprehensive result
        return {
            'assessment_id': assessment_id,
            'transcription': {
                'text': assessment.transcribed_text,
                'confidence': assessment.confidence_score
            },
            'scores': {
                'overall': assessment.overall_score,
                'pronunciation': assessment.pronunciation_score,
                'fluency': assessment.fluency_score,
                'accuracy': assessment.accuracy_score,
                'comprehension': assessment.comprehension_score
            },
            'metrics': {
                'reading_duration': assessment.reading_duration,
                'words_per_minute': assessment.words_per_minute,
                'word_count': word_count
            },
            'errors': assessment.errors_detected,
            'feedback': assessment.feedback_text,
            'recommendations': assessment.recommendations,
            'audio_feedback_available': bool(feedback_package.get('progressive_reading'))
        }

    except Exception as e:
        logger.error(f"Error processing assessment {assessment_id}: {e}")
        # Mark assessment as failed
        if assessment:
            assessment.status = 'failed'
            db.session.commit()
        raise

@assessment_bp.route('/<int:assessment_id>', methods=['GET'])
def get_assessment(assessment_id):
    """Get assessment results"""
    try:
        assessment = Assessment.query.get(assessment_id)
        if not assessment:
            return jsonify({
                'success': False,
                'message': 'Assessment not found'
            }), 404

        # Get related data
        student = Student.query.get(assessment.student_id)
        text = Text.query.get(assessment.text_id)
        audio_feedback = AudioFeedback.query.filter_by(assessment_id=assessment_id).first()
        pronunciation_errors = PronunciationError.query.filter_by(assessment_id=assessment_id).all()

        result = {
            'success': True,
            'assessment': {
                'id': assessment.id,
                'status': assessment.status,
                'created_at': assessment.created_at.isoformat(),
                'completed_at': assessment.completed_at.isoformat() if assessment.completed_at else None,

                'student': {
                    'id': student.id,
                    'name': student.name,
                    'level': student.level
                } if student else None,

                'text': {
                    'id': text.id,
                    'title': text.title,
                    'content': text.content
                } if text else None,

                'transcription': {
                    'text': assessment.transcribed_text,
                    'confidence': assessment.confidence_score
                },

                'scores': {
                    'overall': assessment.overall_score,
                    'pronunciation': assessment.pronunciation_score,
                    'fluency': assessment.fluency_score,
                    'accuracy': assessment.accuracy_score,
                    'comprehension': assessment.comprehension_score
                },

                'metrics': {
                    'reading_duration': assessment.reading_duration,
                    'words_per_minute': assessment.words_per_minute
                },

                'errors': assessment.errors_detected or [],
                'feedback': assessment.feedback_text,
                'recommendations': assessment.recommendations or [],

                'pronunciation_errors': [{
                    'id': error.id,
                    'word_position': error.word_position,
                    'expected_word': error.expected_word,
                    'actual_word': error.actual_word,
                    'error_type': error.error_type,
                    'severity': error.severity,
                    'feedback_message': error.feedback_message
                } for error in pronunciation_errors],

                'audio_feedback': {
                    'slow_reading_available': bool(audio_feedback and audio_feedback.slow_reading_path),
                    'normal_reading_available': bool(audio_feedback and audio_feedback.normal_reading_path),
                    'fast_reading_available': bool(audio_feedback and audio_feedback.fast_reading_path)
                } if audio_feedback else None
            }
        }

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error getting assessment: {e}")
        return jsonify({
            'success': False,
            'message': 'Internal server error'
        }), 500

@assessment_bp.route('/<int:assessment_id>/audio/<speed>', methods=['GET'])
def get_feedback_audio(assessment_id, speed):
    """Get feedback audio file"""
    try:
        if speed not in ['slow', 'normal', 'fast']:
            return jsonify({
                'success': False,
                'message': 'Invalid speed. Use: slow, normal, or fast'
            }), 400

        audio_feedback = AudioFeedback.query.filter_by(assessment_id=assessment_id).first()
        if not audio_feedback:
            return jsonify({
                'success': False,
                'message': 'Audio feedback not found'
            }), 404

        # Get appropriate audio file path
        audio_path = None
        if speed == 'slow':
            audio_path = audio_feedback.slow_reading_path
        elif speed == 'normal':
            audio_path = audio_feedback.normal_reading_path
        elif speed == 'fast':
            audio_path = audio_feedback.fast_reading_path

        if not audio_path or not os.path.exists(audio_path):
            return jsonify({
                'success': False,
                'message': f'{speed.capitalize()} reading audio not available'
            }), 404

        return send_file(audio_path, as_attachment=True)

    except Exception as e:
        logger.error(f"Error getting feedback audio: {e}")
        return jsonify({
            'success': False,
            'message': 'Internal server error'
        }), 500

@assessment_bp.route('/student/<int:student_id>', methods=['GET'])
def get_student_assessments(student_id):
    """Get all assessments for a student"""
    try:
        # Validate student exists
        student = Student.query.get(student_id)
        if not student:
            return jsonify({
                'success': False,
                'message': 'Student not found'
            }), 404

        # Get query parameters
        limit = request.args.get('limit', 10, type=int)
        status = request.args.get('status')

        # Build query
        query = Assessment.query.filter_by(student_id=student_id)

        if status:
            query = query.filter_by(status=status)

        assessments = query.order_by(Assessment.created_at.desc()).limit(limit).all()

        result = {
            'success': True,
            'student': {
                'id': student.id,
                'name': student.name,
                'level': student.level
            },
            'assessments': [{
                'id': assessment.id,
                'text_id': assessment.text_id,
                'status': assessment.status,
                'overall_score': assessment.overall_score,
                'created_at': assessment.created_at.isoformat(),
                'completed_at': assessment.completed_at.isoformat() if assessment.completed_at else None
            } for assessment in assessments]
        }

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error getting student assessments: {e}")
        return jsonify({
            'success': False,
            'message': 'Internal server error'
        }), 500
