from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

class Student(db.Model):
    __tablename__ = 'students'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    grade_level = db.Column(db.Integer, default=1)  # 1-6 for Moroccan primary education
    difficulty_level = db.Column(db.String(20), default='easy')  # easy, medium, hard
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    assessments = db.relationship('Assessment', backref='student', lazy=True, cascade='all, delete-orphan')
    progress_records = db.relationship('ProgressRecord', backref='student', lazy=True, cascade='all, delete-orphan')

class Text(db.Model):
    __tablename__ = 'texts'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    content_with_diacritics = db.Column(db.Text)  # تشكيل version
    grade_level = db.Column(db.Integer, default=1)  # 1-6 for Moroccan primary education
    difficulty_level = db.Column(db.String(20), default='easy')  # easy, medium, hard
    category = db.Column(db.String(50))  # religious, literature, stories, etc.
    word_count = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    assessments = db.relationship('Assessment', backref='text', lazy=True)

class Assessment(db.Model):
    __tablename__ = 'assessments'
    
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id'), nullable=False)
    text_id = db.Column(db.Integer, db.ForeignKey('texts.id'), nullable=False)
    
    # Audio files
    original_audio_path = db.Column(db.String(255))
    processed_audio_path = db.Column(db.String(255))
    
    # Transcription results
    transcribed_text = db.Column(db.Text)
    confidence_score = db.Column(db.Float)
    
    # Assessment scores (0-100)
    overall_score = db.Column(db.Float)
    pronunciation_score = db.Column(db.Float)
    fluency_score = db.Column(db.Float)
    accuracy_score = db.Column(db.Float)
    comprehension_score = db.Column(db.Float)
    
    # Detailed analysis
    errors_detected = db.Column(db.JSON)  # List of error objects
    feedback_text = db.Column(db.Text)
    recommendations = db.Column(db.JSON)  # List of improvement suggestions
    
    # Timing information
    reading_duration = db.Column(db.Float)  # seconds
    words_per_minute = db.Column(db.Float)
    
    # Status and metadata
    status = db.Column(db.String(20), default='processing')  # processing, completed, failed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    
    # Relationships
    pronunciation_errors = db.relationship('PronunciationError', backref='assessment', lazy=True, cascade='all, delete-orphan')

class PronunciationError(db.Model):
    __tablename__ = 'pronunciation_errors'
    
    id = db.Column(db.Integer, primary_key=True)
    assessment_id = db.Column(db.Integer, db.ForeignKey('assessments.id'), nullable=False)
    
    # Error details
    word_position = db.Column(db.Integer)  # Position in text
    expected_word = db.Column(db.String(100))
    actual_word = db.Column(db.String(100))
    error_type = db.Column(db.String(50))  # mispronunciation, missing_diacritic, substitution, etc.
    severity = db.Column(db.String(20))  # low, medium, high
    
    # Correction information
    correction_audio_path = db.Column(db.String(255))
    phonetic_correction = db.Column(db.String(200))
    feedback_message = db.Column(db.Text)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ProgressRecord(db.Model):
    __tablename__ = 'progress_records'
    
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id'), nullable=False)
    
    # Progress metrics
    session_date = db.Column(db.Date, default=datetime.utcnow().date)
    texts_completed = db.Column(db.Integer, default=0)
    average_score = db.Column(db.Float)
    improvement_rate = db.Column(db.Float)  # Percentage improvement
    
    # Skill-specific progress
    pronunciation_progress = db.Column(db.Float)
    fluency_progress = db.Column(db.Float)
    accuracy_progress = db.Column(db.Float)
    
    # Learning statistics
    total_words_read = db.Column(db.Integer, default=0)
    total_errors_corrected = db.Column(db.Integer, default=0)
    study_time_minutes = db.Column(db.Integer, default=0)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class AudioFeedback(db.Model):
    __tablename__ = 'audio_feedback'
    
    id = db.Column(db.Integer, primary_key=True)
    assessment_id = db.Column(db.Integer, db.ForeignKey('assessments.id'), nullable=False)
    
    # Audio feedback files
    slow_reading_path = db.Column(db.String(255))
    normal_reading_path = db.Column(db.String(255))
    fast_reading_path = db.Column(db.String(255))
    
    # Feedback metadata
    voice_type = db.Column(db.String(50), default='arabic_native')
    generation_method = db.Column(db.String(50))  # azure_tts, google_tts, etc.
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    assessment = db.relationship('Assessment', backref='audio_feedback', uselist=False)
