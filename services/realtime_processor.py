import os
import io
import wave
import base64
import tempfile
import logging
from typing import Dict, List
from datetime import datetime
import numpy as np
from pydub import AudioSegment
import librosa

logger = logging.getLogger(__name__)

class RealTimeAudioProcessor:
    def __init__(self, speech_recognizer, assessment_engine, feedback_generator, learning_system):
        self.speech_recognizer = speech_recognizer
        self.assessment_engine = assessment_engine
        self.feedback_generator = feedback_generator
        self.learning_system = learning_system
        
        # Audio buffer for real-time processing
        self.audio_buffer = {}  # session_id -> audio_chunks
        self.session_metadata = {}  # session_id -> metadata
        
    def start_session(self, session_id: str, student_id: int, text_id: int) -> Dict:
        """Start a new real-time recording session"""
        try:
            # Validate student and text
            from models.database import Student, Text
            student = Student.query.get(student_id)
            text = Text.query.get(text_id)
            
            if not student or not text:
                return {
                    'success': False,
                    'message': 'الطالب أو النص غير موجود'
                }
            
            # Initialize session
            self.audio_buffer[session_id] = []
            self.session_metadata[session_id] = {
                'student_id': student_id,
                'text_id': text_id,
                'student_name': student.name,
                'text_title': text.title,
                'text_content': text.content,
                'start_time': datetime.utcnow(),
                'total_chunks': 0,
                'is_active': True
            }
            
            logger.info(f"Started recording session {session_id} for student {student.name}")
            
            return {
                'success': True,
                'message': f'بدأت جلسة التسجيل للطالب {student.name}',
                'session_info': {
                    'student_name': student.name,
                    'text_title': text.title,
                    'text_content': text.content
                }
            }
            
        except Exception as e:
            logger.error(f"Error starting session {session_id}: {e}")
            return {
                'success': False,
                'message': 'خطأ في بدء جلسة التسجيل'
            }
    
    def process_audio_chunk(self, session_id: str, audio_data: str) -> Dict:
        """Process incoming audio chunk in real-time"""
        try:
            if session_id not in self.session_metadata:
                return {
                    'success': False,
                    'message': 'جلسة التسجيل غير موجودة'
                }
            
            # Decode base64 audio data
            audio_bytes = base64.b64decode(audio_data)
            
            # Add to buffer
            self.audio_buffer[session_id].append(audio_bytes)
            self.session_metadata[session_id]['total_chunks'] += 1
            
            # Real-time feedback (every few chunks)
            chunk_count = self.session_metadata[session_id]['total_chunks']
            
            response = {
                'success': True,
                'chunk_received': chunk_count,
                'session_active': True
            }
            
            # Provide real-time feedback every 10 chunks (~2-3 seconds)
            if chunk_count % 10 == 0:
                response['realtime_feedback'] = self._get_realtime_feedback(session_id)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing audio chunk for session {session_id}: {e}")
            return {
                'success': False,
                'message': 'خطأ في معالجة الصوت'
            }
    
    def _get_realtime_feedback(self, session_id: str) -> Dict:
        """Generate real-time feedback during recording"""
        try:
            # Combine recent audio chunks for quick analysis
            recent_chunks = self.audio_buffer[session_id][-10:]  # Last 10 chunks
            
            if not recent_chunks:
                return {'message': 'استمر في القراءة...'}
            
            # Quick audio analysis
            combined_audio = b''.join(recent_chunks)
            
            # Simple volume/activity detection
            audio_array = np.frombuffer(combined_audio, dtype=np.int16)
            volume_level = np.sqrt(np.mean(audio_array**2))
            
            if volume_level < 1000:
                return {
                    'message': 'الصوت منخفض، تحدث بصوت أعلى قليلاً',
                    'volume_warning': True
                }
            elif volume_level > 10000:
                return {
                    'message': 'الصوت عالي، تحدث بصوت أهدأ قليلاً',
                    'volume_warning': True
                }
            else:
                return {
                    'message': 'ممتاز! استمر في القراءة',
                    'volume_ok': True
                }
                
        except Exception as e:
            logger.error(f"Error generating real-time feedback: {e}")
            return {'message': 'استمر في القراءة...'}
    
    def finish_session(self, session_id: str) -> Dict:
        """Finish recording session and process complete audio"""
        try:
            if session_id not in self.session_metadata:
                return {
                    'success': False,
                    'message': 'جلسة التسجيل غير موجودة'
                }
            
            metadata = self.session_metadata[session_id]
            audio_chunks = self.audio_buffer[session_id]
            
            if not audio_chunks:
                return {
                    'success': False,
                    'message': 'لم يتم تسجيل أي صوت'
                }
            
            # Combine all audio chunks
            combined_audio = b''.join(audio_chunks)
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            
            # Convert to proper audio format
            audio_segment = AudioSegment.from_raw(
                io.BytesIO(combined_audio),
                sample_width=2,  # 16-bit
                frame_rate=16000,  # 16kHz
                channels=1  # Mono
            )
            
            audio_segment.export(temp_file.name, format="wav")
            temp_file.close()
            
            # Process with existing assessment pipeline
            result = self._process_complete_assessment(
                temp_file.name,
                metadata['student_id'],
                metadata['text_id'],
                metadata['text_content']
            )
            
            # Cleanup
            os.unlink(temp_file.name)
            self._cleanup_session(session_id)
            
            return result
            
        except Exception as e:
            logger.error(f"Error finishing session {session_id}: {e}")
            self._cleanup_session(session_id)
            return {
                'success': False,
                'message': 'خطأ في معالجة التسجيل النهائي'
            }
    
    def _process_complete_assessment(self, audio_path: str, student_id: int, text_id: int, reference_text: str) -> Dict:
        """Process complete assessment using existing pipeline"""
        try:
            # Create assessment record
            from models.database import db, Assessment
            
            assessment = Assessment(
                student_id=student_id,
                text_id=text_id,
                original_audio_path=audio_path,
                status='processing'
            )
            
            db.session.add(assessment)
            db.session.commit()
            
            # Use existing assessment pipeline
            from routes.assessment_routes import process_assessment_sync
            
            result = process_assessment_sync(assessment.id, audio_path, reference_text)
            
            return {
                'success': True,
                'message': 'تم تقييم القراءة بنجاح! 🎉',
                'assessment_id': assessment.id,
                'result': result
            }
            
        except Exception as e:
            logger.error(f"Error in complete assessment: {e}")
            return {
                'success': False,
                'message': 'خطأ في التقييم النهائي'
            }
    
    def _cleanup_session(self, session_id: str):
        """Clean up session data"""
        try:
            if session_id in self.audio_buffer:
                del self.audio_buffer[session_id]
            if session_id in self.session_metadata:
                del self.session_metadata[session_id]
            logger.info(f"Cleaned up session {session_id}")
        except Exception as e:
            logger.error(f"Error cleaning up session {session_id}: {e}")
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs"""
        return list(self.session_metadata.keys())
    
    def get_session_info(self, session_id: str) -> Dict:
        """Get session information"""
        if session_id in self.session_metadata:
            return self.session_metadata[session_id]
        return None
