import os
import azure.cognitiveservices.speech as speechsdk
from typing import Dict, List, Optional
import logging
import tempfile
import arabic_reshaper
from bidi.algorithm import get_display
from pydub import AudioSegment
import requests
import json

logger = logging.getLogger(__name__)

class ArabicFeedbackGenerator:
    def __init__(self):
        self.azure_speech_config = None
        self._initialize_tts_services()

    def _initialize_tts_services(self):
        """Initialize Text-to-Speech services"""
        try:
            # Initialize Azure Speech Services for TTS
            azure_key = os.getenv('AZURE_SPEECH_KEY')
            azure_region = os.getenv('AZURE_SPEECH_REGION')

            if azure_key and azure_region:
                # Test Azure configuration
                try:
                    test_config = speechsdk.SpeechConfig(
                        subscription=azure_key,
                        region=azure_region
                    )
                    test_config.speech_synthesis_voice_name = "ar-SA-ZariyahNeural"

                    # Validate the configuration works
                    test_synthesizer = speechsdk.SpeechSynthesizer(speech_config=test_config, audio_config=None)

                    self.azure_speech_config = test_config
                    logger.info("Azure TTS initialized successfully with Arabic voice")

                except Exception as config_error:
                    logger.error(f"Azure TTS configuration failed: {config_error}")
                    self.azure_speech_config = None
            else:
                logger.warning("Azure Speech Services not configured - AZURE_SPEECH_KEY and AZURE_SPEECH_REGION environment variables required")
                self.azure_speech_config = None

        except Exception as e:
            logger.error(f"Error initializing TTS services: {e}")
            self.azure_speech_config = None

    def enhance_text_with_diacritics(self, text: str) -> str:
        """Enhance Arabic text with proper diacritics using AI"""
        try:
            # This is a simplified version - in production, you'd use a proper Arabic NLP library
            # or service like Mishkal or similar diacritization services

            # For now, we'll use basic rules and patterns
            enhanced_text = self._apply_basic_diacritics(text)

            # Format for proper display
            reshaped_text = arabic_reshaper.reshape(enhanced_text)
            display_text = get_display(reshaped_text)

            return display_text

        except Exception as e:
            logger.error(f"Error enhancing text with diacritics: {e}")
            return text

    def _apply_basic_diacritics(self, text: str) -> str:
        """Apply basic diacritization rules"""
        # This is a simplified implementation
        # In production, use proper Arabic NLP tools like:
        # - Mishkal (https://github.com/linuxscout/mishkal)
        # - Farasa
        # - CAMeL Tools

        # Basic patterns for common words
        diacritic_rules = {
            'الله': 'اللَّه',
            'محمد': 'مُحَمَّد',
            'كتاب': 'كِتَاب',
            'مدرسة': 'مَدْرَسَة',
            'طالب': 'طَالِب',
            'معلم': 'مُعَلِّم',
            'درس': 'دَرْس',
            'قراءة': 'قِرَاءَة',
            'كلام': 'كَلَام',
            'عربي': 'عَرَبِي'
        }

        enhanced_text = text
        for word, diacritized in diacritic_rules.items():
            enhanced_text = enhanced_text.replace(word, diacritized)

        return enhanced_text

    def generate_correction_audio(self, word: str, output_path: str,
                                speed: str = 'normal') -> bool:
        """Generate audio correction for a specific word"""
        if not self.azure_speech_config:
            logger.warning("Azure TTS not available for audio generation")
            return False

        try:
            # Enhance word with diacritics
            enhanced_word = self.enhance_text_with_diacritics(word)

            # Adjust speech rate based on speed parameter
            rate_map = {
                'slow': '-30%',
                'normal': '0%',
                'fast': '+20%'
            }
            speech_rate = rate_map.get(speed, '0%')

            # Create SSML for better control
            ssml = f"""
            <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="ar-SA">
                <voice name="ar-SA-ZariyahNeural">
                    <prosody rate="{speech_rate}" pitch="0%">
                        {enhanced_word}
                    </prosody>
                </voice>
            </speak>
            """

            # Create synthesizer
            audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=self.azure_speech_config,
                audio_config=audio_config
            )

            # Synthesize speech
            result = synthesizer.speak_ssml_async(ssml).get()

            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                logger.info(f"Audio generated successfully: {output_path}")
                return True
            else:
                logger.error(f"Speech synthesis failed: {result.reason}")
                return False

        except Exception as e:
            logger.error(f"Error generating correction audio: {e}")
            return False

    def generate_progressive_reading_audio(self, text: str, base_output_path: str) -> Dict[str, str]:
        """Generate slow, normal, and fast-paced reading audio"""
        if not self.azure_speech_config:
            return {}

        try:
            # Enhance text with diacritics
            enhanced_text = self.enhance_text_with_diacritics(text)

            audio_files = {}
            speeds = {
                'slow': '-40%',
                'normal': '0%',
                'fast': '+30%'
            }

            for speed_name, rate in speeds.items():
                output_path = f"{base_output_path}_{speed_name}.wav"

                # Create SSML with appropriate rate
                ssml = f"""
                <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="ar-SA">
                    <voice name="ar-SA-ZariyahNeural">
                        <prosody rate="{rate}" pitch="0%">
                            {enhanced_text}
                        </prosody>
                    </voice>
                </speak>
                """

                # Generate audio
                audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)
                synthesizer = speechsdk.SpeechSynthesizer(
                    speech_config=self.azure_speech_config,
                    audio_config=audio_config
                )

                result = synthesizer.speak_ssml_async(ssml).get()

                if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                    audio_files[speed_name] = output_path
                    logger.info(f"Generated {speed_name} reading audio: {output_path}")
                else:
                    logger.error(f"Failed to generate {speed_name} audio: {result.reason}")

            return audio_files

        except Exception as e:
            logger.error(f"Error generating progressive reading audio: {e}")
            return {}

    def generate_error_corrections(self, errors: List[Dict], base_output_dir: str) -> List[Dict]:
        """Generate audio corrections for detected errors"""
        corrections = []

        for i, error in enumerate(errors):
            if error.get('type') in ['mispronunciation', 'missing_diacritics']:
                expected_word = error.get('expected', '')

                if expected_word:
                    # Generate correction audio files
                    correction_files = {}

                    for speed in ['slow', 'normal']:
                        output_path = os.path.join(
                            base_output_dir,
                            f"correction_{i}_{speed}.wav"
                        )

                        if self.generate_correction_audio(expected_word, output_path, speed):
                            correction_files[speed] = output_path

                    if correction_files:
                        correction_data = {
                            'error_id': i,
                            'word': expected_word,
                            'error_type': error.get('type'),
                            'audio_files': correction_files,
                            'phonetic_guide': self._generate_phonetic_guide(expected_word),
                            'practice_tips': self._generate_practice_tips(error)
                        }
                        corrections.append(correction_data)

        return corrections

    def _generate_phonetic_guide(self, word: str) -> str:
        """Generate phonetic pronunciation guide"""
        # This is a simplified version - in production, use proper Arabic phonetic tools
        phonetic_map = {
            'ا': 'aa',
            'ب': 'ba',
            'ت': 'ta',
            'ث': 'tha',
            'ج': 'ja',
            'ح': 'Ha',  # Emphatic H
            'خ': 'kha',
            'د': 'da',
            'ذ': 'dha',
            'ر': 'ra',
            'ز': 'za',
            'س': 'sa',
            'ش': 'sha',
            'ص': 'Sa',  # Emphatic S
            'ض': 'Da',  # Emphatic D
            'ط': 'Ta',  # Emphatic T
            'ظ': 'Za',  # Emphatic Z
            'ع': "'a",  # Ayn
            'غ': 'gha',
            'ف': 'fa',
            'ق': 'qa',
            'ك': 'ka',
            'ل': 'la',
            'م': 'ma',
            'ن': 'na',
            'ه': 'ha',
            'و': 'wa',
            'ي': 'ya'
        }

        phonetic = ''
        for char in word:
            if char in phonetic_map:
                phonetic += phonetic_map[char] + '-'
            else:
                phonetic += char

        return phonetic.rstrip('-')

    def _generate_practice_tips(self, error: Dict) -> List[str]:
        """Generate specific practice tips for each error type"""
        error_type = error.get('type', '')
        tips = []

        if error_type == 'mispronunciation':
            tips.extend([
                "Listen to the correct pronunciation multiple times",
                "Practice the word slowly, focusing on each sound",
                "Record yourself and compare with the model pronunciation",
                "Break the word into syllables and practice each part"
            ])

        elif error_type == 'missing_diacritics':
            tips.extend([
                "Study the word with full diacritics (تشكيل)",
                "Practice vowel sounds: fatha (َ), kasra (ِ), damma (ُ)",
                "Listen to how diacritics change pronunciation",
                "Use texts with complete diacritization for practice"
            ])

        elif error_type == 'omission':
            tips.extend([
                "Read more slowly to avoid skipping words",
                "Point to each word as you read",
                "Practice reading the complete sentence multiple times",
                "Focus on maintaining steady reading pace"
            ])

        return tips

    def generate_comprehensive_feedback_package(self, assessment_result: Dict,
                                              output_dir: str) -> Dict:
        """Generate complete feedback package with audio and text"""
        try:
            os.makedirs(output_dir, exist_ok=True)

            feedback_package = {
                'text_feedback': assessment_result.get('feedback', ''),
                'recommendations': assessment_result.get('recommendations', []),
                'audio_corrections': [],
                'progressive_reading': {},
                'summary_audio': None
            }

            # Generate error corrections
            errors = assessment_result.get('errors', [])
            if errors:
                corrections_dir = os.path.join(output_dir, 'corrections')
                os.makedirs(corrections_dir, exist_ok=True)

                feedback_package['audio_corrections'] = self.generate_error_corrections(
                    errors, corrections_dir
                )

            # Generate progressive reading audio if original text available
            original_text = assessment_result.get('original_text', '')
            if original_text:
                progressive_base = os.path.join(output_dir, 'progressive_reading')
                feedback_package['progressive_reading'] = self.generate_progressive_reading_audio(
                    original_text, progressive_base
                )

            # Generate summary feedback audio
            feedback_text = assessment_result.get('feedback', '')
            if feedback_text:
                summary_audio_path = os.path.join(output_dir, 'feedback_summary.wav')
                if self.generate_correction_audio(feedback_text, summary_audio_path):
                    feedback_package['summary_audio'] = summary_audio_path

            return feedback_package

        except Exception as e:
            logger.error(f"Error generating comprehensive feedback package: {e}")
            return {}

    def create_practice_sequence(self, errors: List[Dict], difficulty_level: str = 'beginner') -> Dict:
        """Create a structured practice sequence based on errors"""
        try:
            practice_sequence = {
                'level': difficulty_level,
                'total_exercises': 0,
                'estimated_time_minutes': 0,
                'exercises': []
            }

            # Group errors by type for structured practice
            error_groups = {}
            for error in errors:
                error_type = error.get('type', 'general')
                if error_type not in error_groups:
                    error_groups[error_type] = []
                error_groups[error_type].append(error)

            exercise_id = 1

            # Create exercises for each error type
            for error_type, error_list in error_groups.items():
                if error_type == 'mispronunciation':
                    exercise = {
                        'id': exercise_id,
                        'type': 'pronunciation_drill',
                        'title': 'Pronunciation Practice',
                        'description': 'Practice correct pronunciation of mispronounced words',
                        'words': [error.get('expected', '') for error in error_list],
                        'estimated_time': len(error_list) * 2,  # 2 minutes per word
                        'difficulty': self._calculate_exercise_difficulty(error_list)
                    }
                    practice_sequence['exercises'].append(exercise)
                    exercise_id += 1

                elif error_type == 'missing_diacritics':
                    exercise = {
                        'id': exercise_id,
                        'type': 'diacritics_practice',
                        'title': 'Diacritics (تشكيل) Practice',
                        'description': 'Learn proper vowel pronunciation with diacritics',
                        'words': [error.get('expected', '') for error in error_list],
                        'estimated_time': len(error_list) * 3,  # 3 minutes per word
                        'difficulty': 'intermediate'
                    }
                    practice_sequence['exercises'].append(exercise)
                    exercise_id += 1

                elif error_type == 'fluency':
                    exercise = {
                        'id': exercise_id,
                        'type': 'fluency_drill',
                        'title': 'Reading Fluency Practice',
                        'description': 'Improve reading speed and smoothness',
                        'estimated_time': 10,
                        'difficulty': 'beginner'
                    }
                    practice_sequence['exercises'].append(exercise)
                    exercise_id += 1

            # Calculate totals
            practice_sequence['total_exercises'] = len(practice_sequence['exercises'])
            practice_sequence['estimated_time_minutes'] = sum(
                ex.get('estimated_time', 0) for ex in practice_sequence['exercises']
            )

            return practice_sequence

        except Exception as e:
            logger.error(f"Error creating practice sequence: {e}")
            return {}

    def _calculate_exercise_difficulty(self, errors: List[Dict]) -> str:
        """Calculate exercise difficulty based on error severity"""
        high_severity_count = sum(1 for error in errors if error.get('severity') == 'high')
        total_errors = len(errors)

        if total_errors == 0:
            return 'beginner'

        severity_ratio = high_severity_count / total_errors

        if severity_ratio > 0.6:
            return 'advanced'
        elif severity_ratio > 0.3:
            return 'intermediate'
        else:
            return 'beginner'

    def generate_text_preview_audio(self, text: str, speed: str = "normal") -> Dict:
        """Generate TTS audio preview with enhanced error handling and retries"""
        if not self.azure_speech_config:
            return {
                'success': False,
                'error': 'خدمة تحويل النص إلى صوت غير متوفرة حاليًا',
                'audio_path': None
            }

        try:
            # Validate and clean input text
            if not text or not text.strip():
                return {
                    'success': False,
                    'error': 'لا يوجد نص للتحويل إلى صوت',
                    'audio_path': None
                }

            # Clean and prepare text
            clean_text = text.strip()
            # Remove any problematic characters that might cause TTS issues
            clean_text = clean_text.replace('"', '').replace("'", '').replace('<', '').replace('>', '')

            # Limit text length to avoid TTS timeout issues
            if len(clean_text) > 1000:
                clean_text = clean_text[:1000] + "..."

            # Create a fresh speech config to avoid any config conflicts
            azure_key = os.getenv('AZURE_SPEECH_KEY')
            azure_region = os.getenv('AZURE_SPEECH_REGION')

            if not azure_key or not azure_region:
                return {
                    'success': False,
                    'error': 'خدمة Azure Speech غير مكونة بشكل صحيح',
                    'audio_path': None
                }

            # Create fresh config for each request
            speech_config = speechsdk.SpeechConfig(subscription=azure_key, region=azure_region)
            speech_config.speech_synthesis_voice_name = "ar-SA-ZariyahNeural"

            # Set output format for better compatibility
            speech_config.set_speech_synthesis_output_format(
                speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm
            )

            # Always use normal speed for simplified UI
            rate = "1.0"

            # Escape text for SSML
            import xml.sax.saxutils as saxutils
            escaped_text = saxutils.escape(clean_text)

            ssml_text = f'''<?xml version="1.0" encoding="UTF-8"?>
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="ar-SA">
    <voice name="ar-SA-ZariyahNeural">
        <prosody rate="{rate}">{escaped_text}</prosody>
    </voice>
</speak>'''

            # Create temporary file for audio output
            temp_audio_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix='.wav',
                prefix=f'text_preview_'
            )
            temp_audio_file.close()

            # Configure audio output
            audio_config = speechsdk.audio.AudioOutputConfig(filename=temp_audio_file.name)

            # Create synthesizer with fresh config
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config,
                audio_config=audio_config
            )

            # Perform synthesis with retries for rate limiting
            max_retries = 2
            retry_delay = 1

            for attempt in range(max_retries + 1):
                try:
                    logger.info(f"Starting TTS synthesis attempt {attempt + 1} for text length: {len(clean_text)}")

                    # Use synchronous call
                    result = synthesizer.speak_ssml_async(ssml_text).get()

                    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                        logger.info("TTS synthesis completed successfully")
                        return {
                            'success': True,
                            'audio_path': temp_audio_file.name,
                            'duration': self._get_audio_duration(temp_audio_file.name),
                            'speed': 'normal',
                            'voice': 'ar-SA-ZariyahNeural'
                        }
                    elif result.reason == speechsdk.ResultReason.Canceled:
                        # Get detailed cancellation reason
                        cancellation = result.cancellation_details
                        error_details = f"TTS Canceled - Reason: {cancellation.reason}"
                        if cancellation.error_details:
                            error_details += f", Details: {cancellation.error_details}"

                        logger.error(error_details)

                        # Check if it's a rate limiting issue that we can retry
                        if cancellation.error_details and ("throttled" in str(cancellation.error_details).lower() or
                                                          "rate" in str(cancellation.error_details).lower() or
                                                          "quota" in str(cancellation.error_details).lower()):
                            if attempt < max_retries:
                                logger.info(f"Rate limited, retrying in {retry_delay} seconds...")
                                import time
                                time.sleep(retry_delay)
                                retry_delay *= 2  # Exponential backoff
                                continue

                        # Clean up temp file
                        try:
                            os.unlink(temp_audio_file.name)
                        except:
                            pass

                        # Provide user-friendly error message
                        if "authentication" in str(cancellation.error_details).lower():
                            return {
                                'success': False,
                                'error': 'خطأ في المصادقة مع خدمة Azure Speech',
                                'audio_path': None
                            }
                        elif "quota" in str(cancellation.error_details).lower() or "throttled" in str(cancellation.error_details).lower():
                            return {
                                'success': False,
                                'error': 'الخدمة مشغولة حاليًا. يمكنك البدء في التسجيل بدون استماع.',
                                'audio_path': None
                            }
                        else:
                            return {
                                'success': False,
                                'error': 'فشل في تحويل النص إلى صوت. يمكنك البدء في التسجيل بدون استماع.',
                                'audio_path': None
                            }
                    else:
                        logger.error(f"TTS synthesis failed with reason: {result.reason}")
                        if attempt < max_retries:
                            logger.info(f"Synthesis failed, retrying in {retry_delay} seconds...")
                            import time
                            time.sleep(retry_delay)
                            retry_delay *= 2
                            continue

                        # Clean up temp file
                        try:
                            os.unlink(temp_audio_file.name)
                        except:
                            pass

                        return {
                            'success': False,
                            'error': 'فشل في توليد الصوت. يمكنك البدء في التسجيل بدون استماع.',
                            'audio_path': None
                        }

                except Exception as synthesis_error:
                    logger.error(f"Exception during TTS synthesis attempt {attempt + 1}: {synthesis_error}")
                    if attempt < max_retries:
                        logger.info(f"Exception occurred, retrying in {retry_delay} seconds...")
                        import time
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue

                    # Clean up temp file
                    try:
                        os.unlink(temp_audio_file.name)
                    except:
                        pass

                    return {
                        'success': False,
                        'error': 'خطأ في معالجة النص. يمكنك البدء في التسجيل بدون استماع.',
                        'audio_path': None
                    }

            # If we get here, all attempts failed
            try:
                os.unlink(temp_audio_file.name)
            except:
                pass

            return {
                'success': False,
                'error': 'فشل في توليد الصوت بعد عدة محاولات. يمكنك البدء في التسجيل بدون استماع.',
                'audio_path': None
            }

        except Exception as e:
            logger.error(f"Error in generate_text_preview_audio: {e}")
            return {
                'success': False,
                'error': 'حدث خطأ غير متوقع. يمكنك البدء في التسجيل بدون استماع.',
                'audio_path': None
            }

    def create_reading_practice_audio(self, text: str, assessment_id: str) -> Dict:
        """Create practice audio files at different speeds for reading practice"""
        try:
            audio_files = {}
            speeds = ['slow', 'normal', 'fast']

            # Create audio cache directory for this assessment
            cache_dir = os.path.join('audio_cache', f'assessment_{assessment_id}')
            os.makedirs(cache_dir, exist_ok=True)

            for speed in speeds:
                # Generate audio for each speed
                result = self.generate_text_preview_audio(text, speed)

                if result['success']:
                    # Move to permanent cache location
                    cache_filename = f'progressive_reading_{speed}.wav'
                    cache_path = os.path.join(cache_dir, cache_filename)

                    # Copy file to cache
                    import shutil
                    shutil.copy2(result['audio_path'], cache_path)

                    # Clean up temporary file
                    os.unlink(result['audio_path'])

                    audio_files[speed] = {
                        'path': cache_path,
                        'duration': result['duration'],
                        'url': f'/audio/{assessment_id}/{cache_filename}'
                    }
                else:
                    logger.error(f"Failed to generate {speed} audio: {result.get('error')}")

            return {
                'success': len(audio_files) > 0,
                'audio_files': audio_files,
                'cache_dir': cache_dir
            }

        except Exception as e:
            logger.error(f"Error creating practice audio: {e}")
            return {
                'success': False,
                'error': str(e),
                'audio_files': {}
            }

    def track_preview_usage(self, session_id: str, text_preview: bool = False) -> Dict:
        """Track usage of preview features with limits - improved reliability"""
        try:
            # This would typically use a database to track usage
            # For now, we'll use session-based tracking with better error handling

            # Create a safer usage directory
            usage_dir = os.path.join(tempfile.gettempdir(), 'tts_usage')
            os.makedirs(usage_dir, exist_ok=True)

            usage_file = os.path.join(usage_dir, f"usage_{session_id}.json")

            try:
                with open(usage_file, 'r') as f:
                    usage_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                usage_data = {
                    'text_preview_count': 0,
                    'last_reset': None
                }

            # Check daily limits (3 previews per session)
            if text_preview:
                if usage_data['text_preview_count'] >= 3:
                    return {
                        'allowed': False,
                        'remaining': 0,
                        'message': 'تم استنفاد محاولات الاستماع المسموحة (3 مرات لكل جلسة)'
                    }

                usage_data['text_preview_count'] += 1

                # Save updated usage with error handling
                try:
                    with open(usage_file, 'w') as f:
                        json.dump(usage_data, f)
                except Exception as write_error:
                    logger.warning(f"Could not save usage data: {write_error}")

                return {
                    'allowed': True,
                    'remaining': 3 - usage_data['text_preview_count'],
                    'message': f'متبقي {3 - usage_data["text_preview_count"]} محاولات استماع'
                }

            return {
                'allowed': True,
                'remaining': 3 - usage_data['text_preview_count'],
                'message': 'يمكنك الاستماع للنص قبل القراءة'
            }

        except Exception as e:
            logger.error(f"Error tracking preview usage: {e}")
            # On error, be permissive and allow usage
            return {
                'allowed': True,
                'remaining': 3,
                'message': 'خدمة التتبع غير متوفرة - يمكنك الاستماع'
            }

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio file duration in seconds"""
        try:
            audio = AudioSegment.from_file(audio_path)
            return len(audio) / 1000.0
        except Exception as e:
            logger.error(f"Error getting audio duration: {e}")
            return 0.0

    def cleanup_temporary_audio(self, audio_path: str):
        """Clean up temporary audio files"""
        try:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
        except Exception as e:
            logger.error(f"Error cleaning up audio file {audio_path}: {e}")

    def _validate_azure_config(self) -> Dict:
        """Validate Azure TTS configuration"""
        try:
            azure_key = os.getenv('AZURE_SPEECH_KEY')
            azure_region = os.getenv('AZURE_SPEECH_REGION')

            if not azure_key:
                return {
                    'valid': False,
                    'error': 'AZURE_SPEECH_KEY environment variable not set',
                    'user_message': 'مفتاح Azure Speech غير محدد في متغيرات البيئة'
                }

            if not azure_region:
                return {
                    'valid': False,
                    'error': 'AZURE_SPEECH_REGION environment variable not set',
                    'user_message': 'منطقة Azure Speech غير محددة في متغيرات البيئة'
                }

            # Test connection with a simple synthesis
            try:
                test_config = speechsdk.SpeechConfig(subscription=azure_key, region=azure_region)
                test_config.speech_synthesis_voice_name = "ar-SA-ZariyahNeural"

                # Quick test without actual synthesis
                synthesizer = speechsdk.SpeechSynthesizer(speech_config=test_config, audio_config=None)

                return {
                    'valid': True,
                    'error': None,
                    'user_message': 'خدمة Azure TTS متاحة'
                }

            except Exception as test_error:
                return {
                    'valid': False,
                    'error': f'Azure TTS connection test failed: {test_error}',
                    'user_message': 'فشل في الاتصال بخدمة Azure TTS'
                }

        except Exception as e:
            return {
                'valid': False,
                'error': f'Config validation error: {e}',
                'user_message': 'خطأ في التحقق من إعدادات الصوت'
            }

    def generate_fallback_audio_message(self, text: str, speed: str = "normal") -> Dict:
        """Generate a fallback response when TTS is not available"""
        return {
            'success': False,
            'error': 'خدمة تحويل النص إلى صوت غير متوفرة حالياً',
            'audio_path': None,
            'fallback_message': f'النص للقراءة ({speed}): {text[:100]}{"..." if len(text) > 100 else ""}',
            'suggestion': 'يرجى قراءة النص بصوت عالٍ للتدرب على النطق'
        }

    def cleanup_old_usage_files(self):
        """Clean up old usage tracking files to prevent accumulation"""
        try:
            usage_dir = os.path.join(tempfile.gettempdir(), 'tts_usage')
            if not os.path.exists(usage_dir):
                return

            import time
            current_time = time.time()

            for filename in os.listdir(usage_dir):
                if filename.startswith('usage_') and filename.endswith('.json'):
                    file_path = os.path.join(usage_dir, filename)
                    try:
                        # Remove files older than 24 hours
                        if os.path.getmtime(file_path) < (current_time - 86400):
                            os.unlink(file_path)
                            logger.info(f"Cleaned up old usage file: {filename}")
                    except Exception as e:
                        logger.warning(f"Error cleaning up usage file {filename}: {e}")
        except Exception as e:
            logger.error(f"Error during usage file cleanup: {e}")
