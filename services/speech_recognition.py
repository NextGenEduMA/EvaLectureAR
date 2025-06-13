import os
import torch
import librosa
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, Wav2Vec2Processor
import azure.cognitiveservices.speech as speechsdk
from typing import Dict, List, Tuple, Optional
import logging
import numpy as np
from pydub import AudioSegment
import tempfile

logger = logging.getLogger(__name__)

class ArabicSpeechRecognizer:
    def __init__(self):
        self.wav2vec2_model = None
        self.wav2vec2_processor = None
        self.azure_speech_config = None
        self._initialize_models()

    def _initialize_models(self):
        """Initialize Wav2Vec2 and Azure Speech models"""
        try:
            # Initialize Wav2Vec2 for Arabic
            model_name = os.getenv('WAV2VEC2_MODEL', 'facebook/wav2vec2-large-xlsr-53')
            logger.info(f"Loading Wav2Vec2 model: {model_name}")

            # Try to load the model with error handling
            try:
                self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained(model_name)
                self.wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(model_name)
                logger.info("Wav2Vec2 model loaded successfully")
            except Exception as model_error:
                logger.warning(f"Failed to load Wav2Vec2 model {model_name}: {model_error}")
                logger.info("Wav2Vec2 will be disabled, using Azure Speech Services only")
                self.wav2vec2_processor = None
                self.wav2vec2_model = None

            # Initialize Azure Speech Services
            azure_key = os.getenv('AZURE_SPEECH_KEY')
            azure_region = os.getenv('AZURE_SPEECH_REGION')

            if azure_key and azure_region:
                self.azure_speech_config = speechsdk.SpeechConfig(
                    subscription=azure_key,
                    region=azure_region
                )
                self.azure_speech_config.speech_recognition_language = "ar-SA"
                logger.info("Azure Speech Services initialized")
            else:
                logger.warning("Azure Speech Services not configured")

        except Exception as e:
            logger.error(f"Error initializing speech recognition models: {e}")
            # Don't raise the exception, allow the app to start without Wav2Vec2
            logger.warning("Speech recognition will use Azure Speech Services only")

    def preprocess_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Enhanced audio preprocessing for better recognition"""
        try:
            # Convert to WAV if necessary
            if not audio_path.lower().endswith('.wav'):
                audio_path = self._convert_to_wav(audio_path)

            # Load audio with librosa - handle PySoundFile warning gracefully
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                warnings.simplefilter("ignore", FutureWarning)
                audio, sample_rate = librosa.load(audio_path, sr=16000)

            # Enhanced audio processing for children's voices
            # Apply noise reduction
            audio = self._reduce_noise(audio)

            # Normalize audio with dynamic range compression
            audio = librosa.util.normalize(audio)

            # Remove silence with more sensitive settings for children
            audio, _ = librosa.effects.trim(audio, top_db=15, frame_length=512, hop_length=64)

            # Apply gentle filtering to enhance clarity (with error handling)
            audio = self._apply_gentle_filter(audio, sample_rate)

            return audio, sample_rate

        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            raise

    def _reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """Simple noise reduction for cleaner speech"""
        try:
            # Apply spectral subtraction for noise reduction
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            phase = np.angle(stft)

            # Estimate noise from the first 0.5 seconds
            noise_frame = int(0.5 * 16000 / 512)  # 0.5 seconds in frames
            noise_profile = np.mean(magnitude[:, :noise_frame], axis=1, keepdims=True)

            # Subtract noise profile with safety factor
            cleaned_magnitude = magnitude - 0.3 * noise_profile
            cleaned_magnitude = np.maximum(cleaned_magnitude, 0.1 * magnitude)

            # Reconstruct audio
            cleaned_stft = cleaned_magnitude * np.exp(1j * phase)
            cleaned_audio = librosa.istft(cleaned_stft)

            return cleaned_audio

        except Exception as e:
            logger.warning(f"Noise reduction failed, using original audio: {e}")
            return audio

    def _apply_gentle_filter(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply gentle filtering to enhance speech clarity"""
        try:
            # High-pass filter to remove low-frequency noise
            audio = librosa.effects.preemphasis(audio, coef=0.97)

            # Band-pass filter for speech frequencies (optimized for children)
            from scipy import signal
            nyquist = sample_rate // 2
            low_freq = 80 / nyquist    # Lower for children's voices
            high_freq = 8000 / nyquist  # Standard upper limit

            # Validate filter frequencies
            if low_freq <= 0 or high_freq >= 1 or low_freq >= high_freq:
                logger.warning(f"Invalid filter frequencies: low={low_freq}, high={high_freq}, using original audio")
                return audio

            b, a = signal.butter(4, [low_freq, high_freq], btype='band')
            audio = signal.filtfilt(b, a, audio)

            return audio

        except Exception as e:
            logger.warning(f"Filtering failed, using original audio: {e}")
            return audio

    def _convert_to_wav(self, audio_path: str) -> str:
        """Convert audio file to WAV format"""
        try:
            audio = AudioSegment.from_file(audio_path)
            wav_path = audio_path.rsplit('.', 1)[0] + '_converted.wav'
            audio.export(wav_path, format="wav")
            return wav_path
        except Exception as e:
            logger.error(f"Error converting audio to WAV: {e}")
            raise

    def _convert_audio_for_azure(self, audio_path: str) -> str:
        """Convert audio to format compatible with Azure Speech Services"""
        try:
            import tempfile
            import soundfile as sf
            import warnings

            # Create temporary file for converted audio
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_file.close()

            # Load audio and ensure it's in the right format for Azure
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                warnings.simplefilter("ignore", FutureWarning)
                audio, sr = librosa.load(audio_path, sr=16000, mono=True)

            # Ensure audio is in the right format
            audio = librosa.util.normalize(audio)

            # Save as WAV with specific format for Azure compatibility
            sf.write(temp_file.name, audio, 16000, format='WAV', subtype='PCM_16')

            logger.info(f"Converted audio for Azure: {temp_file.name}")
            return temp_file.name

        except Exception as e:
            logger.error(f"Error converting audio for Azure: {e}")
            return audio_path

    def transcribe_with_wav2vec2(self, audio_path: str) -> Dict:
        """Transcribe audio using Wav2Vec2 model"""
        if not self.wav2vec2_model or not self.wav2vec2_processor:
            return {
                'transcription': '',
                'confidence': 0.0,
                'error': 'Wav2Vec2 model not available',
                'model': 'wav2vec2'
            }

        try:
            # Preprocess audio
            audio, sample_rate = self.preprocess_audio(audio_path)

            # Process with Wav2Vec2
            inputs = self.wav2vec2_processor(
                audio,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True
            )

            with torch.no_grad():
                logits = self.wav2vec2_model(inputs.input_values).logits

            # Decode predictions
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.wav2vec2_processor.batch_decode(predicted_ids)[0]

            # Calculate confidence score (simplified)
            confidence = torch.softmax(logits, dim=-1).max().item()

            return {
                'transcription': transcription,
                'confidence': confidence,
                'model': 'wav2vec2',
                'language': 'arabic'
            }

        except Exception as e:
            logger.error(f"Error in Wav2Vec2 transcription: {e}")
            return {
                'transcription': '',
                'confidence': 0.0,
                'error': str(e),
                'model': 'wav2vec2'
            }

    def transcribe_with_azure(self, audio_path: str) -> Dict:
        """Enhanced Azure Speech Services transcription with better error handling"""
        if not self.azure_speech_config:
            return {
                'transcription': '',
                'confidence': 0.0,
                'error': 'Azure Speech Services not configured',
                'model': 'azure'
            }

        try:
            # Enhanced Azure configuration for children's speech
            self.azure_speech_config.set_property(
                speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "8000"
            )
            self.azure_speech_config.set_property(
                speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "2000"
            )

            # Convert audio to Azure-compatible format
            azure_audio_path = self._convert_audio_for_azure(audio_path)

            # Create audio config
            audio_config = speechsdk.audio.AudioConfig(filename=azure_audio_path)

            # Create speech recognizer
            speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.azure_speech_config,
                audio_config=audio_config
            )

            # Perform recognition
            result = speech_recognizer.recognize_once()

            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                # Calculate enhanced confidence score
                confidence = self._calculate_azure_confidence(result)

                return {
                    'transcription': result.text.strip(),
                    'confidence': confidence,
                    'model': 'azure',
                    'language': 'arabic',
                    'duration': self._get_audio_duration(audio_path)
                }
            elif result.reason == speechsdk.ResultReason.NoMatch:
                return {
                    'transcription': '',
                    'confidence': 0.0,
                    'error': 'لم يتم التعرف على الكلام - تأكد من وضوح النطق',
                    'model': 'azure'
                }
            else:
                return {
                    'transcription': '',
                    'confidence': 0.0,
                    'error': f'فشل في التعرف على الصوت: {result.reason}',
                    'model': 'azure'
                }

        except Exception as e:
            # Clean up temporary file if created
            if 'azure_audio_path' in locals() and azure_audio_path != audio_path:
                try:
                    os.unlink(azure_audio_path)
                except:
                    pass

            error_msg = f"خطأ في معالجة الصوت: {str(e)}"
            logger.error(f"Error in Azure transcription: {e}")
            return {
                'transcription': '',
                'confidence': 0.0,
                'error': error_msg,
                'model': 'azure'
            }

    def _calculate_azure_confidence(self, result) -> float:
        """Calculate more accurate confidence score from Azure result"""
        try:
            import json

            # Try to get detailed confidence from JSON result
            json_result = result.properties.get(
                speechsdk.PropertyId.SpeechServiceResponse_JsonResult
            )

            if json_result:
                parsed_result = json.loads(json_result)

                # Get confidence from NBest results
                if 'NBest' in parsed_result and parsed_result['NBest']:
                    confidence = parsed_result['NBest'][0].get('Confidence', 0.8)

                    # Additional confidence adjustments for children's speech
                    words = parsed_result['NBest'][0].get('Words', [])
                    if words:
                        # Calculate average word confidence
                        word_confidences = []
                        for word in words:
                            word_conf = word.get('Confidence', 0.8)
                            word_confidences.append(word_conf)

                        avg_word_confidence = sum(word_confidences) / len(word_confidences)
                        # Blend overall confidence with word-level confidence
                        confidence = (confidence * 0.6) + (avg_word_confidence * 0.4)

                    return min(max(confidence, 0.0), 1.0)  # Ensure 0-1 range

            # Enhanced fallback: estimate confidence based on multiple factors
            text_length = len(result.text.strip())
            words = result.text.strip().split()
            word_count = len(words)

            # Base confidence on text characteristics
            if text_length > 20 and word_count > 3:
                return 0.88  # High confidence for longer, multi-word text
            elif text_length > 10 and word_count > 2:
                return 0.78  # Good confidence for medium text
            elif text_length > 5:
                return 0.68  # Medium confidence for short text
            else:
                return 0.55   # Lower confidence for very short text

        except Exception as e:
            logger.warning(f"Could not calculate confidence: {e}")
            return 0.75  # More conservative default confidence

    def get_pronunciation_assessment(self, audio_path: str, reference_text: str) -> Dict:
        """Get detailed pronunciation assessment using Azure"""
        if not self.azure_speech_config:
            return {'error': 'Azure Speech Services not configured'}

        try:
            # Convert audio to Azure-compatible format
            azure_audio_path = self._convert_audio_for_azure(audio_path)

            # Configure pronunciation assessment
            pronunciation_config = speechsdk.PronunciationAssessmentConfig(
                reference_text=reference_text,
                grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
                granularity=speechsdk.PronunciationAssessmentGranularity.Word,
                enable_miscue=True
            )

            # Create audio config
            audio_config = speechsdk.audio.AudioConfig(filename=azure_audio_path)

            # Create speech recognizer with pronunciation assessment
            speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.azure_speech_config,
                audio_config=audio_config
            )

            # Apply pronunciation assessment config
            pronunciation_config.apply_to(speech_recognizer)

            # Perform recognition
            result = speech_recognizer.recognize_once()

            # Clean up temporary file if created
            if azure_audio_path != audio_path:
                try:
                    os.unlink(azure_audio_path)
                except:
                    pass

            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                # Parse pronunciation assessment results
                pronunciation_result = speechsdk.PronunciationAssessmentResult(result)

                return {
                    'accuracy_score': pronunciation_result.accuracy_score,
                    'fluency_score': pronunciation_result.fluency_score,
                    'completeness_score': pronunciation_result.completeness_score,
                    'pronunciation_score': pronunciation_result.pronunciation_score,
                    'transcription': result.text,
                    'word_details': self._parse_word_details(result),
                    'model': 'azure_pronunciation'
                }
            else:
                return {
                    'error': f'Pronunciation assessment failed: {result.reason}',
                    'model': 'azure_pronunciation'
                }

        except Exception as e:
            logger.error(f"Error in pronunciation assessment: {e}")
            return {
                'error': str(e),
                'model': 'azure_pronunciation'
            }

    def _parse_word_details(self, result) -> List[Dict]:
        """Parse detailed word-level pronunciation results"""
        try:
            import json

            # Get detailed results from Azure response
            detailed_result = json.loads(result.properties.get(
                speechsdk.PropertyId.SpeechServiceResponse_JsonResult
            ))

            word_details = []
            if 'NBest' in detailed_result and detailed_result['NBest']:
                words = detailed_result['NBest'][0].get('Words', [])

                for word in words:
                    word_info = {
                        'word': word.get('Word', ''),
                        'accuracy_score': word.get('PronunciationAssessment', {}).get('AccuracyScore', 0),
                        'error_type': word.get('PronunciationAssessment', {}).get('ErrorType', 'None'),
                        'phonemes': []
                    }

                    # Get phoneme details if available
                    phonemes = word.get('PronunciationAssessment', {}).get('Phonemes', [])
                    for phoneme in phonemes:
                        word_info['phonemes'].append({
                            'phoneme': phoneme.get('Phoneme', ''),
                            'accuracy_score': phoneme.get('AccuracyScore', 0)
                        })

                    word_details.append(word_info)

            return word_details

        except Exception as e:
            logger.error(f"Error parsing word details: {e}")
            return []

    def transcribe_audio(self, audio_path: str, reference_text: str = None) -> Dict:
        """Main method to transcribe audio with multiple models"""
        results = {
            'wav2vec2_result': self.transcribe_with_wav2vec2(audio_path),
            'azure_result': self.transcribe_with_azure(audio_path),
            'audio_duration': self._get_audio_duration(audio_path)
        }

        # Add pronunciation assessment if reference text provided
        if reference_text:
            results['pronunciation_assessment'] = self.get_pronunciation_assessment(
                audio_path, reference_text
            )

        # Determine best transcription
        results['best_transcription'] = self._select_best_transcription(results)

        return results

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio file duration in seconds"""
        try:
            audio = AudioSegment.from_file(audio_path)
            return len(audio) / 1000.0  # Convert to seconds
        except Exception as e:
            logger.error(f"Error getting audio duration: {e}")
            return 0.0

    def _select_best_transcription(self, results: Dict) -> Dict:
        """Select the best transcription result based on confidence scores"""
        wav2vec2_confidence = results['wav2vec2_result'].get('confidence', 0)
        azure_confidence = results['azure_result'].get('confidence', 0)

        if azure_confidence > wav2vec2_confidence:
            return {
                'transcription': results['azure_result']['transcription'],
                'confidence': azure_confidence,
                'source': 'azure'
            }
        else:
            return {
                'transcription': results['wav2vec2_result']['transcription'],
                'confidence': wav2vec2_confidence,
                'source': 'wav2vec2'
            }
