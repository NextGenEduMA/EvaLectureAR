import os
import google.generativeai as genai
from typing import Dict, List, Tuple
import logging
import json
import re
from difflib import SequenceMatcher
import arabic_reshaper
from bidi.algorithm import get_display

logger = logging.getLogger(__name__)

class ArabicAssessmentEngine:
    def __init__(self):
        self.gemini_model = None
        self._initialize_gemini()
        
        # Arabic text processing patterns
        self.diacritic_pattern = re.compile(r'[\u064B-\u0652\u0670\u0640]')  # Arabic diacritics
        self.arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F]')  # Arabic characters
    
    def _initialize_gemini(self):
        """Initialize Google Gemini AI"""
        try:
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                logger.warning("Google API key not found")
                return
            
            genai.configure(api_key=api_key)
            model_name = os.getenv('GEMINI_MODEL', 'gemini-pro')
            self.gemini_model = genai.GenerativeModel(model_name)
            logger.info(f"Gemini model initialized: {model_name}")
            
        except Exception as e:
            logger.error(f"Error initializing Gemini: {e}")
    
    def normalize_arabic_text(self, text: str) -> str:
        """Normalize Arabic text for comparison"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize Arabic characters
        text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
        text = text.replace('ة', 'ه')
        text = text.replace('ى', 'ي')
        
        return text
    
    def extract_words_without_diacritics(self, text: str) -> List[str]:
        """Extract Arabic words without diacritics"""
        # Remove diacritics
        text_no_diacritics = self.diacritic_pattern.sub('', text)
        
        # Extract Arabic words
        words = re.findall(self.arabic_pattern, text_no_diacritics)
        return [word for word in words if word.strip()]
    
    def detect_missing_diacritics(self, original_text: str, transcribed_text: str) -> List[Dict]:
        """Detect missing or incorrect diacritics"""
        errors = []
        
        try:
            # Remove diacritics from both texts for word alignment
            original_no_diacritics = self.diacritic_pattern.sub('', original_text)
            transcribed_no_diacritics = self.diacritic_pattern.sub('', transcribed_text)
            
            # Split into words
            original_words = original_no_diacritics.split()
            transcribed_words = transcribed_no_diacritics.split()
            original_with_diacritics = original_text.split()
            
            # Align words and check for diacritic differences
            for i, (orig_word, trans_word) in enumerate(zip(original_words, transcribed_words)):
                if i < len(original_with_diacritics):
                    orig_with_diac = original_with_diacritics[i]
                    
                    # Check if diacritics are missing
                    if self.diacritic_pattern.search(orig_with_diac) and orig_word == trans_word:
                        errors.append({
                            'type': 'missing_diacritics',
                            'position': i,
                            'expected': orig_with_diac,
                            'actual': trans_word,
                            'severity': 'medium',
                            'message': f'Missing diacritics in word: {orig_with_diac}'
                        })
            
            return errors
            
        except Exception as e:
            logger.error(f"Error detecting missing diacritics: {e}")
            return []
    
    def detect_pronunciation_errors(self, original_text: str, transcribed_text: str, 
                                  pronunciation_details: Dict = None) -> List[Dict]:
        """Detect various types of pronunciation errors"""
        errors = []
        
        try:
            # Normalize texts
            orig_normalized = self.normalize_arabic_text(original_text)
            trans_normalized = self.normalize_arabic_text(transcribed_text)
            
            # Word-level comparison
            orig_words = orig_normalized.split()
            trans_words = trans_normalized.split()
            
            # Use sequence matcher for alignment
            matcher = SequenceMatcher(None, orig_words, trans_words)
            
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'replace':
                    # Word substitution errors
                    for k in range(max(i2-i1, j2-j1)):
                        orig_idx = i1 + k if i1 + k < i2 else i2 - 1
                        trans_idx = j1 + k if j1 + k < j2 else j2 - 1
                        
                        if orig_idx < len(orig_words) and trans_idx < len(trans_words):
                            errors.append({
                                'type': 'mispronunciation',
                                'position': orig_idx,
                                'expected': orig_words[orig_idx],
                                'actual': trans_words[trans_idx],
                                'severity': self._calculate_error_severity(
                                    orig_words[orig_idx], trans_words[trans_idx]
                                ),
                                'message': f'Mispronounced: {orig_words[orig_idx]} → {trans_words[trans_idx]}'
                            })
                
                elif tag == 'delete':
                    # Missing words
                    for k in range(i1, i2):
                        errors.append({
                            'type': 'omission',
                            'position': k,
                            'expected': orig_words[k],
                            'actual': '',
                            'severity': 'high',
                            'message': f'Omitted word: {orig_words[k]}'
                        })
                
                elif tag == 'insert':
                    # Extra words
                    for k in range(j1, j2):
                        errors.append({
                            'type': 'insertion',
                            'position': i1,
                            'expected': '',
                            'actual': trans_words[k],
                            'severity': 'medium',
                            'message': f'Inserted extra word: {trans_words[k]}'
                        })
            
            # Add Azure pronunciation assessment errors if available
            if pronunciation_details and 'word_details' in pronunciation_details:
                for word_detail in pronunciation_details['word_details']:
                    if word_detail.get('error_type', 'None') != 'None':
                        errors.append({
                            'type': 'pronunciation_accuracy',
                            'position': -1,  # Position not available from Azure
                            'expected': word_detail['word'],
                            'actual': word_detail['word'],
                            'severity': self._map_azure_error_severity(word_detail.get('accuracy_score', 0)),
                            'accuracy_score': word_detail.get('accuracy_score', 0),
                            'error_type': word_detail.get('error_type'),
                            'message': f'Pronunciation issue with: {word_detail["word"]} (Score: {word_detail.get("accuracy_score", 0)})'
                        })
            
            return errors
            
        except Exception as e:
            logger.error(f"Error detecting pronunciation errors: {e}")
            return []
    
    def _calculate_error_severity(self, expected: str, actual: str) -> str:
        """Calculate error severity based on word similarity"""
        similarity = SequenceMatcher(None, expected, actual).ratio()
        
        if similarity > 0.8:
            return 'low'
        elif similarity > 0.5:
            return 'medium'
        else:
            return 'high'
    
    def _map_azure_error_severity(self, accuracy_score: float) -> str:
        """Map Azure accuracy score to severity level"""
        if accuracy_score >= 80:
            return 'low'
        elif accuracy_score >= 60:
            return 'medium'
        else:
            return 'high'
    
    def calculate_fluency_score(self, audio_duration: float, word_count: int, 
                              errors: List[Dict]) -> Dict:
        """Calculate fluency metrics"""
        try:
            if audio_duration <= 0 or word_count <= 0:
                return {'wpm': 0, 'fluency_score': 0, 'pace_rating': 'unknown'}
            
            # Calculate words per minute
            wpm = (word_count / audio_duration) * 60
            
            # Adjust for errors (each error reduces fluency)
            error_penalty = len(errors) * 2  # 2 WPM penalty per error
            adjusted_wpm = max(0, wpm - error_penalty)
            
            # Calculate fluency score (0-100)
            # Optimal Arabic reading speed: 120-180 WPM
            optimal_wpm = 150
            if adjusted_wpm <= optimal_wpm:
                fluency_score = (adjusted_wpm / optimal_wpm) * 100
            else:
                # Penalty for reading too fast
                fluency_score = max(0, 100 - ((adjusted_wpm - optimal_wpm) / optimal_wpm) * 50)
            
            # Determine pace rating
            if wpm < 80:
                pace_rating = 'slow'
            elif wpm > 200:
                pace_rating = 'fast'
            else:
                pace_rating = 'normal'
            
            return {
                'wpm': round(wpm, 2),
                'adjusted_wpm': round(adjusted_wpm, 2),
                'fluency_score': round(fluency_score, 2),
                'pace_rating': pace_rating,
                'error_count': len(errors)
            }
            
        except Exception as e:
            logger.error(f"Error calculating fluency score: {e}")
            return {'wpm': 0, 'fluency_score': 0, 'pace_rating': 'unknown'}
    
    def calculate_accuracy_score(self, original_text: str, transcribed_text: str, 
                               errors: List[Dict]) -> Dict:
        """Calculate reading accuracy score"""
        try:
            orig_words = self.normalize_arabic_text(original_text).split()
            total_words = len(orig_words)
            
            if total_words == 0:
                return {'accuracy_score': 0, 'error_rate': 100}
            
            # Count different types of errors
            error_counts = {
                'mispronunciation': 0,
                'omission': 0,
                'insertion': 0,
                'missing_diacritics': 0,
                'pronunciation_accuracy': 0
            }
            
            for error in errors:
                error_type = error.get('type', 'unknown')
                if error_type in error_counts:
                    error_counts[error_type] += 1
            
            # Calculate weighted error score
            # Different error types have different weights
            error_weights = {
                'mispronunciation': 1.0,
                'omission': 1.5,
                'insertion': 0.5,
                'missing_diacritics': 0.3,
                'pronunciation_accuracy': 0.8
            }
            
            weighted_errors = sum(
                error_counts[error_type] * error_weights.get(error_type, 1.0)
                for error_type in error_counts
            )
            
            # Calculate accuracy score (0-100)
            accuracy_score = max(0, 100 - (weighted_errors / total_words) * 100)
            error_rate = (len(errors) / total_words) * 100
            
            return {
                'accuracy_score': round(accuracy_score, 2),
                'error_rate': round(error_rate, 2),
                'total_words': total_words,
                'total_errors': len(errors),
                'error_breakdown': error_counts
            }
            
        except Exception as e:
            logger.error(f"Error calculating accuracy score: {e}")
            return {'accuracy_score': 0, 'error_rate': 100}
    
    def generate_comprehensive_assessment(self, original_text: str, transcribed_text: str,
                                        audio_duration: float, pronunciation_details: Dict = None) -> Dict:
        """Generate comprehensive reading assessment"""
        try:
            # Detect errors
            pronunciation_errors = self.detect_pronunciation_errors(
                original_text, transcribed_text, pronunciation_details
            )
            diacritic_errors = self.detect_missing_diacritics(original_text, transcribed_text)
            
            all_errors = pronunciation_errors + diacritic_errors
            
            # Calculate metrics
            word_count = len(self.normalize_arabic_text(original_text).split())
            fluency_metrics = self.calculate_fluency_score(audio_duration, word_count, all_errors)
            accuracy_metrics = self.calculate_accuracy_score(original_text, transcribed_text, all_errors)
            
            # Calculate overall score
            pronunciation_score = pronunciation_details.get('pronunciation_score', 0) if pronunciation_details else 0
            overall_score = (
                fluency_metrics['fluency_score'] * 0.3 +
                accuracy_metrics['accuracy_score'] * 0.4 +
                pronunciation_score * 0.3
            )
            
            # Generate AI feedback using Gemini
            ai_feedback = self.generate_ai_feedback(
                original_text, transcribed_text, all_errors, 
                fluency_metrics, accuracy_metrics
            )
            
            return {
                'overall_score': round(overall_score, 2),
                'pronunciation_score': pronunciation_score,
                'fluency_score': fluency_metrics['fluency_score'],
                'accuracy_score': accuracy_metrics['accuracy_score'],
                'comprehension_score': self._estimate_comprehension_score(accuracy_metrics['accuracy_score']),
                
                'detailed_metrics': {
                    'fluency': fluency_metrics,
                    'accuracy': accuracy_metrics,
                    'pronunciation': pronunciation_details or {}
                },
                
                'errors': all_errors,
                'error_summary': self._summarize_errors(all_errors),
                
                'feedback': ai_feedback,
                'recommendations': self._generate_recommendations(all_errors, fluency_metrics, accuracy_metrics)
            }
            
        except Exception as e:
            logger.error(f"Error generating comprehensive assessment: {e}")
            return {
                'overall_score': 0,
                'error': str(e)
            }
    
    def generate_ai_feedback(self, original_text: str, transcribed_text: str,
                           errors: List[Dict], fluency_metrics: Dict, accuracy_metrics: Dict) -> str:
        """Generate personalized feedback using Gemini AI"""
        if not self.gemini_model:
            return self._generate_basic_feedback(errors, fluency_metrics, accuracy_metrics)
        
        try:
            # Prepare prompt for Gemini
            prompt = f"""
            You are an expert Arabic language teacher providing feedback on a student's reading performance.
            
            Original Arabic text: {original_text}
            Student's transcribed reading: {transcribed_text}
            
            Performance metrics:
            - Fluency score: {fluency_metrics.get('fluency_score', 0)}/100
            - Accuracy score: {accuracy_metrics.get('accuracy_score', 0)}/100
            - Words per minute: {fluency_metrics.get('wpm', 0)}
            - Total errors: {len(errors)}
            
            Error details:
            {json.dumps(errors, ensure_ascii=False, indent=2)}
            
            Please provide constructive, encouraging feedback in Arabic and English that:
            1. Acknowledges the student's strengths
            2. Identifies specific areas for improvement
            3. Provides actionable advice for better pronunciation and fluency
            4. Encourages continued practice
            
            Keep the feedback concise but comprehensive, suitable for language learners.
            """
            
            response = self.gemini_model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating AI feedback: {e}")
            return self._generate_basic_feedback(errors, fluency_metrics, accuracy_metrics)
    
    def _generate_basic_feedback(self, errors: List[Dict], fluency_metrics: Dict, accuracy_metrics: Dict) -> str:
        """Generate basic feedback when AI is not available"""
        feedback_parts = []
        
        # Overall performance
        accuracy_score = accuracy_metrics.get('accuracy_score', 0)
        fluency_score = fluency_metrics.get('fluency_score', 0)
        
        if accuracy_score >= 90:
            feedback_parts.append("Excellent reading accuracy! Your pronunciation is very clear.")
        elif accuracy_score >= 75:
            feedback_parts.append("Good reading accuracy with room for minor improvements.")
        else:
            feedback_parts.append("Focus on improving pronunciation accuracy for better comprehension.")
        
        if fluency_score >= 80:
            feedback_parts.append("Your reading pace is well-balanced and natural.")
        elif fluency_score >= 60:
            feedback_parts.append("Good reading pace, try to maintain consistency.")
        else:
            feedback_parts.append("Practice reading at a steady pace to improve fluency.")
        
        # Error-specific feedback
        error_types = set(error.get('type') for error in errors)
        
        if 'missing_diacritics' in error_types:
            feedback_parts.append("Pay attention to Arabic diacritics (تشكيل) for proper pronunciation.")
        
        if 'mispronunciation' in error_types:
            feedback_parts.append("Focus on difficult words and practice their correct pronunciation.")
        
        if 'omission' in error_types:
            feedback_parts.append("Take your time to read all words completely.")
        
        return " ".join(feedback_parts)
    
    def _estimate_comprehension_score(self, accuracy_score: float) -> float:
        """Estimate comprehension score based on accuracy"""
        # Simple heuristic: comprehension correlates with accuracy
        return min(100, accuracy_score * 1.1)
    
    def _summarize_errors(self, errors: List[Dict]) -> Dict:
        """Summarize errors by type and severity"""
        summary = {
            'total_errors': len(errors),
            'by_type': {},
            'by_severity': {'low': 0, 'medium': 0, 'high': 0}
        }
        
        for error in errors:
            error_type = error.get('type', 'unknown')
            severity = error.get('severity', 'medium')
            
            summary['by_type'][error_type] = summary['by_type'].get(error_type, 0) + 1
            summary['by_severity'][severity] += 1
        
        return summary
    
    def _generate_recommendations(self, errors: List[Dict], fluency_metrics: Dict, 
                                accuracy_metrics: Dict) -> List[str]:
        """Generate specific improvement recommendations"""
        recommendations = []
        
        # Fluency recommendations
        wpm = fluency_metrics.get('wpm', 0)
        if wpm < 80:
            recommendations.append("Practice reading aloud daily to improve reading speed")
        elif wpm > 200:
            recommendations.append("Slow down your reading pace to improve accuracy")
        
        # Accuracy recommendations
        accuracy_score = accuracy_metrics.get('accuracy_score', 0)
        if accuracy_score < 70:
            recommendations.append("Focus on pronunciation drills for difficult Arabic sounds")
        
        # Error-specific recommendations
        error_types = set(error.get('type') for error in errors)
        
        if 'missing_diacritics' in error_types:
            recommendations.append("Study Arabic texts with full diacritics (تشكيل)")
            recommendations.append("Practice vowel sounds and their proper pronunciation")
        
        if 'mispronunciation' in error_types:
            recommendations.append("Use audio resources to hear correct pronunciation")
            recommendations.append("Practice with native Arabic speakers when possible")
        
        if 'omission' in error_types:
            recommendations.append("Read more slowly and carefully to avoid skipping words")
        
        # General recommendations
        if len(errors) > 5:
            recommendations.append("Break down complex texts into smaller sections")
            recommendations.append("Record yourself reading and compare with native speakers")
        
        return recommendations[:5]  # Limit to top 5 recommendations
