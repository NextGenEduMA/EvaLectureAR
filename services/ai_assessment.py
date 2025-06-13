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
            # Use the newer model name
            model_name = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
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

    def generate_personalized_assessment(self, student_data: Dict, text_data: Dict,
                                       transcription_result: Dict, errors: List[Dict],
                                       pronunciation_details: Dict = None) -> Dict:
        """Generate comprehensive personalized assessment using AI"""
        try:
            if not self.gemini_model:
                return self._generate_fallback_assessment(student_data, errors)

            # Prepare assessment prompt
            prompt = self._build_assessment_prompt(
                student_data, text_data, transcription_result, errors, pronunciation_details
            )

            # Generate AI assessment
            response = self.gemini_model.generate_content(prompt)
            ai_feedback = response.text

            # Calculate scores
            pronunciation_score = self._calculate_pronunciation_score(pronunciation_details, errors)
            fluency_metrics = self.calculate_fluency_score(
                transcription_result.get('duration', 0),
                len(text_data.get('content', '').split()),
                errors
            )
            accuracy_metrics = self.calculate_accuracy_score(
                text_data.get('content', ''),
                transcription_result.get('transcription', ''),
                errors
            )

            # Calculate overall score with child-friendly weighting
            overall_score = self._calculate_child_friendly_score(
                pronunciation_score, fluency_metrics['fluency_score'],
                accuracy_metrics['accuracy_score'], student_data
            )

            return {
                'overall_score': overall_score,
                'pronunciation_score': pronunciation_score,
                'fluency_score': fluency_metrics['fluency_score'],
                'accuracy_score': accuracy_metrics['accuracy_score'],
                'comprehension_score': self._estimate_comprehension_score(accuracy_metrics, fluency_metrics),
                'words_per_minute': fluency_metrics.get('wpm', 0),
                'reading_duration': transcription_result.get('duration', 0),
                'errors_detected': errors,
                'personalized_feedback': self._format_child_friendly_feedback(ai_feedback, student_data),
                'recommendations': self._generate_recommendations(student_data, errors, overall_score),
                'encouragement': self._generate_encouragement(overall_score, student_data),
                'areas_for_improvement': self._identify_improvement_areas(errors, fluency_metrics, accuracy_metrics),
                'strengths': self._identify_strengths(overall_score, pronunciation_score, fluency_metrics)
            }

        except Exception as e:
            logger.error(f"Error generating personalized assessment: {e}")
            return self._generate_fallback_assessment(student_data, errors)

    def _build_assessment_prompt(self, student_data: Dict, text_data: Dict,
                                transcription_result: Dict, errors: List[Dict],
                                pronunciation_details: Dict = None) -> str:
        """Build comprehensive prompt for AI assessment"""

        grade_level = student_data.get('grade_level', 1)
        difficulty = student_data.get('difficulty_level', 'easy')
        student_name = student_data.get('name', 'الطالب')

        prompt = f"""
أنت معلم لغة عربية متخصص في تقييم القراءة للأطفال. يرجى تقييم أداء الطالب التالي:

معلومات الطالب:
- الاسم: {student_name}
- المستوى الدراسي: الصف {grade_level}
- مستوى الصعوبة: {difficulty}

النص المطلوب قراءته:
"{text_data.get('content', '')}"

النص الذي قرأه الطالب فعلياً:
"{transcription_result.get('transcription', '')}"

الأخطاء المكتشفة: {len(errors)} خطأ
"""

        if errors:
            prompt += "\nتفاصيل الأخطاء:\n"
            for i, error in enumerate(errors[:5]):  # Limit to 5 errors for brevity
                prompt += f"- {error.get('type', 'خطأ')}: {error.get('message', '')}\n"

        if pronunciation_details:
            prompt += f"\nدرجة النطق من Azure: {pronunciation_details.get('pronunciation_score', 'غير متوفر')}\n"

        prompt += f"""
يرجى تقديم تقييم شامل ومشجع مناسب لطفل في الصف {grade_level} يتضمن:

1. نقاط القوة في الأداء
2. المجالات التي تحتاج تحسين
3. نصائح محددة وعملية للتحسين
4. كلمات تشجيع إيجابية
5. أهداف واضحة للتحسن

اجعل التقييم:
- مشجعاً وإيجابياً
- مناسباً لعمر الطفل
- محدداً وعملياً
- باللغة العربية الواضحة
"""

        return prompt

    def _calculate_child_friendly_score(self, pronunciation: float, fluency: float,
                                      accuracy: float, student_data: Dict) -> float:
        """Calculate overall score with child-friendly weighting"""
        grade_level = student_data.get('grade_level', 1)
        difficulty = student_data.get('difficulty_level', 'easy')

        # Adjust weights based on grade level
        if grade_level <= 2:
            # For younger children, emphasize accuracy over speed
            weights = {'accuracy': 0.5, 'pronunciation': 0.3, 'fluency': 0.2}
        elif grade_level <= 4:
            # Balanced approach for middle grades
            weights = {'accuracy': 0.4, 'pronunciation': 0.35, 'fluency': 0.25}
        else:
            # For older children, include fluency more heavily
            weights = {'accuracy': 0.35, 'pronunciation': 0.35, 'fluency': 0.3}

        # Apply difficulty bonus/penalty
        difficulty_modifier = {'easy': 1.0, 'medium': 1.05, 'hard': 1.1}
        modifier = difficulty_modifier.get(difficulty, 1.0)

        overall = (
            accuracy * weights['accuracy'] +
            pronunciation * weights['pronunciation'] +
            fluency * weights['fluency']
        ) * modifier

        return min(100, max(0, overall))

    def _format_child_friendly_feedback(self, ai_feedback: str, student_data: Dict) -> str:
        """Format AI feedback to be child-friendly"""
        student_name = student_data.get('name', 'الطالب')

        # Add personal touch and encouragement
        formatted_feedback = f"عزيزي {student_name}،\n\n"
        formatted_feedback += ai_feedback
        formatted_feedback += f"\n\nاستمر في التدرب وستصبح أفضل! 🌟"

        return formatted_feedback

    def _generate_recommendations(self, student_data: Dict, errors: List[Dict],
                                overall_score: float) -> List[str]:
        """Generate specific recommendations based on performance"""
        recommendations = []
        grade_level = student_data.get('grade_level', 1)

        # Error-based recommendations
        error_types = [error.get('type', '') for error in errors]

        if 'missing_diacritics' in error_types:
            recommendations.append("تدرب على قراءة النصوص مع التشكيل لتحسين النطق")

        if 'substitution' in error_types or 'mispronunciation' in error_types:
            recommendations.append("استمع للنطق الصحيح للكلمات الصعبة عدة مرات")

        if len(errors) > 5:
            recommendations.append("اقرأ ببطء أكثر للتركيز على دقة النطق")

        # Score-based recommendations
        if overall_score < 60:
            recommendations.append("تدرب على نصوص أسهل أولاً لبناء الثقة")
            recommendations.append("اطلب المساعدة من المعلم أو الأهل")
        elif overall_score < 80:
            recommendations.append("أعد قراءة النص عدة مرات لتحسين الطلاقة")
        else:
            recommendations.append("جرب نصوصاً أكثر صعوبة لتطوير مهاراتك")

        return recommendations[:4]  # Limit to 4 recommendations

    def _generate_encouragement(self, overall_score: float, student_data: Dict) -> str:
        """Generate encouraging message based on performance"""
        student_name = student_data.get('name', 'الطالب')

        if overall_score >= 90:
            return f"ممتاز يا {student_name}! أداء رائع جداً! 🌟"
        elif overall_score >= 80:
            return f"أحسنت يا {student_name}! تحسن ملحوظ! 👏"
        elif overall_score >= 70:
            return f"جيد يا {student_name}! استمر في التدرب! 💪"
        elif overall_score >= 60:
            return f"لا بأس يا {student_name}، أنت تتحسن! 📈"
        else:
            return f"لا تستسلم يا {student_name}، التدرب سيحسن أداءك! 🎯"

    def _identify_improvement_areas(self, errors: List[Dict], fluency_metrics: Dict,
                                  accuracy_metrics: Dict) -> List[str]:
        """Identify specific areas needing improvement"""
        areas = []

        if accuracy_metrics.get('accuracy_score', 0) < 70:
            areas.append("دقة قراءة الكلمات")

        if fluency_metrics.get('fluency_score', 0) < 70:
            areas.append("سرعة القراءة وطلاقتها")

        error_types = [error.get('type', '') for error in errors]
        if 'missing_diacritics' in error_types:
            areas.append("قراءة التشكيل")

        if any('pronunciation' in error.get('type', '') for error in errors):
            areas.append("وضوح النطق")

        return areas

    def _identify_strengths(self, overall_score: float, pronunciation_score: float,
                          fluency_metrics: Dict) -> List[str]:
        """Identify student's strengths"""
        strengths = []

        if pronunciation_score >= 80:
            strengths.append("النطق الواضح")

        if fluency_metrics.get('fluency_score', 0) >= 80:
            strengths.append("الطلاقة في القراءة")

        if overall_score >= 80:
            strengths.append("الأداء العام الممتاز")

        if fluency_metrics.get('pace_rating') == 'normal':
            strengths.append("السرعة المناسبة في القراءة")

        if not strengths:
            strengths.append("المحاولة والجهد المبذول")

        return strengths

    def _generate_fallback_assessment(self, student_data: Dict, errors: List[Dict]) -> Dict:
        """Generate basic assessment when AI is not available"""
        error_count = len(errors)

        # Simple scoring based on error count
        if error_count == 0:
            overall_score = 95
        elif error_count <= 2:
            overall_score = 85
        elif error_count <= 5:
            overall_score = 70
        else:
            overall_score = max(50, 80 - (error_count * 5))

        return {
            'overall_score': overall_score,
            'pronunciation_score': max(60, 100 - (error_count * 10)),
            'fluency_score': 75,
            'accuracy_score': max(50, 100 - (error_count * 8)),
            'comprehension_score': 75,
            'errors_detected': errors,
            'personalized_feedback': f"تم رصد {error_count} خطأ. استمر في التدرب لتحسين أدائك!",
            'recommendations': ["تدرب على القراءة يومياً", "استمع للنطق الصحيح"],
            'encouragement': "أحسنت! استمر في المحاولة! 💪"
        }
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
            trans_words = self.normalize_arabic_text(transcribed_text).split()
            total_words = len(orig_words)

            if total_words == 0:
                return {'accuracy_score': 0, 'error_rate': 100, 'similarity_score': 0}

            # Calculate text similarity using SequenceMatcher
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, orig_words, trans_words).ratio() * 100

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

            # Calculate accuracy score (0-100) - use similarity as base, adjust for errors
            error_penalty = (weighted_errors / total_words) * 100
            accuracy_score = max(0, similarity - error_penalty)

            # If we have no errors but low similarity, use similarity score
            if len(errors) == 0 and similarity > accuracy_score:
                accuracy_score = similarity

            error_rate = (len(errors) / total_words) * 100

            return {
                'accuracy_score': round(accuracy_score, 2),
                'similarity_score': round(similarity, 2),
                'error_rate': round(error_rate, 2),
                'total_words': total_words,
                'total_errors': len(errors),
                'error_breakdown': error_counts
            }

        except Exception as e:
            logger.error(f"Error calculating accuracy score: {e}")
            return {'accuracy_score': 0, 'error_rate': 100, 'similarity_score': 0}

    def generate_comprehensive_assessment(self, original_text: str, transcribed_text: str,
                                        audio_duration: float, pronunciation_details: Dict = None) -> Dict:
        """Generate comprehensive reading assessment"""
        try:
            # Handle empty transcription
            if not transcribed_text or not transcribed_text.strip():
                logger.warning("Empty transcribed text - generating minimal assessment")
                return {
                    'overall_score': 0,
                    'pronunciation_score': 0,
                    'fluency_score': 0,
                    'accuracy_score': 0,
                    'comprehension_score': 0,
                    'detailed_metrics': {
                        'fluency': {'fluency_score': 0, 'words_per_minute': 0},
                        'accuracy': {'accuracy_score': 0, 'similarity_score': 0},
                        'pronunciation': pronunciation_details or {}
                    },
                    'errors': [],
                    'error_summary': 'لم يتم التعرف على الكلام بوضوح',
                    'feedback': 'لم يتم التعرف على الكلام بشكل واضح. يرجى التأكد من وضوح النطق والتسجيل في مكان هادئ.',
                    'recommendations': [
                        'تحدث بوضوح أكبر',
                        'تأكد من عدم وجود ضوضاء في الخلفية',
                        'اقترب من الميكروفون',
                        'تحدث بصوت أعلى قليلاً'
                    ]
                }

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

            # If pronunciation score is 0 but we have transcription, use accuracy as pronunciation indicator
            if pronunciation_score == 0 and transcribed_text.strip():
                pronunciation_score = accuracy_metrics['accuracy_score']

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
                'pronunciation_score': round(pronunciation_score, 2),
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
                'pronunciation_score': 0,
                'fluency_score': 0,
                'accuracy_score': 0,
                'comprehension_score': 0,
                'error': str(e),
                'feedback': 'حدث خطأ في معالجة التقييم. يرجى المحاولة مرة أخرى.',
                'recommendations': ['يرجى المحاولة مرة أخرى']
            }

    def generate_ai_feedback(self, original_text: str, transcribed_text: str,
                           errors: List[Dict], fluency_metrics: Dict, accuracy_metrics: Dict) -> str:
        """Generate personalized feedback using Gemini AI"""
        if not self.gemini_model:
            return self._generate_basic_feedback(errors, fluency_metrics, accuracy_metrics)

        try:
            # Prepare prompt for Gemini
            prompt = f"""
            You are an Arabic language teacher providing brief feedback on a student's reading.

            Original text: {original_text}
            Student read: {transcribed_text}

            Performance:
            - Fluency: {fluency_metrics.get('fluency_score', 0)}/100
            - Accuracy: {accuracy_metrics.get('accuracy_score', 0)}/100
            - Reading speed: {fluency_metrics.get('wpm', 0)} words/min
            - Errors found: {len(errors)}

            Main errors: {[error.get('expected', '') + ' → ' + error.get('actual', '') for error in errors[:3]]}

            Provide feedback in EXACTLY 4 lines maximum:
            Line 1: One sentence acknowledging their performance
            Line 2: One specific area to improve (based on main errors)
            Line 3: One practical tip for improvement
            Line 4: One encouraging closing statement

            Keep it simple, specific, and encouraging. No more than 4 lines total.
            Write in Arabic only.
            """

            response = self.gemini_model.generate_content(prompt)
            return response.text

        except Exception as e:
            logger.error(f"Error generating AI feedback: {e}")
            return self._generate_basic_feedback(errors, fluency_metrics, accuracy_metrics)

    def _generate_basic_feedback(self, errors: List[Dict], fluency_metrics: Dict, accuracy_metrics: Dict) -> str:
        """Generate basic feedback when AI is not available"""
        accuracy_score = accuracy_metrics.get('accuracy_score', 0)
        fluency_score = fluency_metrics.get('fluency_score', 0)
        error_count = len(errors)

        # Generate short, specific feedback
        if accuracy_score >= 80:
            performance_msg = "أداء جيد في القراءة!"
        elif accuracy_score >= 60:
            performance_msg = "أداء مقبول مع وجود بعض الأخطاء."
        else:
            performance_msg = "تحتاج إلى تحسين دقة النطق."

        if error_count > 0:
            main_errors = [error.get('expected', '') for error in errors[:2]]
            improvement_msg = f"ركز على نطق: {', '.join(main_errors)}"
        else:
            improvement_msg = "حافظ على هذا المستوى الممتاز."

        tip_msg = "استمع للنص واقرأ ببطء أكثر."
        encouragement_msg = "استمر في التدرب وستتحسن! 💪"

        return f"{performance_msg}\n{improvement_msg}\n{tip_msg}\n{encouragement_msg}"

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

    def detect_pronunciation_errors(self, original_text: str, transcribed_text: str,
                                  pronunciation_details: Dict = None) -> List[Dict]:
        """Detect pronunciation errors by comparing original and transcribed text"""
        errors = []

        try:
            if not transcribed_text or not transcribed_text.strip():
                return [{
                    'type': 'no_transcription',
                    'severity': 'high',
                    'message': 'لم يتم التعرف على الكلام',
                    'position': 0
                }]

            # Normalize both texts for comparison
            orig_normalized = self.normalize_arabic_text(original_text)
            trans_normalized = self.normalize_arabic_text(transcribed_text)

            orig_words = orig_normalized.split()
            trans_words = trans_normalized.split()

            # Use pronunciation details if available
            if pronunciation_details and not pronunciation_details.get('error'):
                word_details = pronunciation_details.get('word_details', [])

                for word_detail in word_details:
                    accuracy_score = word_detail.get('accuracy_score', 0)
                    word = word_detail.get('word', '')

                    if accuracy_score < 60:  # Poor pronunciation
                        errors.append({
                            'type': 'mispronunciation',
                            'severity': 'high' if accuracy_score < 40 else 'medium',
                            'word': word,
                            'accuracy_score': accuracy_score,
                            'message': f'نطق غير دقيق للكلمة: {word}',
                            'suggestion': 'حاول النطق بوضوح أكبر'
                        })

            # Compare word-by-word using sequence matching
            from difflib import SequenceMatcher
            matcher = SequenceMatcher(None, orig_words, trans_words)

            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'delete':  # Missing words (omissions)
                    for i in range(i1, i2):
                        errors.append({
                            'type': 'omission',
                            'severity': 'high',
                            'word': orig_words[i],
                            'position': i,
                            'message': f'تم حذف الكلمة: {orig_words[i]}',
                            'suggestion': 'تأكد من قراءة جميع الكلمات'
                        })

                elif tag == 'insert':  # Extra words (insertions)
                    for j in range(j1, j2):
                        errors.append({
                            'type': 'insertion',
                            'severity': 'medium',
                            'word': trans_words[j],
                            'position': j,
                            'message': f'تم إضافة كلمة غير موجودة: {trans_words[j]}',
                            'suggestion': 'اقرأ النص فقط كما هو مكتوب'
                        })

                elif tag == 'replace':  # Substituted words (mispronunciations)
                    for i, j in zip(range(i1, i2), range(j1, j2)):
                        if i < len(orig_words) and j < len(trans_words):
                            errors.append({
                                'type': 'mispronunciation',
                                'severity': 'high',
                                'expected': orig_words[i],
                                'actual': trans_words[j],
                                'position': i,
                                'message': f'نطق خاطئ: قيل "{trans_words[j]}" بدلاً من "{orig_words[i]}"',
                                'suggestion': f'النطق الصحيح هو: {orig_words[i]}'
                            })

            return errors

        except Exception as e:
            logger.error(f"Error detecting pronunciation errors: {e}")
            return []

    # ...existing code...
