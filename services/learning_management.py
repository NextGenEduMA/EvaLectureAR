import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from sqlalchemy import func, desc
from models.database import db, Student, Assessment, ProgressRecord, Text
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class LearningManagementSystem:
    def __init__(self):
        self.grade_levels = [1, 2, 3, 4, 5, 6]  # Moroccan primary education levels
        self.difficulty_levels = ['easy', 'medium', 'hard']  # سهل، متوسط، صعب
        self.skill_areas = ['pronunciation', 'fluency', 'accuracy', 'comprehension']
    
    def create_student_profile(self, name: str, email: str, grade_level: int = 1, difficulty_level: str = 'easy') -> Dict:
        """Create a new student profile"""
        try:
            # Check if student already exists
            existing_student = Student.query.filter_by(email=email).first()
            if existing_student:
                return {
                    'success': False,
                    'message': 'Student with this email already exists',
                    'student_id': existing_student.id
                }
            
            # Validate grade level
            if grade_level not in self.grade_levels:
                grade_level = 1
            
            # Validate difficulty level
            if difficulty_level not in self.difficulty_levels:
                difficulty_level = 'easy'
            
            # Create new student
            student = Student(
                name=name,
                email=email,
                grade_level=grade_level,
                difficulty_level=difficulty_level
            )
            
            db.session.add(student)
            db.session.commit()
            
            # Initialize progress record
            self._initialize_student_progress(student.id)
            
            return {
                'success': True,
                'message': 'Student profile created successfully',
                'student_id': student.id,
                'student': {
                    'id': student.id,
                    'name': student.name,
                    'email': student.email,
                    'grade_level': student.grade_level,
                    'difficulty_level': student.difficulty_level,
                    'created_at': student.created_at.isoformat()
                }
            }
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error creating student profile: {e}")
            return {
                'success': False,
                'message': f'Error creating student profile: {str(e)}'
            }
    
    def _initialize_student_progress(self, student_id: int):
        """Initialize progress tracking for a new student"""
        try:
            progress_record = ProgressRecord(
                student_id=student_id,
                session_date=datetime.utcnow().date(),
                texts_completed=0,
                average_score=0.0,
                improvement_rate=0.0,
                pronunciation_progress=0.0,
                fluency_progress=0.0,
                accuracy_progress=0.0,
                total_words_read=0,
                total_errors_corrected=0,
                study_time_minutes=0
            )
            
            db.session.add(progress_record)
            db.session.commit()
            
        except Exception as e:
            logger.error(f"Error initializing student progress: {e}")
            db.session.rollback()
    
    def get_student_profile(self, student_id: int) -> Dict:
        """Get complete student profile with progress"""
        try:
            student = Student.query.get(student_id)
            if not student:
                return {'success': False, 'message': 'Student not found'}
            
            # Get latest progress
            latest_progress = ProgressRecord.query.filter_by(
                student_id=student_id
            ).order_by(desc(ProgressRecord.created_at)).first()
            
            # Get assessment statistics
            assessment_stats = self._get_assessment_statistics(student_id)
            
            return {
                'success': True,
                'student': {
                    'id': student.id,
                    'name': student.name,
                    'email': student.email,
                    'grade_level': student.grade_level,
                    'difficulty_level': student.difficulty_level,
                    'created_at': student.created_at.isoformat(),
                    'total_assessments': len(student.assessments),
                    'latest_progress': {
                        'average_score': latest_progress.average_score if latest_progress else 0,
                        'pronunciation_progress': latest_progress.pronunciation_progress if latest_progress else 0,
                        'fluency_progress': latest_progress.fluency_progress if latest_progress else 0,
                        'accuracy_progress': latest_progress.accuracy_progress if latest_progress else 0,
                        'total_words_read': latest_progress.total_words_read if latest_progress else 0,
                        'study_time_minutes': latest_progress.study_time_minutes if latest_progress else 0
                    },
                    'assessment_stats': assessment_stats
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting student profile: {e}")
            return {'success': False, 'message': str(e)}
    
    def _get_assessment_statistics(self, student_id: int) -> Dict:
        """Get detailed assessment statistics for a student"""
        try:
            assessments = Assessment.query.filter_by(
                student_id=student_id,
                status='completed'
            ).all()
            
            if not assessments:
                return {
                    'total_assessments': 0,
                    'average_overall_score': 0,
                    'average_pronunciation_score': 0,
                    'average_fluency_score': 0,
                    'average_accuracy_score': 0,
                    'improvement_trend': 'no_data'
                }
            
            # Calculate averages
            overall_scores = [a.overall_score for a in assessments if a.overall_score]
            pronunciation_scores = [a.pronunciation_score for a in assessments if a.pronunciation_score]
            fluency_scores = [a.fluency_score for a in assessments if a.fluency_score]
            accuracy_scores = [a.accuracy_score for a in assessments if a.accuracy_score]
            
            # Calculate improvement trend
            improvement_trend = self._calculate_improvement_trend(overall_scores)
            
            return {
                'total_assessments': len(assessments),
                'average_overall_score': np.mean(overall_scores) if overall_scores else 0,
                'average_pronunciation_score': np.mean(pronunciation_scores) if pronunciation_scores else 0,
                'average_fluency_score': np.mean(fluency_scores) if fluency_scores else 0,
                'average_accuracy_score': np.mean(accuracy_scores) if accuracy_scores else 0,
                'improvement_trend': improvement_trend,
                'recent_scores': overall_scores[-5:] if len(overall_scores) >= 5 else overall_scores
            }
            
        except Exception as e:
            logger.error(f"Error calculating assessment statistics: {e}")
            return {}
    
    def _calculate_improvement_trend(self, scores: List[float]) -> str:
        """Calculate improvement trend from score history"""
        if len(scores) < 3:
            return 'insufficient_data'
        
        # Calculate trend using linear regression
        x = np.arange(len(scores))
        y = np.array(scores)
        
        # Simple linear regression
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 2:
            return 'improving'
        elif slope < -2:
            return 'declining'
        else:
            return 'stable'
    
    def update_student_progress(self, student_id: int, assessment_result: Dict) -> bool:
        """Update student progress after an assessment"""
        try:
            student = Student.query.get(student_id)
            if not student:
                return False
            
            # Get or create today's progress record
            today = datetime.utcnow().date()
            progress_record = ProgressRecord.query.filter_by(
                student_id=student_id,
                session_date=today
            ).first()
            
            if not progress_record:
                progress_record = ProgressRecord(
                    student_id=student_id,
                    session_date=today
                )
                db.session.add(progress_record)
            
            # Update progress metrics
            progress_record.texts_completed += 1
            
            # Update skill-specific progress
            if 'pronunciation_score' in assessment_result:
                progress_record.pronunciation_progress = self._update_skill_progress(
                    progress_record.pronunciation_progress,
                    assessment_result['pronunciation_score']
                )
            
            if 'fluency_score' in assessment_result:
                progress_record.fluency_progress = self._update_skill_progress(
                    progress_record.fluency_progress,
                    assessment_result['fluency_score']
                )
            
            if 'accuracy_score' in assessment_result:
                progress_record.accuracy_progress = self._update_skill_progress(
                    progress_record.accuracy_progress,
                    assessment_result['accuracy_score']
                )
            
            # Update overall metrics
            if 'overall_score' in assessment_result:
                progress_record.average_score = self._update_average_score(
                    student_id, assessment_result['overall_score']
                )
            
            # Update word count and error corrections
            if 'word_count' in assessment_result:
                progress_record.total_words_read += assessment_result['word_count']
            
            if 'errors' in assessment_result:
                progress_record.total_errors_corrected += len(assessment_result['errors'])
            
            # Update study time (estimate based on audio duration)
            if 'audio_duration' in assessment_result:
                # Estimate study time as 3x audio duration (including feedback review)
                estimated_study_time = int(assessment_result['audio_duration'] / 60 * 3)
                progress_record.study_time_minutes += estimated_study_time
            
            # Calculate improvement rate
            progress_record.improvement_rate = self._calculate_improvement_rate(student_id)
            
            # Check for level advancement
            self._check_level_advancement(student)
            
            db.session.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error updating student progress: {e}")
            db.session.rollback()
            return False
    
    def _update_skill_progress(self, current_progress: float, new_score: float) -> float:
        """Update skill-specific progress using weighted average"""
        if current_progress == 0:
            return new_score
        
        # Use weighted average with more weight on recent performance
        return (current_progress * 0.7) + (new_score * 0.3)
    
    def _update_average_score(self, student_id: int, new_score: float) -> float:
        """Calculate updated average score"""
        try:
            # Get all completed assessments
            assessments = Assessment.query.filter_by(
                student_id=student_id,
                status='completed'
            ).all()
            
            scores = [a.overall_score for a in assessments if a.overall_score is not None]
            scores.append(new_score)
            
            return np.mean(scores)
            
        except Exception as e:
            logger.error(f"Error updating average score: {e}")
            return new_score
    
    def _calculate_improvement_rate(self, student_id: int) -> float:
        """Calculate improvement rate over recent assessments"""
        try:
            # Get recent assessments (last 10)
            recent_assessments = Assessment.query.filter_by(
                student_id=student_id,
                status='completed'
            ).order_by(desc(Assessment.created_at)).limit(10).all()
            
            if len(recent_assessments) < 2:
                return 0.0
            
            scores = [a.overall_score for a in reversed(recent_assessments) if a.overall_score]
            
            if len(scores) < 2:
                return 0.0
            
            # Calculate improvement rate as percentage change
            first_half = scores[:len(scores)//2]
            second_half = scores[len(scores)//2:]
            
            avg_first = np.mean(first_half)
            avg_second = np.mean(second_half)
            
            if avg_first == 0:
                return 0.0
            
            improvement_rate = ((avg_second - avg_first) / avg_first) * 100
            return round(improvement_rate, 2)
            
        except Exception as e:
            logger.error(f"Error calculating improvement rate: {e}")
            return 0.0
    
    def _check_level_advancement(self, student: Student):
        """Check if student should advance to next difficulty level"""
        try:
            # Get recent performance
            recent_assessments = Assessment.query.filter_by(
                student_id=student.id,
                status='completed'
            ).order_by(desc(Assessment.created_at)).limit(5).all()
            
            if len(recent_assessments) < 3:
                return  # Need at least 3 assessments
            
            # Calculate average score of recent assessments
            recent_scores = [a.overall_score for a in recent_assessments if a.overall_score]
            avg_recent_score = np.mean(recent_scores)
            
            # Difficulty advancement criteria
            advancement_thresholds = {
                'easy': 80,      # Need 80+ average to advance to medium
                'medium': 85     # Need 85+ average to advance to hard
            }
            
            current_difficulty = student.difficulty_level
            threshold = advancement_thresholds.get(current_difficulty)
            
            if threshold and avg_recent_score >= threshold:
                # Advance to next difficulty level
                if current_difficulty == 'easy':
                    student.difficulty_level = 'medium'
                elif current_difficulty == 'medium':
                    student.difficulty_level = 'hard'
                
                logger.info(f"Student {student.id} advanced from {current_difficulty} to {student.difficulty_level}")
            
        except Exception as e:
            logger.error(f"Error checking level advancement: {e}")
    
    def get_recommended_texts(self, student_id: int, limit: int = 5) -> List[Dict]:
        """Get recommended texts based on student grade level and difficulty"""
        try:
            student = Student.query.get(student_id)
            if not student:
                return []
            
            # Get texts at appropriate grade level and difficulty
            base_query = Text.query.filter_by(
                grade_level=student.grade_level,
                difficulty_level=student.difficulty_level
            )
            
            # Exclude already completed texts (optional - for variety)
            completed_text_ids = [a.text_id for a in student.assessments]
            if completed_text_ids:
                base_query = base_query.filter(~Text.id.in_(completed_text_ids))
            
            # Get recommended texts
            recommended_texts = base_query.limit(limit).all()
            
            # If not enough texts at current level, include some from adjacent levels
            if len(recommended_texts) < limit:
                # Try same grade level but different difficulty
                for adj_difficulty in self._get_adjacent_difficulties(student.difficulty_level):
                    additional_texts = Text.query.filter_by(
                        grade_level=student.grade_level,
                        difficulty_level=adj_difficulty
                    ).filter(~Text.id.in_(completed_text_ids)).limit(
                        limit - len(recommended_texts)
                    ).all()
                    recommended_texts.extend(additional_texts)
                    
                    if len(recommended_texts) >= limit:
                        break
                
                # If still not enough, try adjacent grade levels
                if len(recommended_texts) < limit:
                    for adj_grade in self._get_adjacent_grades(student.grade_level):
                        additional_texts = Text.query.filter_by(
                            grade_level=adj_grade,
                            difficulty_level=student.difficulty_level
                        ).filter(~Text.id.in_(completed_text_ids)).limit(
                            limit - len(recommended_texts)
                        ).all()
                        recommended_texts.extend(additional_texts)
                        
                        if len(recommended_texts) >= limit:
                            break
            
            return [{
                'id': text.id,
                'title': text.title,
                'content': text.content[:200] + '...' if len(text.content) > 200 else text.content,
                'grade_level': text.grade_level,
                'difficulty_level': text.difficulty_level,
                'category': text.category,
                'word_count': text.word_count,
                'estimated_time_minutes': self._estimate_reading_time(text.word_count, student.grade_level, student.difficulty_level)
            } for text in recommended_texts]
            
        except Exception as e:
            logger.error(f"Error getting recommended texts: {e}")
            return []
    
    def _get_adjacent_grades(self, current_grade: int) -> List[int]:
        """Get adjacent grade levels"""
        adjacent = []
        
        if current_grade > 1:
            adjacent.append(current_grade - 1)
        if current_grade < 6:
            adjacent.append(current_grade + 1)
        
        return adjacent
    
    def _get_adjacent_difficulties(self, current_difficulty: str) -> List[str]:
        """Get adjacent difficulty levels"""
        difficulty_index = self.difficulty_levels.index(current_difficulty)
        adjacent = []
        
        if difficulty_index > 0:
            adjacent.append(self.difficulty_levels[difficulty_index - 1])
        if difficulty_index < len(self.difficulty_levels) - 1:
            adjacent.append(self.difficulty_levels[difficulty_index + 1])
        
        return adjacent
    
    def _estimate_reading_time(self, word_count: int, grade_level: int, difficulty_level: str = 'easy') -> int:
        """Estimate reading time based on word count, grade level, and difficulty"""
        # Base reading speeds by grade level (words per minute)
        base_reading_speeds = {
            1: 30,   # Grade 1: Very slow, learning to read
            2: 45,   # Grade 2: Building fluency
            3: 60,   # Grade 3: Developing fluency
            4: 80,   # Grade 4: Good fluency
            5: 100,  # Grade 5: Strong fluency
            6: 120   # Grade 6: Advanced fluency
        }
        
        # Difficulty multipliers
        difficulty_multipliers = {
            'easy': 1.0,
            'medium': 0.8,  # Slower reading for medium difficulty
            'hard': 0.6     # Much slower for hard difficulty
        }
        
        base_wpm = base_reading_speeds.get(grade_level, 60)
        multiplier = difficulty_multipliers.get(difficulty_level, 1.0)
        adjusted_wpm = base_wpm * multiplier
        
        return max(1, int(word_count / adjusted_wpm))
    
    def generate_progress_report(self, student_id: int, days: int = 30) -> Dict:
        """Generate comprehensive progress report"""
        try:
            student = Student.query.get(student_id)
            if not student:
                return {'success': False, 'message': 'Student not found'}
            
            # Get progress records for the specified period
            start_date = datetime.utcnow().date() - timedelta(days=days)
            progress_records = ProgressRecord.query.filter(
                ProgressRecord.student_id == student_id,
                ProgressRecord.session_date >= start_date
            ).order_by(ProgressRecord.session_date).all()
            
            # Get assessments for the period
            assessments = Assessment.query.filter(
                Assessment.student_id == student_id,
                Assessment.created_at >= datetime.combine(start_date, datetime.min.time()),
                Assessment.status == 'completed'
            ).order_by(Assessment.created_at).all()
            
            # Calculate summary statistics
            total_study_time = sum(p.study_time_minutes for p in progress_records)
            total_texts_completed = sum(p.texts_completed for p in progress_records)
            total_words_read = sum(p.total_words_read for p in progress_records)
            
            # Performance trends
            performance_trend = self._analyze_performance_trend(assessments)
            
            # Skill analysis
            skill_analysis = self._analyze_skill_development(assessments)
            
            return {
                'success': True,
                'report': {
                    'student_info': {
                        'name': student.name,
                        'grade_level': student.grade_level,
                        'difficulty_level': student.difficulty_level,
                        'report_period_days': days
                    },
                    'summary_stats': {
                        'total_study_time_minutes': total_study_time,
                        'total_study_hours': round(total_study_time / 60, 1),
                        'total_texts_completed': total_texts_completed,
                        'total_words_read': total_words_read,
                        'total_assessments': len(assessments),
                        'average_session_time': round(total_study_time / max(1, len(progress_records)), 1)
                    },
                    'performance_trend': performance_trend,
                    'skill_analysis': skill_analysis,
                    'recommendations': self._generate_personalized_recommendations(
                        student, assessments, skill_analysis
                    )
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating progress report: {e}")
            return {'success': False, 'message': str(e)}
    
    def _analyze_performance_trend(self, assessments: List[Assessment]) -> Dict:
        """Analyze performance trend over time"""
        if not assessments:
            return {'trend': 'no_data', 'scores': []}
        
        scores = [a.overall_score for a in assessments if a.overall_score]
        dates = [a.created_at.date() for a in assessments if a.overall_score]
        
        if len(scores) < 2:
            return {'trend': 'insufficient_data', 'scores': scores}
        
        # Calculate trend
        trend = self._calculate_improvement_trend(scores)
        
        # Calculate score statistics
        score_stats = {
            'current_score': scores[-1],
            'best_score': max(scores),
            'average_score': round(np.mean(scores), 2),
            'improvement_from_first': round(scores[-1] - scores[0], 2) if len(scores) > 1 else 0
        }
        
        return {
            'trend': trend,
            'scores': scores,
            'dates': [d.isoformat() for d in dates],
            'statistics': score_stats
        }
    
    def _analyze_skill_development(self, assessments: List[Assessment]) -> Dict:
        """Analyze development in specific skills"""
        skills = {
            'pronunciation': [a.pronunciation_score for a in assessments if a.pronunciation_score],
            'fluency': [a.fluency_score for a in assessments if a.fluency_score],
            'accuracy': [a.accuracy_score for a in assessments if a.accuracy_score]
        }
        
        skill_analysis = {}
        
        for skill, scores in skills.items():
            if scores:
                skill_analysis[skill] = {
                    'current_level': scores[-1],
                    'average': round(np.mean(scores), 2),
                    'best': max(scores),
                    'improvement': round(scores[-1] - scores[0], 2) if len(scores) > 1 else 0,
                    'trend': self._calculate_improvement_trend(scores)
                }
            else:
                skill_analysis[skill] = {
                    'current_level': 0,
                    'average': 0,
                    'best': 0,
                    'improvement': 0,
                    'trend': 'no_data'
                }
        
        return skill_analysis
    
    def _generate_personalized_recommendations(self, student: Student, 
                                            assessments: List[Assessment], 
                                            skill_analysis: Dict) -> List[str]:
        """Generate personalized recommendations based on performance"""
        recommendations = []
        
        # Analyze weakest skills
        skill_scores = {
            skill: data.get('current_level', 0) 
            for skill, data in skill_analysis.items()
        }
        
        weakest_skill = min(skill_scores, key=skill_scores.get)
        weakest_score = skill_scores[weakest_skill]
        
        # Skill-specific recommendations
        if weakest_skill == 'pronunciation' and weakest_score < 70:
            recommendations.extend([
                "Focus on pronunciation practice with audio feedback",
                "Practice difficult Arabic sounds with native speaker recordings",
                "Use the slow-paced reading feature to improve accuracy"
            ])
        
        if weakest_skill == 'fluency' and weakest_score < 70:
            recommendations.extend([
                "Practice reading aloud daily to improve fluency",
                "Start with shorter texts and gradually increase length",
                "Focus on maintaining steady reading pace"
            ])
        
        if weakest_skill == 'accuracy' and weakest_score < 70:
            recommendations.extend([
                "Study texts with complete diacritics (تشكيل)",
                "Practice word recognition exercises",
                "Review common Arabic vocabulary"
            ])
        
        # Grade-specific recommendations
        if student.grade_level <= 2:
            recommendations.append("Practice basic Arabic alphabet and vowel sounds")
            recommendations.append("Focus on simple words and short sentences")
        elif student.grade_level <= 4:
            recommendations.append("Work on reading comprehension with longer texts")
            recommendations.append("Practice reading with proper diacritics")
        else:
            recommendations.append("Challenge yourself with more complex literary texts")
            recommendations.append("Focus on advanced grammar and expression")
        
        # Study frequency recommendations
        if len(assessments) < 5:
            recommendations.append("Aim for regular practice sessions (3-4 times per week)")
        
        return recommendations[:5]  # Limit to top 5 recommendations
