#!/usr/bin/env python3
"""
Direct text generator for educational content using Google Gemini AI
Creates short, grade-appropriate texts for primary school students (ages 6-12)
"""

import os
import sys
import logging
import time
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import google.generativeai as genai
from flask import Flask
from dotenv import load_dotenv

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import database models
from models.database import db, Text

# Load environment variables
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class GenerationResult:
    """Represents a text generation result"""
    content: str
    title: str
    word_count: int
    processing_time: float
    success: bool
    error_message: Optional[str] = None

class DirectTextGenerator:
    """
    Generates short educational texts directly using Google Gemini AI
    Optimized for primary school students (grades 1-6)
    """

    def __init__(self):
        """Initialize the text generator with Google AI API"""
        self.api_key = os.getenv('GOOGLE_API_KEY')
        
        if not self.api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY environment variable")
        
        # Configure Google AI
        genai.configure(api_key=self.api_key)
        
        # LLM configuration - optimized for short educational content
        self.model_name = "gemini-1.5-flash"
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024,
        }
        
        # Initialize Gemini model
        try:
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config,
            )
            # Test the model connection
            test_response = self.model.generate_content("Test", request_options={"timeout": 10})
            logger.info("Direct text generator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            raise

    def create_prompt(self, grade_level: int, difficulty_level: str, topic: str) -> str:
        """
        Create a prompt for short educational text generation
        
        Args:
            grade_level: Student grade level (1-6)
            difficulty_level: Text difficulty (easy, medium, hard)
            topic: The main topic of the text
            
        Returns:
            Formatted prompt for the LLM
        """
        # Map grade levels to appropriate complexity
        complexity_by_grade = {
            1: "كلمات بسيطة جداً وجمل قصيرة",
            2: "كلمات بسيطة وجمل قصيرة",
            3: "كلمات مألوفة وجمل متوسطة الطول",
            4: "مفردات متنوعة وجمل مركبة بسيطة",
            5: "مفردات متنوعة وجمل مركبة",
            6: "مفردات غنية وجمل متنوعة التركيب"
        }
        
        # Adjust line count based on grade level
        min_lines = 2
        max_lines = min(3 + grade_level // 2, 5)  # Gradually increase max lines with grade
        
        # Create prompt with specific instructions for short texts
        prompt = f"""أنت معلم متخصص في إنشاء نصوص تعليمية قصيرة باللغة العربية للأطفال.

المطلوب: نص تعليمي قصير عن {topic}
الصف الدراسي: الصف {grade_level} (عمر {5+grade_level}-{6+grade_level} سنوات)
مستوى الصعوبة: {difficulty_level}

يجب أن يكون النص:
1. قصيراً جداً ({min_lines}-{max_lines} أسطر فقط)
2. مناسباً لمستوى الصف {grade_level}
3. يستخدم {complexity_by_grade[grade_level]}
4. تعليمياً ومفيداً
5. مكتوباً بلغة عربية فصحى سهلة
6. خالياً من الأخطاء اللغوية والإملائية

اكتب النص مباشرة دون عنوان أو مقدمات أو تعليقات إضافية.
"""
        return prompt

    def generate_text(self, grade_level: int, difficulty_level: str, topic: str) -> GenerationResult:
        """
        Generate short educational text based on parameters
        
        Args:
            grade_level: Student grade level (1-6)
            difficulty_level: Text difficulty (easy, medium, hard)
            topic: The main topic of the text
            
        Returns:
            GenerationResult with the generated content
        """
        start_time = time.time()
        
        try:
            # Create prompt
            prompt = self.create_prompt(
                grade_level=grade_level,
                difficulty_level=difficulty_level,
                topic=topic
            )
            
            logger.info(f"Generating text for grade {grade_level}, topic: {topic}")
            
            # Generate content with timeout handling
            response = self.model.generate_content(prompt, request_options={"timeout": 30})
            
            if not response.text:
                return GenerationResult(
                    content="",
                    title="",
                    word_count=0,
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message="لم يتم إنشاء محتوى"
                )
            
            # Clean up the generated text
            content = response.text.strip()
            
            # Count lines and words
            lines = [line for line in content.split('\n') if line.strip()]
            line_count = len(lines)
            word_count = len(content.split())
            
            # Validate line count
            min_lines = 2
            max_lines = min(3 + grade_level // 2, 5)
            
            if line_count < min_lines or line_count > max_lines:
                logger.warning(f"Generated text has {line_count} lines, outside target range {min_lines}-{max_lines}")
                
                # Try to fix by truncating if too long
                if line_count > max_lines:
                    content = '\n'.join(lines[:max_lines])
                    line_count = max_lines
                    word_count = len(content.split())
            
            # Generate a simple title
            title = f"نص عن {topic} - الصف {grade_level}"
            
            processing_time = time.time() - start_time
            
            return GenerationResult(
                content=content,
                title=title,
                word_count=word_count,
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return GenerationResult(
                content="",
                title="",
                word_count=0,
                processing_time=time.time() - start_time,
                success=False,
                error_message=f"خطأ في توليد النص: {str(e)}"
            )

    def save_to_database(self, result: GenerationResult, grade_level: int, 
                        difficulty_level: str, topic: str) -> Optional[int]:
        """
        Save generated text to database
        
        Args:
            result: The generation result
            grade_level: Student grade level
            difficulty_level: Text difficulty
            topic: The main topic
            
        Returns:
            ID of the created text record or None if failed
        """
        try:
            # Initialize Flask app context
            app = Flask(__name__)
            app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///arabic_assessment.db')
            app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
            db.init_app(app)
            
            with app.app_context():
                # Create new text record
                text = Text(
                    title=result.title,
                    content=result.content,
                    grade_level=grade_level,
                    difficulty_level=difficulty_level,
                    category=topic,
                    word_count=result.word_count
                )
                
                db.session.add(text)
                db.session.commit()
                
                logger.info(f"Saved text to database with ID: {text.id}")
                return text.id
                
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
            return None

def generate_batch_texts():
    """Generate multiple texts for each grade level using direct generation"""
    try:
        # Initialize generator
        generator = DirectTextGenerator()
        
        # Get user input for configuration
        texts_per_grade = int(input("\nHow many texts to generate per grade level? (default: 3): ") or "3")
        
        # Topics for text generation - educational topics for primary school
        topics = [
            "القراءة", "الكتابة", "الحساب", "العلوم", "التاريخ", "الجغرافيا", 
            "الحيوانات", "النباتات", "البيئة", "الصحة", "الرياضة", "الفن", 
            "الأسرة", "المدرسة", "الفصول", "الطعام", "الماء", "الهواء"
        ]
        
        # Configuration
        grades = range(1, 7)  # Grades 1-6
        difficulties = {
            1: "easy",
            2: "easy",
            3: "medium",
            4: "medium",
            5: "hard",
            6: "hard"
        }
        
        total_texts = len(grades) * texts_per_grade
        generated_count = 0
        failed_count = 0
        
        print(f"\nWill generate {texts_per_grade} texts for each grade (1-6)")
        print(f"Total texts to generate: {total_texts}")
        
        confirm = input("\nProceed with generation? (y/n): ").lower()
        if confirm != 'y':
            print("Generation cancelled.")
            return 0, 0
        
        print("\nStarting text generation...")
        
        # Generate texts for each grade
        for grade in grades:
            logger.info(f"Generating texts for grade {grade}...")
            
            # Set difficulty based on grade level
            difficulty = difficulties[grade]
            
            for i in range(texts_per_grade):
                # Select random topic
                topic = random.choice(topics)
                
                logger.info(f"Generating text {i+1}/{texts_per_grade} for grade {grade}: {topic} ({difficulty})")
                
                # Generate text
                result = generator.generate_text(
                    grade_level=grade,
                    difficulty_level=difficulty,
                    topic=topic
                )
                
                if result.success:
                    # Save to database
                    text_id = generator.save_to_database(
                        result=result,
                        grade_level=grade,
                        difficulty_level=difficulty,
                        topic=topic
                    )
                    
                    if text_id:
                        print(f"✓ Grade {grade} - {topic}: {result.word_count} words ({result.content.count('\\n')+1} lines)")
                        generated_count += 1
                    else:
                        print(f"✗ Failed to save text to database: {topic}")
                        failed_count += 1
                else:
                    print(f"✗ Failed to generate text: {topic} - {result.error_message}")
                    failed_count += 1
                
                # Add delay between generations
                time.sleep(2)
            
            # Add longer delay between grade levels
            time.sleep(3)
        
        return generated_count, failed_count
                
    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        return 0, 0

def check_database_status():
    """Check if database is properly configured and accessible"""
    try:
        app = Flask(__name__)
        app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///arabic_assessment.db')
        app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        db.init_app(app)
        
        with app.app_context():
            # Try to query the database
            text_count = Text.query.count()
            logger.info(f"Database connection successful. Current text count: {text_count}")
            return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False

def main():
    """Main function to run the text generation"""
    print("=" * 50)
    print("EvaLectureAR - Short Educational Text Generator")
    print("=" * 50)
    
    # Check database connection
    print("\nChecking database connection...")
    if not check_database_status():
        print("❌ Database connection failed. Please check your configuration.")
        return 1
    
    try:
        # Generate texts
        success_count, fail_count = generate_batch_texts()
        
        if success_count > 0:
            print(f"\n✅ Successfully generated {success_count} texts!")
            if fail_count > 0:
                print(f"⚠️ {fail_count} texts failed to generate")
        else:
            print("\n❌ Text generation failed. Please check the logs.")
        
        return 0 if success_count > 0 else 1
        
    except KeyboardInterrupt:
        print("\n\nGeneration interrupted by user.")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
