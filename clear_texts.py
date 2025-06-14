#!/usr/bin/env python3
"""
PostgreSQL database clearing script for EvaLectureAR
Clears all texts and related data from PostgreSQL database
"""

import os
import sys
import logging
from pathlib import Path
from flask import Flask

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import database models
from models.database import db, Text, Student, Assessment, PronunciationError, ProgressRecord, AudioFeedback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clear_postgres_texts():
    """Clear all texts from PostgreSQL database"""
    try:
        # Initialize Flask app context
        app = Flask(__name__)

        # Use PostgreSQL connection from environment
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            print("❌ DATABASE_URL environment variable not set!")
            print("Please set your PostgreSQL connection string:")
            print("export DATABASE_URL='postgresql://username:password@localhost/dbname'")
            return False

        app.config['SQLALCHEMY_DATABASE_URI'] = database_url
        app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        db.init_app(app)

        with app.app_context():
            print("🔍 Checking PostgreSQL database...")

            # Count existing records
            text_count = Text.query.count()
            student_count = Student.query.count()
            assessment_count = Assessment.query.count()
            error_count = PronunciationError.query.count()
            progress_count = ProgressRecord.query.count()
            audio_feedback_count = AudioFeedback.query.count()

            total_records = text_count + student_count + assessment_count + error_count + progress_count + audio_feedback_count

            print(f"📊 Database summary:")
            print(f"   - Texts: {text_count}")
            print(f"   - Students: {student_count}")
            print(f"   - Assessments: {assessment_count}")
            print(f"   - Pronunciation Errors: {error_count}")
            print(f"   - Progress Records: {progress_count}")
            print(f"   - Audio Feedback: {audio_feedback_count}")
            print(f"   - Total Records: {total_records}")

            if total_records == 0:
                print("✅ Database is already empty.")
                return True

            # Show clearing options
            print("\nClearing options:")
            print("1. Clear ALL texts only")
            print("2. Clear ALL database records (texts, students, assessments, etc.)")
            print("3. Clear texts by grade level")
            print("4. Clear texts by difficulty level")
            print("5. Cancel")

            choice = input("\nEnter your choice (1-5): ").strip()

            if choice == '1':
                # Clear only texts
                if text_count == 0:
                    print("✅ No texts to clear.")
                    return True

                confirm = input(f"⚠️  Are you sure you want to delete ALL {text_count} texts? (y/n): ").lower()
                if confirm == 'y':
                    # Delete all texts
                    deleted_count = Text.query.delete()
                    db.session.commit()
                    print(f"✅ Successfully deleted {deleted_count} texts.")
                    return True
                else:
                    print("❌ Operation cancelled.")
                    return False

            elif choice == '2':
                # Clear all records
                confirm = input(f"⚠️  Are you sure you want to delete ALL {total_records} records from the database? (y/n): ").lower()
                if confirm == 'y':
                    print("🗑️  Clearing all database records...")

                    # Clear tables in order (respecting foreign key constraints)
                    audio_deleted = AudioFeedback.query.delete()
                    errors_deleted = PronunciationError.query.delete()
                    assessments_deleted = Assessment.query.delete()
                    progress_deleted = ProgressRecord.query.delete()
                    texts_deleted = Text.query.delete()
                    students_deleted = Student.query.delete()

                    db.session.commit()

                    print(f"✅ Successfully cleared all records:")
                    print(f"   - Audio Feedback: {audio_deleted}")
                    print(f"   - Pronunciation Errors: {errors_deleted}")
                    print(f"   - Assessments: {assessments_deleted}")
                    print(f"   - Progress Records: {progress_deleted}")
                    print(f"   - Texts: {texts_deleted}")
                    print(f"   - Students: {students_deleted}")

                    return True
                else:
                    print("❌ Operation cancelled.")
                    return False

            elif choice == '3':
                # Clear by grade level
                if text_count == 0:
                    print("✅ No texts to clear.")
                    return True

                grade = input("Enter grade level to clear (1-6): ").strip()
                try:
                    grade = int(grade)
                    if grade < 1 or grade > 6:
                        print("❌ Invalid grade level. Must be between 1 and 6.")
                        return False

                    # Count texts for this grade
                    grade_texts = Text.query.filter_by(grade_level=grade)
                    grade_count = grade_texts.count()

                    if grade_count == 0:
                        print(f"✅ No texts found for grade {grade}.")
                        return True

                    confirm = input(f"⚠️  Delete {grade_count} texts for grade {grade}? (y/n): ").lower()
                    if confirm == 'y':
                        deleted_count = grade_texts.delete()
                        db.session.commit()
                        print(f"✅ Successfully deleted {deleted_count} texts for grade {grade}.")
                        return True
                    else:
                        print("❌ Operation cancelled.")
                        return False

                except ValueError:
                    print("❌ Invalid input. Please enter a number.")
                    return False

            elif choice == '4':
                # Clear by difficulty level
                if text_count == 0:
                    print("✅ No texts to clear.")
                    return True

                print("Available difficulty levels: easy, medium, hard")
                difficulty = input("Enter difficulty level to clear: ").strip().lower()

                if difficulty not in ['easy', 'medium', 'hard']:
                    print("❌ Invalid difficulty level. Must be easy, medium, or hard.")
                    return False

                # Count texts for this difficulty
                diff_texts = Text.query.filter_by(difficulty_level=difficulty)
                diff_count = diff_texts.count()

                if diff_count == 0:
                    print(f"✅ No texts found with difficulty '{difficulty}'.")
                    return True

                confirm = input(f"⚠️  Delete {diff_count} texts with difficulty '{difficulty}'? (y/n): ").lower()
                if confirm == 'y':
                    deleted_count = diff_texts.delete()
                    db.session.commit()
                    print(f"✅ Successfully deleted {deleted_count} texts with difficulty '{difficulty}'.")
                    return True
                else:
                    print("❌ Operation cancelled.")
                    return False

            else:
                print("❌ Operation cancelled.")
                return False

    except Exception as e:
        logger.error(f"Error clearing PostgreSQL database: {e}")
        print(f"❌ Error: {e}")
        return False

def main():
    """Main function"""
    print("=" * 60)
    print("EvaLectureAR - PostgreSQL Database Text Cleanup")
    print("=" * 60)

    success = clear_postgres_texts()

    if success:
        print("\n✅ Database cleanup completed successfully.")
    else:
        print("\n❌ Database cleanup failed or was cancelled.")

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
