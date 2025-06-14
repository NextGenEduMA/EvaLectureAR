#!/usr/bin/env python3
"""
Enhanced Educational Arabic Text Generator for EvaLectureAR
Creates natural narrative texts for Moroccan primary students (grades 1-6)
Using Google Gemini AI with proper storytelling techniques
"""

import os
import sys
import logging
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple
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
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TextSpec:
    """Specification for text generation"""
    grade: int
    difficulty: str
    topic: str
    category: str
    min_words: int
    max_words: int

class NaturalTextGenerator:
    """
    Generates natural Arabic educational texts using storytelling techniques
    """

    def __init__(self):
        """Initialize the text generator"""
        self.api_key = os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("Google API key not found")

        # Configure Google AI
        genai.configure(api_key=self.api_key)

        # Initialize Gemini model with conservative settings for quality
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "temperature": 0.8,  # More creative but controlled
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": 600,
            }
        )

        # Natural story openings for different contexts
        self.story_openings = {
            "character_intro": [
                "في قرية صغيرة قرب جبال الأطلس، كان {name} يعيش مع جدته الحنونة",
                "كان {name} طفلاً نشيطاً يحب اكتشاف العالم من حوله",
                "منذ الصغر، تميز {name} بحبه الشديد لـ",
                "في بيت بسيط وسط مدينة {city}، نشأ {name} وترعرع",
            ],
            "scene_setting": [
                "في صباح يوم جميل من أيام الربيع",
                "عندما حل المساء على المدينة العتيقة",
                "في ذلك اليوم المشمس من شهر",
                "بينما كانت الشمس تغرب خلف التلال",
                "في ساحة صغيرة وسط السوق القديم",
            ],
            "family_story": [
                "في كل مساء، كانت العائلة تجتمع حول المائدة",
                "كان الجد يحكي لأحفاده قصصاً رائعة عن",
                "عندما تعود الأم من عملها، تبدأ طقوس",
                "في البيت المغربي التقليدي، كان الجميع يساهم في",
            ],
            "discovery": [
                "أثناء تجواله في الحديقة، اكتشف الطفل شيئاً مدهشاً",
                "في رحلة إلى الريف، لاحظت البنت الصغيرة",
                "عندما فتح الكتاب لأول مرة، وجد داخله",
                "بينما كان يساعد والده في العمل، تعلم",
            ]
        }

        # Moroccan names for characters
        self.moroccan_names = {
            "boys": ["أحمد", "محمد", "يوسف", "عمر", "خالد", "حسن", "إبراهيم", "عبدالله"],
            "girls": ["فاطمة", "عائشة", "خديجة", "مريم", "زينب", "سعاد", "نادية", "سلمى"]
        }

        # Moroccan cities and places
        self.moroccan_places = ["فاس", "مراكش", "الرباط", "مكناس", "تطوان", "شفشاون", "الصويرة", "أغادير"]

        # Grade-appropriate topics with natural integration
        self.grade_topics = {
            1: {
                "easy": [
                    ("الأسرة", "family", "أفراد الأسرة وحبهم"),
                    ("البيت", "home", "غرف البيت ووظائفها"),
                    ("الحيوانات الأليفة", "pets", "القطط والكلاب"),
                    ("الألوان", "colors", "ألوان الطبيعة الجميلة"),
                    ("الطعام", "food", "وجبات اليوم الثلاث")
                ],
                "medium": [
                    ("المدرسة", "school", "الفصل والأصدقاء"),
                    ("الألعاب", "games", "اللعب مع الأصدقاء"),
                    ("النظافة", "hygiene", "العادات الصحية"),
                    ("الأصدقاء", "friends", "صداقات الطفولة"),
                    ("الطبيعة", "nature", "الشمس والقمر والنجوم")
                ]
            },
            2: {
                "easy": [
                    ("الفصول", "seasons", "تغيرات الطقس"),
                    ("الملابس", "clothes", "ملابس كل موسم"),
                    ("الرياضة", "sports", "أنشطة بدنية مختلفة"),
                    ("الصحة", "health", "زيارة الطبيب"),
                    ("وسائل النقل", "transport", "رحلات يومية")
                ],
                "medium": [
                    ("المغرب", "morocco", "وطننا الجميل"),
                    ("المهن", "professions", "أعمال الكبار"),
                    ("البحر", "sea", "رحلة إلى الشاطئ"),
                    ("الجبال", "mountains", "جمال الأطلس"),
                    ("الأعياد", "holidays", "فرحة العيد")
                ],
                "hard": [
                    ("التراث المغربي", "heritage", "عادات الأجداد"),
                    ("الزراعة", "agriculture", "الفلاح والأرض")
                ]
            },
            3: {
                "easy": [
                    ("الماء", "water", "دورة الماء في الطبيعة"),
                    ("الهواء", "air", "أهمية التنفس"),
                    ("النباتات", "plants", "نمو الأشجار"),
                    ("القراءة", "reading", "عشق الكتب"),
                    ("الأرقام", "numbers", "الحساب في الحياة")
                ],
                "medium": [
                    ("مدن المغرب", "cities", "رحلة عبر المملكة"),
                    ("الحرف التقليدية", "crafts", "صناعات يدوية"),
                    ("الأسواق", "markets", "تجارة تقليدية"),
                    ("الصحراء", "desert", "عجائب الصحراء"),
                    ("الموسيقى", "music", "آلات تراثية")
                ],
                "hard": [
                    ("تاريخ المغرب", "history", "حضارة عريقة"),
                    ("جغرافية المغرب", "geography", "موقع مميز")
                ]
            },
            4: {
                "easy": [
                    ("الفضاء", "space", "النجوم والكواكب"),
                    ("الحشرات", "insects", "عالم النحل"),
                    ("الطيور", "birds", "هجرة الطيور"),
                    ("الغابة", "forest", "أشجار الأرز"),
                    ("الأنهار", "rivers", "مياه جارية")
                ],
                "medium": [
                    ("الصناعة", "industry", "مصانع ومنتجات"),
                    ("السياحة", "tourism", "زوار من العالم"),
                    ("المطبخ المغربي", "cuisine", "نكهات أصيلة"),
                    ("المهرجانات", "festivals", "احتفالات شعبية"),
                    ("الرياضة", "sports", "أبطال مغاربة")
                ],
                "hard": [
                    ("الاختراعات", "inventions", "علماء مسلمون"),
                    ("البيئة", "environment", "حماية الطبيعة")
                ]
            },
            5: {
                "medium": [
                    ("الطاقة", "energy", "مصادر الكهرباء"),
                    ("الاتصالات", "communication", "تطور التكنولوجيا"),
                    ("الطب", "medicine", "علاج الأمراض"),
                    ("الآثار", "monuments", "معالم خالدة"),
                    ("الفنون", "arts", "إبداع وجمال")
                ],
                "hard": [
                    ("العلوم", "science", "تجارب واكتشافات"),
                    ("الحضارة الإسلامية", "civilization", "إنجازات علمية"),
                    ("الاقتصاد", "economy", "تجارة واستثمار"),
                    ("التكنولوجيا", "technology", "عصر رقمي"),
                    ("الأدب", "literature", "شعر وقصص")
                ]
            },
            6: {
                "medium": [
                    ("المستقبل", "future", "مهن جديدة"),
                    ("الإعلام", "media", "أخبار ومعلومات"),
                    ("اللغات", "languages", "تواصل عالمي"),
                    ("الثقافة", "culture", "تنوع حضاري"),
                    ("الفلسفة", "philosophy", "تأملات عميقة")
                ],
                "hard": [
                    ("الديمقراطية", "democracy", "حقوق وواجبات"),
                    ("التنمية المستدامة", "sustainability", "مستقبل أخضر"),
                    ("الذكاء الاصطناعي", "ai", "تقنيات ذكية"),
                    ("التغير المناخي", "climate", "تحديات بيئية"),
                    ("الابتكار", "innovation", "إبداع وحلول")
                ]
            }
        }

    def create_natural_opening(self, topic: str, grade: int) -> str:
        """Create a natural story opening"""
        opening_type = random.choice(["character_intro", "scene_setting", "family_story", "discovery"])
        opening_template = random.choice(self.story_openings[opening_type])

        # Fill in variables
        if "{name}" in opening_template:
            gender = random.choice(["boys", "girls"])
            name = random.choice(self.moroccan_names[gender])
            opening_template = opening_template.replace("{name}", name)

        if "{city}" in opening_template:
            city = random.choice(self.moroccan_places)
            opening_template = opening_template.replace("{city}", city)

        return opening_template

    def generate_text(self, spec: TextSpec) -> Tuple[bool, str, str, int]:
        """Generate a natural educational text"""
        try:
            # Create natural opening
            story_opening = self.create_natural_opening(spec.topic, spec.grade)

            # Define complexity levels by grade
            complexity_guidelines = {
                1: "استخدم جملاً قصيرة وبسيطة جداً. كلمات سهلة ومألوفة فقط. لا تستخدم تراكيب معقدة.",
                2: "استخدم جملاً قصيرة ومتوسطة. كلمات بسيطة مع إدخال بعض المفردات الجديدة تدريجياً.",
                3: "استخدم جملاً متوسطة الطول. مفردات متنوعة مع شرح المفاهيم الجديدة ضمن السياق.",
                4: "استخدم جملاً متوسطة وبعض الجمل الطويلة. مفردات أكثر تنوعاً وتراكيب أكثر تعقيداً.",
                5: "استخدم جملاً متنوعة الطول. مفردات متقدمة وتراكيب معقدة مع ربط الأفكار.",
                6: "استخدم جملاً طويلة ومعقدة. مفردات متقدمة وتراكيب أدبية مع تحليل عميق للمفاهيم."
            }

            vocabulary_level = {
                1: "مفردات أساسية جداً (أمي، أبي، بيت، طعام، لعب)",
                2: "مفردات بسيطة مع إضافات (مدرسة، صديق، حديقة، جميل)",
                3: "مفردات متوسطة (طبيعة، رحلة، اكتشاف، تعلم)",
                4: "مفردات متنوعة (تجربة، مغامرة، ملاحظة، استكشاف)",
                5: "مفردات متقدمة (حضارة، تطور، إنجاز، ابتكار)",
                6: "مفردات أدبية (تأمل، فلسفة، عمق، رؤية)"
            }

            # Create sophisticated prompt
            prompt = f"""
أنت كاتب قصص تعليمية محترف للأطفال المغاربة. اكتب نصاً طبيعياً وممتعاً.

المعطيات:
- الصف: {spec.grade} (عمر {5 + spec.grade}-{6 + spec.grade} سنوات)
- الموضوع: {spec.topic}
- مستوى الصعوبة: {spec.difficulty}
- عدد الكلمات المطلوب: {spec.min_words}-{spec.max_words} كلمة بالضبط
- مستوى التعقيد: {complexity_guidelines[spec.grade]}
- مستوى المفردات: {vocabulary_level[spec.grade]}

ابدأ النص بهذه الجملة أو شيء مشابه لها:
"{story_opening}"

قواعد مهمة جداً:
1. لا تستخدم أبداً: "يا أطفال" أو "هل تعرفون" أو "دعونا نتعلم"
2. لا تسأل أسئلة مباشرة للقارئ
3. اكتب كقصة أو سرد طبيعي فقط
4. اجعل المعلومة التعليمية جزءاً طبيعياً من السرد
5. استخدم أسماء وأماكن مغربية
6. انه النص بشكل طبيعي، لا بسؤال أو دعوة للتفكير
7. استخدم حركات التشكيل في الكلمات المهمة
8. التزم بعدد الكلمات المحدد: {spec.min_words}-{spec.max_words} كلمة

أسلوب السرد المطلوب:
- سرد طبيعي وانسيابي مناسب للصف {spec.grade}
- شخصيات حقيقية ومواقف واقعية
- معلومات مدمجة بذكاء في القصة
- لغة عربية فصحى مبسطة للصف {spec.grade}

اكتب النص فقط:
"""

            # Generate text
            response = self.model.generate_content(prompt)

            if not response or not response.text:
                return False, "", "", 0

            content = response.text.strip()

            # Clean any unwanted elements
            lines = content.split('\n')
            clean_content = []

            for line in lines:
                line = line.strip()
                if line and not line.startswith(('النص:', 'القصة:', 'المطلوب:')):
                    clean_content.append(line)

            content = ' '.join(clean_content)

            # Check for quality issues
            if any(phrase in content for phrase in ['يا أطفال', 'هل تعرفون', 'دعونا', 'أليس كذلك']):
                logger.warning("Generated text contains unwanted phrases")
                return False, "", "", 0

            word_count = len(content.split())

            # Stricter word count validation based on grade
            min_threshold = spec.min_words * 0.9  # Allow 10% tolerance
            max_threshold = spec.max_words * 1.1

            if word_count < min_threshold:
                logger.warning(f"Text too short for grade {spec.grade}: {word_count} words (min: {min_threshold:.0f})")
                return False, "", "", 0

            if word_count > max_threshold:
                logger.warning(f"Text too long for grade {spec.grade}: {word_count} words (max: {max_threshold:.0f})")
                return False, "", "", 0

            title = f"{spec.topic} - الصف {spec.grade}"

            logger.info(f"Generated text for grade {spec.grade}: {word_count} words (target: {spec.min_words}-{spec.max_words})")

            return True, title, content, word_count

        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return False, "", "", 0

    def create_text_specs(self, texts_per_grade: int = 10) -> List[TextSpec]:
        """Create specifications for all texts with progressive difficulty"""
        specs = []

        for grade in range(1, 7):
            grade_data = self.grade_topics[grade]

            # Progressive word count and complexity by grade
            word_ranges = {
                1: (50, 80),    # Very short for beginners
                2: (80, 120),   # Short
                3: (120, 170),  # Medium-short
                4: (170, 220),  # Medium
                5: (220, 280),  # Medium-long
                6: (280, 350)   # Longer and more complex
            }

            min_words, max_words = word_ranges[grade]

            # Collect all topics
            all_topics = []
            for difficulty, topics in grade_data.items():
                for topic_ar, topic_en, description in topics:
                    all_topics.append((topic_ar, topic_en, description, difficulty))

            # Create specs
            for i in range(texts_per_grade):
                topic_ar, topic_en, description, difficulty = all_topics[i % len(all_topics)]

                spec = TextSpec(
                    grade=grade,
                    difficulty=difficulty,
                    topic=topic_ar,
                    category=topic_en,
                    min_words=min_words,
                    max_words=max_words
                )
                specs.append(spec)

        return specs

    def generate_batch_texts(self, texts_per_grade: int = 10, max_retries: int = 3) -> Tuple[int, int]:
        """Generate all texts and save to database"""
        try:
            app = Flask(__name__)
            app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
            app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
            db.init_app(app)

            with app.app_context():
                specs = self.create_text_specs(texts_per_grade)
                total_texts = len(specs)

                logger.info(f"Starting generation of {total_texts} natural texts")

                success_count = 0
                failure_count = 0

                for i, spec in enumerate(specs, 1):
                    logger.info(f"Generating {i}/{total_texts}: Grade {spec.grade} - {spec.topic}")

                    for attempt in range(max_retries):
                        success, title, content, word_count = self.generate_text(spec)

                        if success:
                            text_record = Text(
                                title=title,
                                content=content,
                                grade_level=spec.grade,
                                difficulty_level=spec.difficulty,
                                category=spec.category,
                                word_count=word_count
                            )

                            db.session.add(text_record)
                            db.session.commit()

                            logger.info(f"✅ Saved text ID {text_record.id} ({word_count} words)")
                            success_count += 1
                            break
                        else:
                            if attempt < max_retries - 1:
                                logger.warning(f"Retry {attempt + 1}/{max_retries}")
                                time.sleep(3)
                            else:
                                logger.error(f"❌ Failed after {max_retries} attempts")
                                failure_count += 1

                    time.sleep(1.5)  # Rate limiting

                return success_count, failure_count

        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            return 0, 0

def display_generated_texts():
    """Display generated texts beautifully"""
    try:
        app = Flask(__name__)
        app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
        app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        db.init_app(app)

        with app.app_context():
            print("\n" + "="*90)
            print("📚 النصوص التعليمية المولدة - GENERATED EDUCATIONAL TEXTS")
            print("="*90)

            for grade in range(1, 7):
                texts = Text.query.filter_by(grade_level=grade).order_by(Text.difficulty_level, Text.id).all()

                if texts:
                    print(f"\n🎓 الصف {grade} - Grade {grade} ({len(texts)} نص)")
                    print("-" * 70)

                    for j, text in enumerate(texts, 1):
                        difficulty_emoji = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}
                        emoji = difficulty_emoji.get(text.difficulty_level, "⚪")

                        print(f"\n{emoji} {j}. {text.title}")
                        print(f"    📊 {text.difficulty_level} | 📝 {text.word_count} كلمة | 📚 {text.category}")
                        print(f"    📖 {text.content[:120]}...")

            # Statistics
            total_texts = Text.query.count()
            avg_words = db.session.query(db.func.avg(Text.word_count)).scalar()

            print("\n" + "="*90)
            print("📊 إحصائيات التوليد - GENERATION STATISTICS")
            print("="*90)
            print(f"📚 إجمالي النصوص: {total_texts}")
            print(f"📝 متوسط الكلمات: {avg_words:.1f}")

            for difficulty in ['easy', 'medium', 'hard']:
                count = Text.query.filter_by(difficulty_level=difficulty).count()
                emoji = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}[difficulty]
                print(f"{emoji} {difficulty}: {count} نص")

            print("="*90)

    except Exception as e:
        logger.error(f"Error displaying texts: {e}")

def main():
    """Main execution"""
    print("="*90)
    print("🎓 مولد النصوص التعليمية الطبيعية - Natural Educational Text Generator")
    print("="*90)
    print("📚 توليد نصوص طبيعية للطلاب المغاربة (الصفوف 1-6)")
    print("🎯 الهدف: 10 نصوص لكل صف")
    print("🧠 النموذج: Google Gemini 1.5 Flash")
    print("✨ الأسلوب: سرد طبيعي بدون أسئلة مباشرة")
    print("="*90)

    try:
        print("\n🚀 تهيئة مولد النصوص...")
        generator = NaturalTextGenerator()

        print("📝 بداية توليد النصوص...")
        start_time = time.time()

        success_count, failure_count = generator.generate_batch_texts(texts_per_grade=10)

        generation_time = time.time() - start_time

        print(f"\n🎉 انتهاء التوليد في {generation_time:.1f} ثانية!")
        print(f"✅ تم توليد: {success_count} نص")
        print(f"❌ فشل في توليد: {failure_count} نص")

        if success_count > 0:
            print("\n📖 عرض النصوص المولدة...")
            display_generated_texts()

        return 0 if failure_count == 0 else 1

    except Exception as e:
        logger.error(f"Execution failed: {e}")
        print(f"❌ خطأ: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
