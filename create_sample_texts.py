#!/usr/bin/env python3
"""
Script to populate the database with sample Arabic texts for testing
"""

from app import app, db
from models.database import Text
from datetime import datetime

def create_sample_texts():
    """Create sample Arabic texts for different grade levels and difficulties"""

    sample_texts = [
        {
            'title': 'البيت الجميل',
            'content': 'هَذَا بَيْتٌ جَمِيلٌ. فِي الْبَيْتِ أُسْرَةٌ سَعِيدَةٌ. الْأَبُ يَعْمَلُ فِي الْمَكْتَبِ. الْأُمُّ تَطْبُخُ الطَّعَامَ. الْأَطْفَالُ يَلْعَبُونَ فِي الْحَدِيقَةِ.',
            'grade_level': 1,
            'difficulty_level': 'easy',
            'category': 'قصص'
        },
        {
            'title': 'المدرسة',
            'content': 'يُحِبُّ أَحْمَدُ الذَّهَابَ إِلَى الْمَدْرَسَةِ. يَلْعَبُ مَعَ أَصْدِقَائِهِ فِي الْفُسْحَةِ وَيَدْرُسُ بِجِدٍّ فِي الْفَصْلِ. مُعَلِّمُهُ يُحِبُّهُ كَثِيرًا لِأَنَّهُ طَالِبٌ مُجْتَهِدٌ.',
            'grade_level': 2,
            'difficulty_level': 'easy',
            'category': 'تعليمية'
        },
        {
            'title': 'فصل الربيع',
            'content': 'فِي فَصْلِ الرَّبِيعِ تَتَفَتَّحُ الْأَزْهَارُ وَتَخْضَرُّ الْأَشْجَارُ. الطَّقْسُ يَصْبِحُ مُعْتَدِلًا وَالنَّاسُ يَخْرُجُونَ لِلتَّنَزُّهِ فِي الْحَدَائِقِ. الطُّيُورُ تُغَرِّدُ بِأَلْحَانٍ جَمِيلَةٍ.',
            'grade_level': 2,
            'difficulty_level': 'medium',
            'category': 'طبيعة'
        },
        {
            'title': 'أهمية الماء',
            'content': 'الْمَاءُ مُهِمٌّ لِلْحَيَاةِ عَلَى الْأَرْضِ. نَشْرَبُ الْمَاءَ كُلَّ يَوْمٍ وَنَسْتَخْدِمُهُ فِي الطَّبْخِ وَالتَّنْظِيفِ. النَّبَاتَاتُ تَحْتَاجُ إِلَى الْمَاءِ لِتَنْمُوَ. يَجِبُ أَنْ نُحَافِظَ عَلَى الْمَاءِ وَلَا نُبَذِّرَهُ.',
            'grade_level': 3,
            'difficulty_level': 'easy',
            'category': 'علوم'
        },
        {
            'title': 'الصداقة',
            'content': 'الصَّدَاقَةُ كَنْزٌ ثَمِينٌ لَا يُقَدَّرُ بِثَمَنٍ. الصَّدِيقُ الْوَفِيُّ يَقِفُ بِجَانِبِكَ فِي السَّرَّاءِ وَالضَّرَّاءِ وَيُسَاعِدُكَ فِي الْمُلِمَّاتِ. يَجِبُ أَنْ نَخْتَارَ أَصْدِقَاءَنَا بِعِنَايَةٍ وَنَكُونَ أَوْفِيَاءَ لَهُمْ.',
            'grade_level': 3,
            'difficulty_level': 'medium',
            'category': 'قيم'
        },
        {
            'title': 'القراءة غذاء العقل',
            'content': 'الْقِرَاءَةُ غِذَاءُ الْعَقْلِ وَالرُّوحِ. تُوَسِّعُ آفَاقَنَا وَتُنَمِّي مَعْرِفَتَنَا وَتُطَوِّرُ قُدْرَاتِنَا اللُّغَوِيَّةَ. مَنْ يَقْرَأُ كَثِيرًا يَكْتَسِبُ ثَقَافَةً وَاسِعَةً وَيُصْبِحُ أَكْثَرَ حِكْمَةً وَفَهْمًا لِلْعَالَمِ مِنْ حَوْلِهِ.',
            'grade_level': 4,
            'difficulty_level': 'medium',
            'category': 'تعليمية'
        },
        {
            'title': 'العلم نور',
            'content': 'الْعِلْمُ نُورٌ يُضِيءُ طَرِيقَ الْحَيَاةِ وَيُرْشِدُنَا إِلَى الصَّوَابِ. مَنْ طَلَبَ الْعِلْمَ مِنَ الْمَهْدِ إِلَى اللَّحْدِ وَجَدَ السَّعَادَةَ وَالنَّجَاحَ فِي دُنْيَاهُ وَآخِرَتِهِ. الْعِلْمُ يَرْفَعُ مَقَامَ الْإِنْسَانِ وَيَجْعَلُهُ قَادِرًا عَلَى خِدْمَةِ وَطَنِهِ وَأُمَّتِهِ.',
            'grade_level': 5,
            'difficulty_level': 'hard',
            'category': 'حكمة'
        },
        {
            'title': 'حب الوطن',
            'content': 'حُبُّ الْوَطَنِ مِنَ الْإِيمَانِ. الْوَطَنُ هُوَ الْأَرْضُ الَّتِي وُلِدْنَا عَلَيْهَا وَنَشَأْنَا فِي أَحْضَانِهَا. يَجِبُ عَلَيْنَا أَنْ نُحِبَّ وَطَنَنَا وَنُحَافِظَ عَلَيْهِ وَنَعْمَلَ مِنْ أَجْلِ تَقَدُّمِهِ وَازْدِهَارِهِ. الْوَطَنُ يَحْتَاجُ إِلَى أَبْنَائِهِ الْمُخْلِصِينَ الَّذِينَ يَعْمَلُونَ بِجِدٍّ وَإِخْلَاصٍ.',
            'grade_level': 6,
            'difficulty_level': 'hard',
            'category': 'وطنية'
        }
    ]

    with app.app_context():
        try:
            # Check if texts already exist
            existing_count = Text.query.count()
            if existing_count > 0:
                print(f"📚 Found {existing_count} existing texts in database")
                return True

            # Create sample texts
            for text_data in sample_texts:
                text = Text(
                    title=text_data['title'],
                    content=text_data['content'],
                    content_with_diacritics=text_data['content'],  # Same content with diacritics
                    grade_level=text_data['grade_level'],
                    difficulty_level=text_data['difficulty_level'],
                    category=text_data['category'],
                    word_count=len(text_data['content'].split()),
                    created_at=datetime.utcnow()
                )

                db.session.add(text)

            db.session.commit()

            print(f"✅ Successfully created {len(sample_texts)} sample texts!")

            # Print summary
            for level in [1, 2, 3, 4, 5, 6]:
                count = Text.query.filter_by(grade_level=level).count()
                print(f"📖 Grade {level}: {count} texts")

            return True

        except Exception as e:
            db.session.rollback()
            print(f"❌ Error creating sample texts: {e}")
            return False

if __name__ == "__main__":
    print("📝 Creating sample Arabic texts...")
    success = create_sample_texts()
    if success:
        print("🎉 Sample texts creation completed!")
    else:
        print("💥 Sample texts creation failed!")
