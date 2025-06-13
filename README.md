# EvaLectureAR - Intelligent Arabic Reading Assessment System
## Academic Project Report

---

## Abstract

EvaLectureAR is an advanced AI-powered educational platform designed to assess and improve Arabic reading skills for primary school students (grades 1-6) in the Moroccan educational system. The system integrates cutting-edge technologies including Automatic Speech Recognition (ASR), Natural Language Processing (NLP), Retrieval-Augmented Generation (RAG), and real-time audio processing to provide comprehensive reading assessment and personalized feedback.

The platform addresses critical challenges in Arabic language education by providing automated, objective, and detailed assessment of pronunciation, fluency, accuracy, and comprehension. It leverages cloud-based AI services, machine learning models, and advanced text processing techniques to deliver real-time feedback and generate adaptive learning recommendations.

---

## 1. Introduction

### 1.1 Problem Statement

Traditional Arabic reading assessment methods in primary education face several limitations:
- **Subjectivity**: Manual assessment by teachers introduces inconsistency
- **Scalability**: Individual assessment is time-consuming and resource-intensive
- **Real-time Feedback**: Delayed feedback reduces learning effectiveness
- **Detailed Analysis**: Limited granular analysis of specific pronunciation errors
- **Adaptive Learning**: Lack of personalized learning paths based on individual progress

### 1.2 Objectives

The primary objectives of EvaLectureAR are:
1. **Automated Assessment**: Provide objective, consistent evaluation of Arabic reading skills
2. **Real-time Processing**: Deliver immediate feedback during reading sessions
3. **Comprehensive Analysis**: Evaluate pronunciation, fluency, accuracy, and comprehension
4. **Personalized Learning**: Generate adaptive recommendations based on individual performance
5. **Knowledge Integration**: Utilize RAG technology to enhance educational content generation
6. **Progress Tracking**: Monitor student development over time with detailed analytics

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           EvaLectureAR System Architecture                       │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Frontend  │    │  Mobile Client  │    │   Admin Panel   │
│    (HTML/JS)    │    │   (Optional)    │    │   (Teachers)    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
┌──────────────────────────────────────────────────────────────────────────────────┐
│                               API Gateway Layer                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                 │
│  │   Flask Routes  │  │  WebSocket API  │  │   REST API      │                 │
│  │   (HTTP/HTTPS)  │  │   (Real-time)   │  │  (CRUD Ops)     │                 │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                 │
└──────────────────────────┬───────────────────────────────────────────────────────┘
                           │
┌──────────────────────────────────────────────────────────────────────────────────┐
│                            Core Service Layer                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                 │
│  │Speech Recognition│  │  AI Assessment  │  │Feedback Generator│                │
│  │   - Wav2Vec2     │  │   - Gemini AI   │  │  - Azure TTS    │                 │
│  │   - Azure STT    │  │   - NLP Engine  │  │  - Audio Synth  │                 │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                 │
│                                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                 │
│  │Learning Mgmt Sys│  │  RAG Pipeline   │  │Realtime Processor│                │
│  │- Progress Track │  │ - Vector Store  │  │- WebSocket Mgmt │                 │
│  │- Recommendations│  │ - Doc Embeddings│  │- Stream Process │                 │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                 │
└──────────────────────────┬───────────────────────────────────────────────────────┘
                           │
┌──────────────────────────────────────────────────────────────────────────────────┐
│                            Data Layer                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                 │
│  │  PostgreSQL DB  │  │   Redis Cache   │  │   File Storage  │                 │
│  │  - User Data    │  │  - Sessions     │  │  - Audio Files  │                 │
│  │  - Assessments  │  │  - Temp Data    │  │  - Documents    │                 │
│  │  - Progress     │  │  - Real-time    │  │  - Models       │                 │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                 │
│                                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐                                       │
│  │  Vector Store   │  │   ML Models     │                                       │
│  │  - FAISS Index  │  │  - Embeddings   │                                       │
│  │  - Embeddings   │  │  - Checkpoints  │                                       │
│  └─────────────────┘  └─────────────────┘                                       │
└──────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────┐
│                        External Services Integration                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                 │
│  │  Google AI API  │  │Azure Cognitive  │  │  Hugging Face   │                 │
│  │  - Gemini LLM   │  │   Services      │  │   - Models      │                 │
│  │  - Embeddings   │  │  - Speech STT   │  │   - Transformers│                 │
│  │               │  │  - TTS          │  │                 │                 │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                 │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Architecture

#### 2.2.1 Frontend Layer
- **Web Interface**: Responsive HTML5/CSS3/JavaScript interface with RTL support for Arabic
- **Real-time Communication**: WebSocket implementation for live audio streaming
- **Audio Recording**: Web Audio API integration for high-quality audio capture
- **User Interface**: Modern, child-friendly design with visual feedback mechanisms

#### 2.2.2 API Layer
- **Flask Framework**: RESTful API endpoints for all system operations
- **Socket.IO**: Real-time bidirectional communication for live assessment
- **CORS Support**: Cross-origin resource sharing for web client integration
- **Rate Limiting**: Request throttling and abuse prevention mechanisms

#### 2.2.3 Core Services
- **Speech Recognition Service**: Multi-model ASR with Wav2Vec2 and Azure Speech Services
- **AI Assessment Engine**: Comprehensive evaluation using Google Gemini AI
- **Feedback Generation Service**: Automated feedback creation with Azure TTS
- **Learning Management System**: Progress tracking and adaptive learning algorithms
- **RAG Pipeline**: Knowledge retrieval and content generation system

### 2.3 Data Flow Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Audio     │───▶│  Speech     │───▶│   Text      │───▶│ Assessment  │
│  Recording  │    │Recognition  │    │Transcription│    │  Analysis   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                 │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐           │
│  Feedback   │◀───│   Report    │◀───│   Score     │◀──────────┘
│ Generation  │    │Generation   │    │Calculation  │
└─────────────┘    └─────────────┘    └─────────────┘
       │
       ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Audio     │    │  Database   │    │  Progress   │
│  Synthesis  │    │   Storage   │    │  Tracking   │
└─────────────┘    └─────────────┘    └─────────────┘
```

---

## 3. RAG (Retrieval-Augmented Generation) System

### 3.1 RAG Architecture Overview

The RAG system enhances the educational platform by providing intelligent content generation and question-answering capabilities based on educational materials.

```
┌──────────────────────────────────────────────────────────────────────┐
│                      RAG Pipeline Architecture                       │
└──────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF Documents │───▶│   Document      │───▶│    Text         │
│   (Educational  │    │   Processing    │    │  Extraction     │
│    Materials)   │    │   (PyMuPDF)     │    │   & Cleaning    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐           │
│   Chunking      │◀───│   Text          │◀──────────┘
│   Strategy      │    │ Preprocessing   │
│ (Semantic Split)│    │ (Arabic NLP)    │
└─────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Embedding     │───▶│   Vector        │───▶│   FAISS         │
│   Generation    │    │ Representation  │    │   Index         │
│ (Sentence-BERT) │    │  (768-dim)      │    │  (Similarity)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                              ┌─────────────────┐     │
                              │   User Query    │     │
                              │  (Arabic Text)  │     │
                              └─────────────────┘     │
                                       │              │
                                       ▼              ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Response      │◀───│   LLM           │◀───│   Retrieval     │
│  Generation     │    │  Generation     │    │   Process       │
│  (Gemini AI)    │    │  (Context +     │    │ (Top-K Docs)    │
└─────────────────┘    │   Query)        │    └─────────────────┘
                       └─────────────────┘
```

### 3.2 RAG Components

#### 3.2.1 Document Processing Pipeline
- **PDF Extraction**: Utilizes PyMuPDF for accurate Arabic text extraction
- **Text Cleaning**: Removes formatting artifacts and normalizes Arabic text
- **Chunking Strategy**: Semantic chunking preserving context and meaning
- **Metadata Enrichment**: Adds document source, page numbers, and content type

#### 3.2.2 Embedding System
- **Model**: Sentence-Transformers with Arabic language support
- **Dimension**: 768-dimensional dense vectors for semantic representation
- **Storage**: Efficient embedding storage with metadata mapping
- **Indexing**: FAISS-based similarity search with configurable distance metrics

#### 3.2.3 Vector Store
- **Backend**: FAISS (Facebook AI Similarity Search)
- **Index Types**: Support for flat, IVF, and HNSW indices
- **Scalability**: Designed for growing document collections
- **Persistence**: Serialized storage for quick system initialization

#### 3.2.4 Retrieval Mechanism
- **Similarity Search**: Cosine similarity for semantic matching
- **Ranking**: Relevance scoring with confidence thresholds
- **Context Window**: Configurable context size for LLM input
- **Filtering**: Grade-level and difficulty-based content filtering

#### 3.2.5 Generation Engine
- **LLM**: Google Gemini 1.5 Flash for fast, accurate responses
- **Prompt Engineering**: Optimized prompts for educational content
- **Context Integration**: Seamless integration of retrieved context
- **Output Formatting**: Structured responses with source attribution

### 3.3 RAG Use Cases

1. **Educational Content Generation**: Create grade-appropriate reading materials
2. **Question Answering**: Answer student queries about lesson content
3. **Assessment Question Creation**: Generate comprehension questions
4. **Adaptive Learning**: Personalize content based on student performance
5. **Teacher Resources**: Provide additional explanations and materials

---

## 4. Core Technologies and Services

### 4.1 Speech Recognition System

#### 4.1.1 Multi-Model Architecture
The system implements a dual-model approach for robust Arabic speech recognition:

**Primary Model: Wav2Vec2**
- **Model**: Facebook's wav2vec2-large-xlsr-53
- **Language Support**: Multilingual model with Arabic fine-tuning
- **Processing**: Local inference for privacy and speed
- **Accuracy**: Optimized for children's voice patterns

**Secondary Model: Azure Speech Services**
- **Service**: Microsoft Azure Cognitive Services
- **Language**: Arabic (Saudi Arabia) - ar-SA
- **Features**: Real-time streaming, noise reduction
- **Fallback**: Primary backup when Wav2Vec2 unavailable

#### 4.1.2 Audio Processing Pipeline
```python
# Audio Preprocessing Steps
1. Format Conversion (to WAV 16kHz)
2. Noise Reduction (Spectral Subtraction)
3. Normalization (Dynamic Range Compression)
4. Silence Removal (Adaptive Thresholding)
5. Feature Extraction (MFCC, Spectral Features)
```

### 4.2 AI Assessment Engine

#### 4.2.1 Assessment Dimensions
The system evaluates four key aspects of reading:

**Pronunciation Assessment**
- Phonetic accuracy analysis
- Diacritic mark detection
- Common mispronunciation patterns
- Voice quality evaluation

**Fluency Assessment**
- Reading speed calculation (WPM)
- Pause pattern analysis
- Rhythm and intonation evaluation
- Stuttering detection

**Accuracy Assessment**
- Word-level accuracy scoring
- Error type classification
- Omission and substitution detection
- Context-aware corrections

**Comprehension Assessment**
- Content understanding evaluation
- Question-answer generation
- Semantic similarity analysis
- Knowledge retention testing

#### 4.2.2 AI Models Integration
- **Primary AI**: Google Gemini 1.5 Flash
- **Capabilities**: Natural language understanding, multilingual support
- **Tasks**: Error analysis, feedback generation, comprehension assessment
- **Configuration**: Temperature: 0.3, Top-p: 0.8, Max tokens: 1024

### 4.3 Database Schema

```sql
-- Core Tables Structure
CREATE TABLE students (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(120) UNIQUE NOT NULL,
    grade_level INTEGER DEFAULT 1,
    difficulty_level VARCHAR(20) DEFAULT 'easy',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE texts (
    id SERIAL PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    content TEXT NOT NULL,
    content_with_diacritics TEXT,
    grade_level INTEGER DEFAULT 1,
    difficulty_level VARCHAR(20) DEFAULT 'easy',
    category VARCHAR(50),
    word_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE assessments (
    id SERIAL PRIMARY KEY,
    student_id INTEGER REFERENCES students(id),
    text_id INTEGER REFERENCES texts(id),
    transcribed_text TEXT,
    overall_score FLOAT,
    pronunciation_score FLOAT,
    fluency_score FLOAT,
    accuracy_score FLOAT,
    comprehension_score FLOAT,
    errors_detected JSONB,
    feedback_text TEXT,
    recommendations JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 5. API Documentation

### 5.1 Assessment Endpoints

#### Create Assessment
```http
POST /api/assessments
Content-Type: multipart/form-data

Parameters:
- audio: Audio file (WAV, MP3, M4A, OGG, FLAC)
- student_id: Student identifier
- text_id: Text identifier for assessment

Response:
{
    "success": true,
    "assessment_id": 123,
    "message": "Assessment created successfully"
}
```

#### Get Assessment Results
```http
GET /api/assessments/{assessment_id}/results

Response:
{
    "success": true,
    "assessment": {
        "overall_score": 85.5,
        "pronunciation_score": 88.0,
        "fluency_score": 82.0,
        "accuracy_score": 90.0,
        "comprehension_score": 83.0,
        "errors_detected": [...],
        "feedback_text": "...",
        "recommendations": [...]
    }
}
```

### 5.2 RAG System Endpoints

#### Upload Knowledge Base
```http
POST /api/rag/upload-pdf
Content-Type: multipart/form-data

Parameters:
- pdf: PDF document for knowledge base

Response:
{
    "success": true,
    "message": "Successfully processed document",
    "stats": {
        "total_documents": 150,
        "status": "ready"
    }
}
```

#### Generate Educational Content
```http
POST /api/rag/generate-text
Content-Type: application/json

Body:
{
    "grade_level": 3,
    "difficulty_level": "medium",
    "topic": "القراءة",
    "prompt": "أنشئ نصاً تعليمياً عن أهمية القراءة"
}

Response:
{
    "success": true,
    "generated_text": "...",
    "sources": [...],
    "confidence": 0.92
}
```

### 5.3 WebSocket Events

#### Real-time Assessment
```javascript
// Client Events
socket.emit('start_recording', {
    student_id: 123,
    text_id: 456
});

socket.emit('audio_chunk', {
    audio_data: base64_encoded_audio,
    chunk_index: 1
});

// Server Events
socket.on('assessment_complete', (data) => {
    console.log('Assessment results:', data.results);
});

socket.on('realtime_feedback', (data) => {
    console.log('Live feedback:', data.feedback);
});
```

---

## 6. Installation and Setup

### 6.1 Prerequisites

**System Requirements:**
- Python 3.10 or higher
- PostgreSQL 13+
- Redis 6.0+
- FFmpeg (for audio processing)
- Git

**API Keys Required:**
- Google AI API Key (Gemini)
- Azure Speech Services Key
- Azure Speech Region

### 6.2 Local Development Setup

#### Step 1: Clone Repository
```bash
git clone https://github.com/your-username/EvaLectureAR.git
cd EvaLectureAR
```

#### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 4: Environment Configuration
Create `.env` file in project root:
```env
# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/arabic_assessment

# API Keys
GOOGLE_API_KEY=your_google_ai_api_key
AZURE_SPEECH_KEY=your_azure_speech_key
AZURE_SPEECH_REGION=your_azure_region

# Flask Configuration
SECRET_KEY=your_secret_key_here
FLASK_ENV=development

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
```

#### Step 5: Database Setup
```bash
# Create database
createdb arabic_assessment

# Initialize database
flask db init
flask db migrate -m "Initial migration"
flask db upgrade
```

#### Step 6: Initialize RAG System
```bash
# Place your PDF documents in rag_documents/pdfs/
python setup_rag.py
```

#### Step 7: Run Application
```bash
python app.py
```

The application will be available at `http://localhost:5000`

### 6.3 Docker Deployment

#### Step 1: Build and Run with Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

#### Step 2: Access Application
- **Web Interface**: http://localhost:5000
- **Database**: localhost:5433
- **Redis**: localhost:6380

### 6.4 Production Deployment

#### Environment Variables for Production
```env
FLASK_ENV=production
DATABASE_URL=postgresql://user:pass@prod-db:5432/arabic_assessment
REDIS_URL=redis://prod-redis:6379/0
SECRET_KEY=production_secret_key
```

#### Security Considerations
- Use HTTPS in production
- Configure proper firewall rules
- Enable database encryption
- Implement rate limiting
- Set up monitoring and logging

---

## 7. Usage Guide

### 7.1 Student Registration

1. Navigate to the main page
2. Fill in student information:
   - Full Name (in Arabic)
   - Email Address
   - Grade Level (1-6)
   - Difficulty Preference

### 7.2 Reading Assessment Process

#### Step 1: Text Selection
- System recommends appropriate texts based on grade level
- Teachers can upload custom texts
- Texts include diacritical marks for pronunciation guidance

#### Step 2: Audio Recording
- Click "Start Recording" button
- Read the displayed text aloud
- Real-time visual feedback during recording
- Click "Stop Recording" when finished

#### Step 3: Assessment Processing
- Automatic speech recognition
- AI-powered error analysis
- Comprehensive scoring calculation
- Feedback generation

#### Step 4: Results Review
- Detailed score breakdown
- Specific error identification
- Personalized improvement recommendations
- Audio feedback for corrections

### 7.3 Teacher Dashboard

#### Progress Monitoring
- Individual student progress tracking
- Class-wide performance analytics
- Trend analysis over time
- Exportable reports

#### Content Management
- Upload custom reading texts
- Manage difficulty levels
- Create assessment templates
- RAG-powered content generation
