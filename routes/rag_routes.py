from flask import Blueprint, request, jsonify
import logging
from rag.rag_pipeline import RAGPipeline
from models.database import db, Text
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

logger = logging.getLogger(__name__)

rag_bp = Blueprint('rag', __name__)

# Initialize RAG system with timeout handling
def get_rag_system():
    """Get or initialize RAG system with timeout handling"""
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: RAGPipeline())
            rag = future.result(timeout=10)  # 10 second timeout
            
            # Initialize system with timeout
            with ThreadPoolExecutor(max_workers=1) as init_executor:
                init_future = executor.submit(lambda: rag.initialize_system())
                init_future.result(timeout=30)  # 30 second timeout
                
            return rag
    except (TimeoutError, FutureTimeoutError) as e:
        logger.error(f"RAG system initialization timed out: {e}")
        raise
    except Exception as e:
        logger.error(f"Error initializing RAG system: {e}")
        raise

@rag_bp.route('/api/rag/upload-pdf', methods=['POST'])
def upload_pdf_knowledge_base():
    """Upload and process PDF book for knowledge base"""
    try:
        if 'pdf' not in request.files:
            return jsonify({'success': False, 'error': 'No PDF file provided'}), 400

        pdf_file = request.files['pdf']
        if pdf_file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        if not pdf_file.filename.lower().endswith('.pdf'):
            return jsonify({'success': False, 'error': 'File must be a PDF'}), 400

        # Save uploaded file temporarily
        temp_path = os.path.join('temp', pdf_file.filename)
        os.makedirs('temp', exist_ok=True)
        pdf_file.save(temp_path)

        try:
            # Initialize RAG system to process the PDF
            try:
                rag_system = get_rag_system()
                
                return jsonify({
                    'success': True,
                    'message': f'Successfully processed {pdf_file.filename}',
                    'stats': {
                        'total_documents': len(rag_system.vector_store.embedded_chunks) if rag_system.vector_store else 0,
                        'status': 'ready'
                    },
                    'knowledge_base_name': os.path.splitext(pdf_file.filename)[0]
                })
            except (TimeoutError, FutureTimeoutError):
                return jsonify({'success': False, 'error': 'RAG system initialization timed out'}), 504
            except Exception as rag_error:
                logger.error(f"RAG system error: {rag_error}")
                return jsonify({'success': False, 'error': str(rag_error)}), 500

        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    except Exception as e:
        logger.error(f"Error uploading PDF knowledge base: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@rag_bp.route('/api/rag/generate-text', methods=['POST'])
def generate_rag_text():
    """Generate text using RAG system"""
    try:
        data = request.get_json()

        # Extract parameters
        grade_level = data.get('grade_level', 1)
        difficulty_level = data.get('difficulty_level', 'easy')
        topic = data.get('topic', 'عام')
        prompt = data.get('prompt', '')
        kb_name = data.get('knowledge_base', 'arabic_book_kb')

        # Initialize system if needed
        try:
            rag_system = get_rag_system()
            
            # Create enhanced prompt for text generation
            enhanced_prompt = f"""
باستخدام المحتوى التعليمي، قم بإنشاء نص تعليمي للصف {grade_level}
مستوى الصعوبة: {difficulty_level}
الموضوع: {topic}
الطلب: {prompt}

يجب أن يكون النص مناسب للمرحلة العمرية ومفيد تعليمياً.
"""

            # Generate using RAG with timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(lambda: rag_system.ask(enhanced_prompt))
                response = future.result(timeout=60)  # 60 second timeout

            # Count words in the generated text
            word_count = len(response.answer.split())

            # Save generated text to database
            text_record = Text(
                title=f"نص مولد بـ RAG - الصف {grade_level}",
                content=response.answer,
                grade_level=grade_level,
                difficulty_level=difficulty_level,
                category=topic,
                word_count=word_count
            )

            db.session.add(text_record)
            db.session.commit()

            return jsonify({
                'success': True,
                'text': response.answer,
                'word_count': word_count,
                'confidence': response.confidence,
                'processing_time': response.processing_time,
                'text_id': text_record.id,
                'title': text_record.title,
                'sources': response.sources
            })
            
        except (TimeoutError, FutureTimeoutError) as timeout_error:
            logger.error(f"RAG generation timed out: {timeout_error}")
            return jsonify({'success': False, 'error': 'RAG generation timed out'}), 504
            
        except Exception as rag_error:
            logger.error(f"RAG generation error: {rag_error}")
            return jsonify({'success': False, 'error': str(rag_error)}), 500

    except Exception as e:
        logger.error(f"Error generating RAG text: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@rag_bp.route('/api/rag/knowledge-bases', methods=['GET'])
def list_knowledge_bases():
    """List available knowledge bases"""
    try:
        # For the new RAG system, we check if there are any processed documents
        try:
            rag_system = get_rag_system()
            
            knowledge_bases = []
            if rag_system.vector_store and rag_system.vector_store.embedded_chunks:
                # Get unique source files
                source_files = set()
                for chunk in rag_system.vector_store.embedded_chunks:
                    source_files.add(chunk.chunk.source_file)

                for source_file in source_files:
                    knowledge_bases.append({
                        'name': os.path.basename(source_file),
                        'description': f'كتاب: {os.path.basename(source_file)}',
                        'total_chunks': len([c for c in rag_system.vector_store.embedded_chunks
                                           if c.chunk.source_file == source_file]),
                        'created_at': 'متاح',
                        'status': 'ready'
                    })

            return jsonify({'knowledge_bases': knowledge_bases})
            
        except (TimeoutError, FutureTimeoutError):
            logger.warning("RAG system initialization timed out when listing knowledge bases")
            return jsonify({'knowledge_bases': [], 'error': 'RAG system initialization timed out'})
            
        except Exception as rag_error:
            logger.error(f"RAG system error when listing knowledge bases: {rag_error}")
            return jsonify({'knowledge_bases': [], 'error': str(rag_error)})

    except Exception as e:
        logger.error(f"Error listing knowledge bases: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@rag_bp.route('/api/rag/stats/<kb_name>', methods=['GET'])
def get_knowledge_base_stats(kb_name):
    """Get detailed statistics for a knowledge base"""
    try:
        # Load the specified knowledge base
        try:
            rag_system = get_rag_system()
            
            # Get system statistics
            stats = {
                'total_documents': len(rag_system.vector_store.embedded_chunks) if rag_system.vector_store else 0,
                'status': 'ready' if rag_system.vector_store and rag_system.vector_store.embedded_chunks else 'empty'
            }
            return jsonify({'success': True, 'stats': stats, 'knowledge_base': kb_name})
            
        except (TimeoutError, FutureTimeoutError):
            logger.warning(f"RAG system initialization timed out when getting stats for {kb_name}")
            return jsonify({'success': False, 'error': 'RAG system initialization timed out'}), 504
            
        except Exception as rag_error:
            logger.error(f"RAG system error when getting stats: {rag_error}")
            return jsonify({'success': False, 'error': str(rag_error)}), 500

    except Exception as e:
        logger.error(f"Error getting knowledge base stats: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@rag_bp.route('/api/rag/search', methods=['POST'])
def search_knowledge_base():
    """Search knowledge base for relevant content"""
    try:
        data = request.get_json()

        query = data.get('query', '')
        kb_name = data.get('knowledge_base', 'arabic_book_kb')
        grade_level = data.get('grade_level')
        topic = data.get('topic')
        top_k = data.get('top_k', 5)

        if not query:
            return jsonify({'success': False, 'error': 'Query is required'}), 400

        try:
            # Initialize RAG pipeline with timeout handling
            rag_system = get_rag_system()
            
            # Search using the RAG system with timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(lambda: rag_system.ask(query, top_k=top_k))
                response = future.result(timeout=60)  # 60 second timeout

            # Format results
            results = []
            for source in response.sources:
                results.append({
                    'id': source.get('chunk_id', ''),
                    'text': source.get('content_preview', ''),
                    'page_number': source.get('page_number', 0),
                    'source_file': source.get('source_file', ''),
                    'similarity_score': source.get('similarity_score', 0.0),
                    'preview': source.get('content_preview', '')
                })

            return jsonify({
                'success': True,
                'query': query,
                'answer': response.answer,
                'results': results,
                'total_found': len(results),
                'confidence': response.confidence,
                'processing_time': response.processing_time
            })
            
        except (TimeoutError, FutureTimeoutError):
            logger.warning(f"RAG search timed out for query: {query}")
            return jsonify({'success': False, 'error': 'Search timed out'}), 504
            
        except Exception as rag_error:
            logger.error(f"RAG search error: {rag_error}")
            return jsonify({'success': False, 'error': str(rag_error)}), 500

    except Exception as e:
        logger.error(f"Error searching knowledge base: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
