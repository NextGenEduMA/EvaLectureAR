"""
End-to-end RAG pipeline for Arabic document question answering
Combines retrieval from vector store with Gemini LLM generation
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import google.generativeai as genai
from dataclasses import dataclass
from .vector_store import VectorStore, SearchResult, AdvancedVectorStore
from .embed_documents import DocumentEmbedder
import time
import json
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables - use absolute path
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(env_path)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RAGResponse:
    """Represents a complete RAG response"""
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    confidence: float
    processing_time: float
    model_used: str

class RAGPipeline:
    """
    Complete RAG (Retrieval-Augmented Generation) pipeline
    """

    def __init__(self, api_key: Optional[str] = None,
                 storage_dir: str = "models",
                 documents_dir: str = "rag_documents/pdfs"):
        """
        Initialize RAG pipeline

        Args:
            api_key: Google AI API key
            storage_dir: Directory for storing embeddings and index
            documents_dir: Directory containing PDF documents
        """
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY environment variable")

        # Configure Google AI
        genai.configure(api_key=self.api_key)

        # Initialize components
        self.vector_store = AdvancedVectorStore(storage_dir)
        self.embedder = DocumentEmbedder(api_key, documents_dir, storage_dir)

        # LLM configuration
        self.llm_model = "gemini-1.5-flash"  # Can be changed to gemini-pro
        self.generation_config = {
            "temperature": 0.3,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 1024,
        }

        # Initialize Gemini model with timeout handling
        try:
            self.model = genai.GenerativeModel(
                model_name=self.llm_model,
                generation_config=self.generation_config,
            )
            # Test the model connection
            test_response = self.model.generate_content("Test", request_options={"timeout": 10})
            logger.info("RAG pipeline initialized successfully with Google AI")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            self.model = None

    def initialize_system(self, force_rebuild: bool = False) -> bool:
        """
        Initialize the RAG system (load or build embeddings and index)

        Args:
            force_rebuild: Force rebuilding embeddings and index

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Initializing RAG system...")

            # Try to load existing index
            if not force_rebuild and self.vector_store.load_index():
                logger.info("Successfully loaded existing vector store")
                return True

            logger.info("Building new embeddings and vector store...")

            # Step 1: Embed all documents
            embedded_chunks = self.embedder.embed_all_documents(
                chunk_size=800,
                overlap=100,
                batch_size=5,
                delay=1.0,
                force_reload=force_rebuild
            )

            if not embedded_chunks:
                logger.error("Failed to create embeddings")
                return False

            # Step 2: Build vector store index
            success = self.vector_store.build_index(embedded_chunks, index_type="flat")

            if success:
                logger.info("RAG system initialized successfully!")
                return True
            else:
                logger.error("Failed to build vector store index")
                return False

        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}")
            return False

    def retrieve_relevant_chunks(self, query: str, top_k: int = 5,
                               min_similarity: float = 0.3,
                               source_files: Optional[List[str]] = None) -> List[SearchResult]:
        """
        Retrieve relevant document chunks for a query

        Args:
            query: User query in Arabic
            top_k: Number of relevant chunks to retrieve
            min_similarity: Minimum similarity threshold
            source_files: Optional filter by source files

        Returns:
            List of relevant search results
        """
        try:
            if source_files:
                # Use filtered search
                return self.vector_store.search_with_filters(
                    query=query,
                    top_k=top_k,
                    source_files=source_files,
                    min_similarity=min_similarity
                )
            else:
                # Regular search
                return self.vector_store.search(
                    query=query,
                    top_k=top_k,
                    min_similarity=min_similarity
                )

        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []

    def generate_context_prompt(self, query: str, search_results: List[SearchResult]) -> str:
        """
        Generate a prompt with context for the LLM

        Args:
            query: User query
            search_results: Retrieved document chunks

        Returns:
            Formatted prompt with context
        """
        # Prepare context from search results
        context_parts = []
        for i, result in enumerate(search_results, 1):
            chunk = result.chunk.chunk
            context_part = f"""
مصدر {i}: من {chunk.source_file} - صفحة {chunk.page_number}
المحتوى: {chunk.content}
---"""
            context_parts.append(context_part)

        context = "\n".join(context_parts)

        # Create the prompt
        prompt = f"""أنت مساعد ذكي متخصص في الإجابة على الأسئلة وإنشاء محتوى تعليمي باللغة العربية.

السياق المتاح:
{context}

السؤال أو الطلب: {query}

تعليمات:
1. استخدم المعلومات المقدمة في السياق كمصدر أساسي
2. يمكنك إضافة معلومات تعليمية عامة لإثراء المحتوى إذا كان الطلب يتعلق بإنشاء نص تعليمي
3. استشهد بالمصادر عند استخدام معلومات منها (مثل: "حسب المصدر 1...")
4. أجب باللغة العربية بشكل واضح ومفصل
5. إذا كان الطلب لإنشاء محتوى تعليمي، قم بإنشاء محتوى مناسب للفئة العمرية المحددة
6. إذا كان الطلب للإجابة على سؤال ولم تجد معلومات كافية، يمكنك تقديم إجابة عامة مفيدة

الإجابة:"""

        return prompt

    def generate_answer(self, prompt: str) -> Tuple[str, float]:
        """
        Generate answer using Gemini LLM with timeout handling

        Args:
            prompt: The formatted prompt with context

        Returns:
            Tuple of (answer, confidence_score)
        """
        try:
            if not self.model:
                return "خطأ: لم يتم تهيئة نموذج الذكاء الاصطناعي بشكل صحيح.", 0.0

            # Generate response with timeout
            import time
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

            def generate_content():
                return self.model.generate_content(prompt, request_options={"timeout": 30})

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(generate_content)
                response = future.result(timeout=45)  # 45 second timeout

            if response.text:
                # Simple confidence estimation based on response length and content
                confidence = min(0.9, len(response.text) / 500)  # Simple heuristic

                # Check if response indicates insufficient information
                no_info_phrases = [
                    "لا توجد معلومات كافية",
                    "غير متوفر في المصادر",
                    "لا يمكنني الإجابة",
                    "المعلومات غير كافية"
                ]

                for phrase in no_info_phrases:
                    if phrase in response.text:
                        confidence = max(0.1, confidence - 0.4)
                        break

                return response.text.strip(), confidence
            else:
                return "عذراً، لم أتمكن من توليد إجابة.", 0.1

        except (TimeoutError, FutureTimeoutError) as timeout_error:
            logger.error(f"Timeout generating answer: {timeout_error}")
            return "عذراً، انتهت مهلة توليد الإجابة. حاول مرة أخرى.", 0.0
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"حدث خطأ في توليد الإجابة: {str(e)}", 0.0

    def ask(self, query: str, top_k: int = 5, min_similarity: float = 0.3,
            source_files: Optional[List[str]] = None) -> RAGResponse:
        """
        Complete RAG query: retrieve relevant context and generate answer

        Args:
            query: User question in Arabic
            top_k: Number of relevant chunks to retrieve
            min_similarity: Minimum similarity threshold for retrieval
            source_files: Optional filter by source files

        Returns:
            RAGResponse with answer and metadata
        """
        start_time = time.time()

        try:
            logger.info(f"Processing RAG query: '{query[:100]}...'")

            # Step 1: Retrieve relevant chunks
            search_results = self.retrieve_relevant_chunks(
                query=query,
                top_k=top_k,
                min_similarity=min_similarity,
                source_files=source_files
            )

            if not search_results:
                return RAGResponse(
                    answer="لم أجد معلومات ذات صلة بسؤالك في المصادر المتاحة.",
                    sources=[],
                    query=query,
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    model_used=self.llm_model
                )

            # Step 2: Generate context prompt
            prompt = self.generate_context_prompt(query, search_results)

            # Step 3: Generate answer
            answer, confidence = self.generate_answer(prompt)

            # Step 4: Prepare sources information
            sources = []
            for result in search_results:
                chunk = result.chunk.chunk
                source_info = {
                    "source_file": chunk.source_file,
                    "page_number": chunk.page_number,
                    "chunk_id": chunk.chunk_id,
                    "similarity_score": result.similarity_score,
                    "content_preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
                }
                sources.append(source_info)

            # Step 5: Save search history
            self.vector_store.save_search_history(query, search_results)

            processing_time = time.time() - start_time

            logger.info(f"RAG query completed in {processing_time:.2f}s with confidence {confidence:.2f}")

            return RAGResponse(
                answer=answer,
                sources=sources,
                query=query,
                confidence=confidence,
                processing_time=processing_time,
                model_used=self.llm_model
            )

        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            return RAGResponse(
                answer=f"حدث خطأ أثناء معالجة السؤال: {str(e)}",
                sources=[],
                query=query,
                confidence=0.0,
                processing_time=time.time() - start_time,
                model_used=self.llm_model
            )

    def batch_ask(self, queries: List[str], **kwargs) -> List[RAGResponse]:
        """
        Process multiple queries in batch

        Args:
            queries: List of queries to process
            **kwargs: Additional arguments passed to ask()

        Returns:
            List of RAGResponse objects
        """
        responses = []
        total_queries = len(queries)

        logger.info(f"Processing {total_queries} queries in batch")

        for i, query in enumerate(queries, 1):
            logger.info(f"Processing query {i}/{total_queries}")

            response = self.ask(query, **kwargs)
            responses.append(response)

            # Small delay between queries to be respectful to API limits
            if i < total_queries:
                time.sleep(0.5)

        return responses

    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        vector_stats = self.vector_store.get_index_stats()
        embedding_stats = self.vector_store.embedding_storage.get_storage_info()

        return {
            "vector_store": vector_stats,
            "embeddings": embedding_stats,
            "llm_model": self.llm_model,
            "system_ready": vector_stats.get("index_exists", False)
        }

    def suggest_related_questions(self, query: str, top_k: int = 3) -> List[str]:
        """
        Suggest related questions based on retrieved content

        Args:
            query: Original query
            top_k: Number of suggestions to generate

        Returns:
            List of suggested questions
        """
        try:
            # Get relevant chunks
            search_results = self.retrieve_relevant_chunks(query, top_k=5)

            if not search_results:
                return []

            # Create prompt for generating related questions
            context = "\n".join([result.chunk.chunk.content[:300]
                               for result in search_results[:3]])

            suggestion_prompt = f"""بناءً على النص التالي، اقترح {top_k} أسئلة ذات صلة يمكن أن يسألها المستخدم:

النص:
{context}

السؤال الأصلي: {query}

اقترح {top_k} أسئلة مختلفة ومفيدة (كل سؤال في سطر منفصل):"""

            response = self.model.generate_content(suggestion_prompt)

            if response.text:
                # Parse suggested questions
                questions = [q.strip() for q in response.text.strip().split('\n')
                           if q.strip() and not q.strip().startswith('السؤال')]
                return questions[:top_k]

            return []

        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return []

def test_rag_pipeline():
    """Test the complete RAG pipeline"""
    try:
        # Initialize pipeline
        rag = RAGPipeline()

        # Initialize system
        if not rag.initialize_system():
            logger.error("Failed to initialize RAG system")
            return

        # Get system stats
        stats = rag.get_system_stats()
        print("RAG System Statistics:")
        print(json.dumps(stats, ensure_ascii=False, indent=2))

        # Test queries
        test_queries = [
            "ما هو التعلم؟",
            "كيف يتعلم الطلاب بشكل أفضل؟",
            "ما هي أهمية التعليم؟"
        ]

        for query in test_queries:
            print(f"\n{'='*50}")
            print(f"سؤال: {query}")
            print('='*50)

            # Ask question
            response = rag.ask(query, top_k=3)

            print(f"الإجابة: {response.answer}")
            print(f"الثقة: {response.confidence:.2f}")
            print(f"وقت المعالجة: {response.processing_time:.2f}s")
            print(f"عدد المصادر: {len(response.sources)}")

            if response.sources:
                print("\nالمصادر:")
                for i, source in enumerate(response.sources, 1):
                    print(f"  {i}. {source['source_file']} - صفحة {source['page_number']}")
                    print(f"     درجة التشابه: {source['similarity_score']:.3f}")

            # Get suggestions
            suggestions = rag.suggest_related_questions(query, 2)
            if suggestions:
                print(f"\nأسئلة مقترحة:")
                for i, suggestion in enumerate(suggestions, 1):
                    print(f"  {i}. {suggestion}")

    except Exception as e:
        logger.error(f"Error in test: {e}")

def main():
    """Main function for testing"""
    test_rag_pipeline()

if __name__ == "__main__":
    main()
