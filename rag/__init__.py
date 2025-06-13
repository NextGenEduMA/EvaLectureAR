"""
RAG (Retrieval-Augmented Generation) system for Arabic documents

This package provides a complete RAG implementation for Arabic PDF documents
with the following components:

- load_documents.py: PDF loading and intelligent text chunking
- embed_documents.py: Document embedding using Google's models
- vector_store.py: Vector storage and similarity search using FAISS
- rag_pipeline.py: Complete end-to-end RAG pipeline

Example usage:
    from rag import RAGPipeline

    # Initialize RAG system
    rag = RAGPipeline()
    rag.initialize_system()

    # Ask questions
    response = rag.ask("ما هو التعلم؟")
    print(response.answer)
"""

from .rag_pipeline import RAGPipeline, RAGResponse
from .vector_store import VectorStore, AdvancedVectorStore, SearchResult
from .embed_documents import DocumentEmbedder, EmbeddedChunk
from .load_documents import DocumentLoader, DocumentChunk

__version__ = "1.0.0"
__author__ = "EvaLectureAR Team"

__all__ = [
    "RAGPipeline",
    "RAGResponse",
    "VectorStore",
    "AdvancedVectorStore",
    "SearchResult",
    "DocumentEmbedder",
    "EmbeddedChunk",
    "DocumentLoader",
    "DocumentChunk"
]
