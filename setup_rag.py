#!/usr/bin/env python3
"""
Setup script for initializing the RAG system with the Arabic PDF
Run this once to process your PDF and create the vector store
"""

import os
import sys
from pathlib import Path
import logging

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rag import RAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_rag_system():
    """Initialize the RAG system with the existing PDF"""
    try:
        logger.info("🚀 Setting up RAG system...")

        # Check if Google API key is set
        if not os.getenv('GOOGLE_API_KEY'):
            logger.error("❌ GOOGLE_API_KEY environment variable not set!")
            logger.info("Please set your Google AI API key:")
            logger.info("export GOOGLE_API_KEY='your-api-key-here'")
            return False

        # Check if PDF exists
        pdf_path = Path("rag_documents/pdfs/tarl-ar-6.pdf")
        if not pdf_path.exists():
            logger.error(f"❌ PDF file not found: {pdf_path}")
            logger.info("Please ensure your Arabic PDF is placed in rag_documents/pdfs/")
            return False

        logger.info(f"📚 Found PDF: {pdf_path}")

        # Initialize RAG pipeline
        logger.info("⚙️ Initializing RAG pipeline...")
        rag = RAGPipeline()

        # Process the documents and create embeddings
        logger.info("🔄 Processing documents and creating vector store...")
        rag.initialize_system()

        logger.info("✅ RAG system setup completed successfully!")
        logger.info(f"📊 Processed documents: {len(rag.vector_store.embedded_chunks) if rag.vector_store else 0}")

        # Optional: Test with a simple query
        logger.info("🧪 Testing with a simple query...")
        test_response = rag.ask("ما هو هذا الكتاب؟")
        logger.info(f"📖 Test response: {test_response.answer[:100]}...")

        return True

    except Exception as e:
        logger.error(f"❌ RAG system setup failed: {e}")
        return False

if __name__ == "__main__":
    success = setup_rag_system()
    if success:
        print("\n🎉 RAG system is ready!")
        print("You can now:")
        print("1. Run the Flask application: python app.py")
        print("2. Test the system: python test_rag.py")
        print("3. Use the RAG API endpoints")
    else:
        print("\n❌ Setup failed. Please check the logs above.")

    sys.exit(0 if success else 1)
