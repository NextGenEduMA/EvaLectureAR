#!/usr/bin/env python3
"""
Simple test script to verify RAG functionality without hanging
"""

import os
import sys
import time
import logging
from dotenv import load_dotenv

# Add the project root to path
sys.path.append('/Users/macbook/Projects/EvaLectureAR')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_rag_pipeline():
    """Test RAG pipeline with timeout"""
    try:
        # Load .env from the correct path - ensure absolute path
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
        load_dotenv(env_path)

        # Check if Google API key is available
        google_key = os.getenv('GOOGLE_API_KEY')
        if not google_key:
            logger.error("GOOGLE_API_KEY not found in environment")
            return False
        else:
            logger.info(f"Found Google API key: {google_key[:10]}...")

        logger.info("Testing RAG pipeline initialization...")

        # Import with timeout
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

        def init_rag():
            from rag.rag_pipeline import RAGPipeline
            rag = RAGPipeline()
            result = rag.initialize_system()
            return rag, result

        # Test initialization with timeout
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(init_rag)
            start_time = time.time()

            try:
                rag, init_result = future.result(timeout=60)
                init_time = time.time() - start_time

                if init_result:
                    logger.info(f"RAG initialization successful in {init_time:.2f}s")

                    # Test a simple query
                    logger.info("Testing simple query...")
                    query_start = time.time()

                    def test_query():
                        return rag.ask("ما هي أهمية القراءة؟")

                    with ThreadPoolExecutor(max_workers=1) as query_executor:
                        query_future = query_executor.submit(test_query)
                        response = query_future.result(timeout=30)

                    query_time = time.time() - query_start
                    logger.info(f"Query completed in {query_time:.2f}s")
                    logger.info(f"Response: {response.answer[:100]}...")

                    return True
                else:
                    logger.error("RAG initialization failed")
                    return False

            except FutureTimeoutError:
                logger.error("RAG initialization timed out after 60 seconds")
                return False

    except Exception as e:
        logger.error(f"Error testing RAG pipeline: {e}")
        return False

def test_simple_gemini():
    """Test simple Gemini API call"""
    try:
        import google.generativeai as genai

        # Load .env from the correct path - ensure absolute path
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
        load_dotenv(env_path)

        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            logger.error("GOOGLE_API_KEY not found")
            return False
        else:
            logger.info(f"Found Google API key: {api_key[:10]}...")

        genai.configure(api_key=api_key)

        model = genai.GenerativeModel('gemini-1.5-flash')

        logger.info("Testing simple Gemini API call...")
        start_time = time.time()

        response = model.generate_content("Hello, respond in Arabic with 'مرحبا'")

        elapsed = time.time() - start_time
        logger.info(f"Gemini API call completed in {elapsed:.2f}s")
        logger.info(f"Response: {response.text}")

        return True

    except Exception as e:
        logger.error(f"Error testing Gemini API: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting RAG system tests...")

    # Test 1: Simple Gemini API
    logger.info("=" * 50)
    logger.info("Test 1: Simple Gemini API")
    gemini_ok = test_simple_gemini()

    # Test 2: Full RAG pipeline
    logger.info("=" * 50)
    logger.info("Test 2: Full RAG Pipeline")
    rag_ok = test_rag_pipeline()

    # Summary
    logger.info("=" * 50)
    logger.info("Test Results:")
    logger.info(f"  Gemini API: {'✓ PASS' if gemini_ok else '✗ FAIL'}")
    logger.info(f"  RAG Pipeline: {'✓ PASS' if rag_ok else '✗ FAIL'}")

    if gemini_ok and rag_ok:
        logger.info("All tests passed! RAG system should work properly.")
        sys.exit(0)
    else:
        logger.error("Some tests failed. Check the logs above.")
        sys.exit(1)
