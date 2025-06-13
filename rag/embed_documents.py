"""
Document embedding functionality for Arabic text chunks
Handles embedding generation using Google's embedding models and storage
"""

import os
import logging
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import google.generativeai as genai
from dataclasses import dataclass, asdict
from .load_documents import DocumentChunk, DocumentLoader
import time
from dotenv import load_dotenv

# Load environment variables - use absolute path
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(env_path)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EmbeddedChunk:
    """Represents a document chunk with its embedding"""
    chunk: DocumentChunk
    embedding: np.ndarray
    embedding_model: str
    created_at: str

class EmbeddingGenerator:
    """Handles embedding generation using Google's Gemini models"""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "models/text-embedding-004"):
        """
        Initialize the embedding generator

        Args:
            api_key: Google AI API key (will use env var if not provided)
            model_name: The embedding model to use
        """
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        self.model_name = model_name

        if not self.api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY environment variable or pass api_key parameter")

        # Configure the API
        genai.configure(api_key=self.api_key)

        # Test the connection
        self._test_connection()

    def _test_connection(self):
        """Test the connection to Google AI"""
        try:
            # Try to list models to test connection
            models = list(genai.list_models())
            logger.info(f"Successfully connected to Google AI. Available models: {len(models)}")
        except Exception as e:
            logger.error(f"Failed to connect to Google AI: {e}")
            raise

    def generate_embedding(self, text: str, title: str = "") -> np.ndarray:
        """
        Generate embedding for a text chunk

        Args:
            text: The text to embed
            title: Optional title for better context

        Returns:
            numpy array representing the embedding
        """
        try:
            # Prepare the content
            content = f"{title}\n\n{text}" if title else text

            # Generate embedding with timeout
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

            def generate_embedding():
                return genai.embed_content(
                    model=self.model_name,
                    content=content,
                    task_type="retrieval_document"
                )

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(generate_embedding)
                result = future.result(timeout=30)  # 30 second timeout

            embedding = np.array(result['embedding'], dtype=np.float32)
            return embedding

        except (TimeoutError, FutureTimeoutError) as timeout_error:
            logger.error(f"Timeout generating embedding: {timeout_error}")
            return np.zeros(768, dtype=np.float32)  # Default embedding dimension
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(768, dtype=np.float32)  # Default embedding dimension

    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query

        Args:
            query: The search query

        Returns:
            numpy array representing the query embedding
        """
        try:
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

            def generate_query_embedding():
                return genai.embed_content(
                    model=self.model_name,
                    content=query,
                    task_type="retrieval_query"
                )

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(generate_query_embedding)
                result = future.result(timeout=30)  # 30 second timeout

            embedding = np.array(result['embedding'], dtype=np.float32)
            return embedding

        except (TimeoutError, FutureTimeoutError) as timeout_error:
            logger.error(f"Timeout generating query embedding: {timeout_error}")
            return np.zeros(768, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(768, dtype=np.float32)

    def batch_generate_embeddings(self, chunks: List[DocumentChunk],
                                 batch_size: int = 10, delay: float = 1.0) -> List[EmbeddedChunk]:
        """
        Generate embeddings for a batch of document chunks

        Args:
            chunks: List of document chunks to embed
            batch_size: Number of chunks to process at once
            delay: Delay between batches to respect rate limits

        Returns:
            List of embedded chunks
        """
        embedded_chunks = []
        total_chunks = len(chunks)

        logger.info(f"Starting to generate embeddings for {total_chunks} chunks")

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_chunks + batch_size - 1) // batch_size

            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)")

            for chunk in batch:
                try:
                    # Generate title from source file and page
                    title = f"من {chunk.source_file} - صفحة {chunk.page_number}"

                    # Generate embedding
                    embedding = self.generate_embedding(chunk.content, title)

                    # Create embedded chunk
                    embedded_chunk = EmbeddedChunk(
                        chunk=chunk,
                        embedding=embedding,
                        embedding_model=self.model_name,
                        created_at=time.strftime("%Y-%m-%d %H:%M:%S")
                    )

                    embedded_chunks.append(embedded_chunk)

                except Exception as e:
                    logger.error(f"Error processing chunk {chunk.chunk_id}: {e}")
                    continue

            # Add delay between batches to respect rate limits
            if i + batch_size < total_chunks and delay > 0:
                logger.info(f"Waiting {delay} seconds before next batch...")
                time.sleep(delay)

        logger.info(f"Successfully generated embeddings for {len(embedded_chunks)}/{total_chunks} chunks")
        return embedded_chunks

class EmbeddingStorage:
    """Handles storage and retrieval of embeddings"""

    def __init__(self, storage_dir: str = "models"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

        self.embeddings_file = self.storage_dir / "embeddings.pkl"
        self.metadata_file = self.storage_dir / "embeddings_metadata.json"

    def save_embeddings(self, embedded_chunks: List[EmbeddedChunk]) -> bool:
        """
        Save embeddings to disk

        Args:
            embedded_chunks: List of embedded chunks to save

        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare data for storage
            embeddings_data = []
            metadata = {
                "total_chunks": len(embedded_chunks),
                "embedding_model": embedded_chunks[0].embedding_model if embedded_chunks else "",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "chunks_info": []
            }

            for embedded_chunk in embedded_chunks:
                # Store embedding and chunk data separately
                embeddings_data.append({
                    "chunk_id": embedded_chunk.chunk.chunk_id,
                    "embedding": embedded_chunk.embedding,
                    "chunk_data": asdict(embedded_chunk.chunk)
                })

                # Add to metadata
                metadata["chunks_info"].append({
                    "chunk_id": embedded_chunk.chunk.chunk_id,
                    "source_file": embedded_chunk.chunk.source_file,
                    "page_number": embedded_chunk.chunk.page_number,
                    "content_length": len(embedded_chunk.chunk.content),
                    "word_count": embedded_chunk.chunk.metadata.get('word_count', 0)
                })

            # Save embeddings (binary format for efficiency)
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(embeddings_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Save metadata (JSON format for readability)
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            logger.info(f"Successfully saved {len(embedded_chunks)} embeddings to {self.embeddings_file}")
            return True

        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
            return False

    def load_embeddings(self) -> List[EmbeddedChunk]:
        """
        Load embeddings from disk

        Returns:
            List of embedded chunks
        """
        try:
            if not self.embeddings_file.exists():
                logger.warning(f"Embeddings file not found: {self.embeddings_file}")
                return []

            # Load embeddings data
            with open(self.embeddings_file, 'rb') as f:
                embeddings_data = pickle.load(f)

            # Load metadata
            metadata = {}
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

            # Reconstruct embedded chunks
            embedded_chunks = []
            embedding_model = metadata.get("embedding_model", "unknown")

            for data in embeddings_data:
                # Reconstruct document chunk
                chunk_data = data["chunk_data"]
                chunk = DocumentChunk(
                    content=chunk_data["content"],
                    metadata=chunk_data["metadata"],
                    chunk_id=chunk_data["chunk_id"],
                    page_number=chunk_data["page_number"],
                    source_file=chunk_data["source_file"]
                )

                # Create embedded chunk
                embedded_chunk = EmbeddedChunk(
                    chunk=chunk,
                    embedding=data["embedding"],
                    embedding_model=embedding_model,
                    created_at=metadata.get("created_at", "unknown")
                )

                embedded_chunks.append(embedded_chunk)

            logger.info(f"Successfully loaded {len(embedded_chunks)} embeddings from disk")
            return embedded_chunks

        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return []

    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about stored embeddings"""
        info = {
            "embeddings_file_exists": self.embeddings_file.exists(),
            "metadata_file_exists": self.metadata_file.exists(),
            "embeddings_file_size": 0,
            "metadata": {}
        }

        if self.embeddings_file.exists():
            info["embeddings_file_size"] = self.embeddings_file.stat().st_size

        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    info["metadata"] = json.load(f)
            except Exception as e:
                logger.error(f"Error reading metadata: {e}")

        return info

class DocumentEmbedder:
    """Main class that orchestrates document loading and embedding"""

    def __init__(self, api_key: Optional[str] = None,
                 documents_dir: str = "rag_documents/pdfs",
                 storage_dir: str = "models"):
        self.loader = DocumentLoader(documents_dir)
        self.generator = EmbeddingGenerator(api_key)
        self.storage = EmbeddingStorage(storage_dir)

    def embed_all_documents(self, chunk_size: int = 1000, overlap: int = 100,
                          batch_size: int = 10, delay: float = 1.0,
                          force_reload: bool = False) -> List[EmbeddedChunk]:
        """
        Complete pipeline: load documents, generate embeddings, and store them

        Args:
            chunk_size: Maximum size of text chunks
            overlap: Overlap between chunks
            batch_size: Number of chunks to process in each batch
            delay: Delay between batches
            force_reload: Force reloading even if embeddings exist

        Returns:
            List of embedded chunks
        """
        # Check if embeddings already exist
        if not force_reload:
            existing_embeddings = self.storage.load_embeddings()
            if existing_embeddings:
                logger.info(f"Found existing embeddings ({len(existing_embeddings)} chunks)")
                return existing_embeddings

        logger.info("Starting document embedding pipeline...")

        # Step 1: Load and chunk documents
        logger.info("Step 1: Loading and chunking documents...")
        chunks = self.loader.load_documents(chunk_size, overlap)

        if not chunks:
            logger.error("No document chunks found")
            return []

        # Step 2: Generate embeddings
        logger.info("Step 2: Generating embeddings...")
        embedded_chunks = self.generator.batch_generate_embeddings(
            chunks, batch_size, delay
        )

        if not embedded_chunks:
            logger.error("No embeddings generated")
            return []

        # Step 3: Store embeddings
        logger.info("Step 3: Storing embeddings...")
        success = self.storage.save_embeddings(embedded_chunks)

        if success:
            logger.info("Document embedding pipeline completed successfully!")
        else:
            logger.error("Failed to save embeddings")

        return embedded_chunks

def main():
    """Test the embedding functionality"""
    try:
        # Initialize embedder
        embedder = DocumentEmbedder()

        # Run the complete pipeline
        embedded_chunks = embedder.embed_all_documents(
            chunk_size=800,
            overlap=100,
            batch_size=5,
            delay=1.5
        )

        if embedded_chunks:
            print(f"Successfully embedded {len(embedded_chunks)} document chunks")

            # Show example
            first_chunk = embedded_chunks[0]
            print(f"\nExample embedded chunk:")
            print(f"Source: {first_chunk.chunk.source_file}")
            print(f"Page: {first_chunk.chunk.page_number}")
            print(f"Content preview: {first_chunk.chunk.content[:100]}...")
            print(f"Embedding shape: {first_chunk.embedding.shape}")
            print(f"Embedding model: {first_chunk.embedding_model}")

        else:
            print("No documents were embedded")

    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()
