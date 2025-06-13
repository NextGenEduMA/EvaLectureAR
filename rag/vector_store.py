"""
Vector store management for similarity search and retrieval
Handles storage, indexing, and querying of document embeddings
"""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path
from .embed_documents import EmbeddedChunk, EmbeddingGenerator, EmbeddingStorage
import faiss  # For efficient similarity search
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Represents a search result with relevance score"""
    chunk: EmbeddedChunk
    similarity_score: float
    rank: int

class VectorStore:
    """
    Vector store for efficient similarity search using FAISS
    """

    def __init__(self, storage_dir: str = "models"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

        # FAISS index file
        self.index_file = self.storage_dir / "faiss_index.bin"
        self.chunk_mapping_file = self.storage_dir / "chunk_mapping.pkl"

        # Initialize components
        self.embedding_storage = EmbeddingStorage(storage_dir)
        self.embedding_generator = None  # Will be initialized when needed

        # FAISS index and data
        self.index = None
        self.embedded_chunks = []
        self.dimension = 768  # Default embedding dimension

    def _initialize_embedding_generator(self):
        """Initialize embedding generator if not already initialized"""
        if self.embedding_generator is None:
            try:
                self.embedding_generator = EmbeddingGenerator()
            except Exception as e:
                logger.error(f"Failed to initialize embedding generator: {e}")
                raise

    def build_index(self, embedded_chunks: Optional[List[EmbeddedChunk]] = None,
                   index_type: str = "flat") -> bool:
        """
        Build FAISS index from embedded chunks

        Args:
            embedded_chunks: List of embedded chunks (will load from storage if None)
            index_type: Type of FAISS index ("flat", "ivf", "hnsw")

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load embedded chunks if not provided
            if embedded_chunks is None:
                embedded_chunks = self.embedding_storage.load_embeddings()

            if not embedded_chunks:
                logger.error("No embedded chunks found to build index")
                return False

            self.embedded_chunks = embedded_chunks

            # Extract embeddings
            embeddings = np.vstack([chunk.embedding for chunk in embedded_chunks])
            self.dimension = embeddings.shape[1]

            logger.info(f"Building FAISS index with {len(embedded_chunks)} vectors of dimension {self.dimension}")

            # Create FAISS index based on type
            if index_type == "flat":
                # Flat index (exact search, good for small datasets)
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine similarity)
            elif index_type == "ivf":
                # IVF index (approximate search, good for medium datasets)
                nlist = min(100, len(embedded_chunks) // 4)  # Number of clusters
                quantizer = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
                self.index.train(embeddings)
            elif index_type == "hnsw":
                # HNSW index (approximate search, good for large datasets)
                m = 16  # Number of connections
                self.index = faiss.IndexHNSWFlat(self.dimension, m)
            else:
                raise ValueError(f"Unsupported index type: {index_type}")

            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)

            # Add embeddings to index
            self.index.add(embeddings)

            # Save index and chunk mapping
            self._save_index()

            logger.info(f"Successfully built FAISS index with {self.index.ntotal} vectors")
            return True

        except Exception as e:
            logger.error(f"Error building index: {e}")
            return False

    def _save_index(self) -> bool:
        """Save FAISS index and chunk mapping to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_file))

            # Save chunk mapping
            with open(self.chunk_mapping_file, 'wb') as f:
                pickle.dump(self.embedded_chunks, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info("Successfully saved FAISS index and chunk mapping")
            return True

        except Exception as e:
            logger.error(f"Error saving index: {e}")
            return False

    def load_index(self) -> bool:
        """Load FAISS index and chunk mapping from disk"""
        try:
            if not self.index_file.exists() or not self.chunk_mapping_file.exists():
                logger.warning("Index files not found")
                return False

            # Load FAISS index
            self.index = faiss.read_index(str(self.index_file))
            self.dimension = self.index.d

            # Load chunk mapping
            with open(self.chunk_mapping_file, 'rb') as f:
                self.embedded_chunks = pickle.load(f)

            logger.info(f"Successfully loaded FAISS index with {self.index.ntotal} vectors")
            return True

        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False

    def search(self, query: str, top_k: int = 5,
               min_similarity: float = 0.3) -> List[SearchResult]:
        """
        Search for similar documents using query text

        Args:
            query: Search query in Arabic
            top_k: Number of top results to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of search results ordered by similarity
        """
        try:
            # Initialize embedding generator if needed
            self._initialize_embedding_generator()

            # Load index if not loaded
            if self.index is None:
                if not self.load_index():
                    logger.error("No index found. Please build index first.")
                    return []

            # Generate query embedding
            query_embedding = self.embedding_generator.generate_query_embedding(query)
            query_embedding = query_embedding.reshape(1, -1)

            # Normalize query embedding
            faiss.normalize_L2(query_embedding)

            # Search
            similarities, indices = self.index.search(query_embedding, top_k)

            # Prepare results
            results = []
            for rank, (idx, similarity) in enumerate(zip(indices[0], similarities[0])):
                if similarity >= min_similarity and idx < len(self.embedded_chunks):
                    result = SearchResult(
                        chunk=self.embedded_chunks[idx],
                        similarity_score=float(similarity),
                        rank=rank + 1
                    )
                    results.append(result)

            logger.info(f"Found {len(results)} relevant results for query: '{query[:50]}...'")
            return results

        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []

    def search_by_embedding(self, query_embedding: np.ndarray, top_k: int = 5,
                          min_similarity: float = 0.3) -> List[SearchResult]:
        """
        Search using a pre-computed embedding

        Args:
            query_embedding: Pre-computed query embedding
            top_k: Number of top results to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of search results ordered by similarity
        """
        try:
            # Load index if not loaded
            if self.index is None:
                if not self.load_index():
                    logger.error("No index found. Please build index first.")
                    return []

            # Ensure correct shape and normalization
            query_embedding = query_embedding.reshape(1, -1)
            faiss.normalize_L2(query_embedding)

            # Search
            similarities, indices = self.index.search(query_embedding, top_k)

            # Prepare results
            results = []
            for rank, (idx, similarity) in enumerate(zip(indices[0], similarities[0])):
                if similarity >= min_similarity and idx < len(self.embedded_chunks):
                    result = SearchResult(
                        chunk=self.embedded_chunks[idx],
                        similarity_score=float(similarity),
                        rank=rank + 1
                    )
                    results.append(result)

            return results

        except Exception as e:
            logger.error(f"Error during embedding search: {e}")
            return []

    def get_all_chunks_by_source(self, source_file: str) -> List[EmbeddedChunk]:
        """Get all chunks from a specific source file"""
        if not self.embedded_chunks:
            self.load_index()

        return [chunk for chunk in self.embedded_chunks
                if chunk.chunk.source_file == source_file]

    def get_chunk_by_id(self, chunk_id: str) -> Optional[EmbeddedChunk]:
        """Get a specific chunk by its ID"""
        if not self.embedded_chunks:
            self.load_index()

        for chunk in self.embedded_chunks:
            if chunk.chunk.chunk_id == chunk_id:
                return chunk
        return None

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        stats = {
            "index_exists": self.index is not None,
            "total_vectors": 0,
            "dimension": self.dimension,
            "total_chunks": len(self.embedded_chunks),
            "unique_sources": 0,
            "index_file_size": 0,
            "chunk_mapping_size": 0
        }

        if self.index is not None:
            stats["total_vectors"] = self.index.ntotal

        if self.embedded_chunks:
            unique_sources = set(chunk.chunk.source_file for chunk in self.embedded_chunks)
            stats["unique_sources"] = len(unique_sources)
            stats["source_files"] = list(unique_sources)

        if self.index_file.exists():
            stats["index_file_size"] = self.index_file.stat().st_size

        if self.chunk_mapping_file.exists():
            stats["chunk_mapping_size"] = self.chunk_mapping_file.stat().st_size

        return stats

    def rebuild_index(self, index_type: str = "flat") -> bool:
        """Rebuild the index from stored embeddings"""
        logger.info("Rebuilding vector store index...")

        # Load embeddings from storage
        embedded_chunks = self.embedding_storage.load_embeddings()

        if not embedded_chunks:
            logger.error("No embeddings found to rebuild index")
            return False

        # Build new index
        return self.build_index(embedded_chunks, index_type)

class AdvancedVectorStore(VectorStore):
    """
    Advanced vector store with additional features
    """

    def __init__(self, storage_dir: str = "models"):
        super().__init__(storage_dir)
        self.search_history_file = self.storage_dir / "search_history.json"
        self.search_history = []

    def search_with_filters(self, query: str, top_k: int = 5,
                          source_files: Optional[List[str]] = None,
                          page_range: Optional[Tuple[int, int]] = None,
                          min_similarity: float = 0.3) -> List[SearchResult]:
        """
        Search with additional filters

        Args:
            query: Search query
            top_k: Number of results
            source_files: Filter by specific source files
            page_range: Filter by page range (min_page, max_page)
            min_similarity: Minimum similarity threshold

        Returns:
            Filtered search results
        """
        # Get initial results
        results = self.search(query, top_k * 2, min_similarity)  # Get more results for filtering

        # Apply filters
        filtered_results = []
        for result in results:
            chunk = result.chunk.chunk

            # Filter by source files
            if source_files and chunk.source_file not in source_files:
                continue

            # Filter by page range
            if page_range:
                min_page, max_page = page_range
                if not (min_page <= chunk.page_number <= max_page):
                    continue

            filtered_results.append(result)

            # Stop when we have enough results
            if len(filtered_results) >= top_k:
                break

        return filtered_results

    def get_related_chunks(self, chunk_id: str, top_k: int = 3) -> List[SearchResult]:
        """Find chunks related to a given chunk"""
        chunk = self.get_chunk_by_id(chunk_id)
        if not chunk:
            return []

        # Use the chunk's embedding to find similar chunks
        results = self.search_by_embedding(chunk.embedding, top_k + 1)  # +1 to exclude self

        # Remove the original chunk from results
        return [r for r in results if r.chunk.chunk.chunk_id != chunk_id][:top_k]

    def save_search_history(self, query: str, results: List[SearchResult]):
        """Save search query and results for analysis"""
        history_entry = {
            "timestamp": __import__('time').strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "num_results": len(results),
            "top_similarity": results[0].similarity_score if results else 0,
            "result_sources": [r.chunk.chunk.source_file for r in results]
        }

        self.search_history.append(history_entry)

        # Save to file
        try:
            with open(self.search_history_file, 'w', encoding='utf-8') as f:
                json.dump(self.search_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving search history: {e}")

def test_vector_store():
    """Test the vector store functionality"""
    try:
        # Initialize vector store
        vector_store = VectorStore()

        # Try to load existing index
        if not vector_store.load_index():
            logger.info("No existing index found, building new one...")
            success = vector_store.build_index()
            if not success:
                logger.error("Failed to build index")
                return

        # Get stats
        stats = vector_store.get_index_stats()
        print(f"Vector Store Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # Test search
        test_queries = [
            "ما هو التعلم؟",
            "التعليم والمدرسة",
            "الطلاب والمعلمين"
        ]

        for query in test_queries:
            print(f"\nTesting search for: '{query}'")
            results = vector_store.search(query, top_k=3)

            for i, result in enumerate(results, 1):
                print(f"  {i}. Score: {result.similarity_score:.3f}")
                print(f"     Source: {result.chunk.chunk.source_file}")
                print(f"     Page: {result.chunk.chunk.page_number}")
                print(f"     Preview: {result.chunk.chunk.content[:100]}...")
                print()

    except Exception as e:
        logger.error(f"Error in test: {e}")

def main():
    """Main function for testing"""
    test_vector_store()

if __name__ == "__main__":
    main()
