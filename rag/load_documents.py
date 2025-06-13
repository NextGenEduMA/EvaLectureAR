"""
Document loading and chunking functionality for Arabic PDFs
Handles PDF extraction, text preprocessing, and intelligent chunking
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import PyPDF2
import fitz  # PyMuPDF - better for Arabic text
import re
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    page_number: int
    source_file: str

class ArabicTextPreprocessor:
    """Handles Arabic text preprocessing and normalization"""

    @staticmethod
    def clean_arabic_text(text: str) -> str:
        """Clean and normalize Arabic text"""
        if not text:
            return ""

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove unwanted characters but keep Arabic punctuation
        text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s\.\,\;\:\!\?\-\(\)]', '', text)

        # Normalize Arabic characters
        arabic_normalizations = {
            'ي': 'ي',  # Normalize different forms of Ya
            'ى': 'ي',
            'ة': 'ة',  # Ta Marbuta
            'ه': 'ه',  # Ha
        }

        for old_char, new_char in arabic_normalizations.items():
            text = text.replace(old_char, new_char)

        return text.strip()

    @staticmethod
    def is_meaningful_chunk(text: str, min_words: int = 5) -> bool:
        """Check if a text chunk is meaningful enough to process"""
        if not text or len(text.strip()) < 10:
            return False

        # Count Arabic words
        arabic_words = re.findall(r'[\u0600-\u06FF]+', text)
        return len(arabic_words) >= min_words

class DocumentLoader:
    """Handles loading and processing of PDF documents"""

    def __init__(self, documents_dir: str = "rag_documents/pdfs"):
        self.documents_dir = Path(documents_dir)
        self.preprocessor = ArabicTextPreprocessor()

    def extract_text_from_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract text from PDF with page information"""
        pages_text = []

        try:
            # Try PyMuPDF first (better for Arabic)
            doc = fitz.open(str(pdf_path))
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()

                if text.strip():
                    pages_text.append({
                        'page_number': page_num + 1,
                        'content': text,
                        'source_file': pdf_path.name
                    })
            doc.close()

        except Exception as e:
            logger.warning(f"PyMuPDF failed for {pdf_path.name}, trying PyPDF2: {e}")

            # Fallback to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page_num, page in enumerate(pdf_reader.pages):
                        text = page.extract_text()

                        if text.strip():
                            pages_text.append({
                                'page_number': page_num + 1,
                                'content': text,
                                'source_file': pdf_path.name
                            })

            except Exception as e2:
                logger.error(f"Both PDF extraction methods failed for {pdf_path.name}: {e2}")
                raise

        logger.info(f"Extracted {len(pages_text)} pages from {pdf_path.name}")
        return pages_text

    def intelligent_chunk_text(self, text: str, max_chunk_size: int = 1000,
                             overlap: int = 100) -> List[str]:
        """
        Intelligently chunk Arabic text based on sentence boundaries and meaning
        """
        if not text or len(text) < max_chunk_size:
            return [text] if text else []

        # Arabic sentence boundaries
        sentence_endings = ['.', '!', '?', '؟', '۔']

        # Split by sentences first
        sentences = []
        current_sentence = ""

        for char in text:
            current_sentence += char
            if char in sentence_endings:
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""

        # Add remaining text
        if current_sentence.strip():
            sentences.append(current_sentence.strip())

        # Group sentences into chunks
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # If adding this sentence would exceed max size, start new chunk
            if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())

                # Add overlap from previous chunk
                if overlap > 0 and chunks:
                    overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def load_documents(self, chunk_size: int = 1000, overlap: int = 100) -> List[DocumentChunk]:
        """Load all PDF documents and return processed chunks"""
        all_chunks = []

        if not self.documents_dir.exists():
            raise FileNotFoundError(f"Documents directory not found: {self.documents_dir}")

        pdf_files = list(self.documents_dir.glob("*.pdf"))

        if not pdf_files:
            logger.warning(f"No PDF files found in {self.documents_dir}")
            return []

        logger.info(f"Found {len(pdf_files)} PDF files to process")

        for pdf_file in pdf_files:
            try:
                logger.info(f"Processing {pdf_file.name}...")

                # Extract text from PDF
                pages_data = self.extract_text_from_pdf(pdf_file)

                for page_data in pages_data:
                    # Clean the text
                    cleaned_text = self.preprocessor.clean_arabic_text(page_data['content'])

                    if not self.preprocessor.is_meaningful_chunk(cleaned_text):
                        continue

                    # Chunk the text
                    chunks = self.intelligent_chunk_text(cleaned_text, chunk_size, overlap)

                    for i, chunk_text in enumerate(chunks):
                        if self.preprocessor.is_meaningful_chunk(chunk_text):
                            chunk_id = f"{pdf_file.stem}_page{page_data['page_number']}_chunk{i+1}"

                            chunk = DocumentChunk(
                                content=chunk_text,
                                metadata={
                                    'source_file': page_data['source_file'],
                                    'page_number': page_data['page_number'],
                                    'chunk_index': i,
                                    'chunk_size': len(chunk_text),
                                    'word_count': len(re.findall(r'[\u0600-\u06FF]+', chunk_text))
                                },
                                chunk_id=chunk_id,
                                page_number=page_data['page_number'],
                                source_file=page_data['source_file']
                            )

                            all_chunks.append(chunk)

                logger.info(f"Successfully processed {pdf_file.name}")

            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {e}")
                continue

        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks

    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about the documents"""
        chunks = self.load_documents()

        if not chunks:
            return {"total_chunks": 0, "total_files": 0}

        stats = {
            "total_chunks": len(chunks),
            "total_files": len(set(chunk.source_file for chunk in chunks)),
            "total_words": sum(chunk.metadata.get('word_count', 0) for chunk in chunks),
            "avg_chunk_size": sum(len(chunk.content) for chunk in chunks) / len(chunks),
            "files_processed": list(set(chunk.source_file for chunk in chunks))
        }

        return stats

def main():
    """Test the document loader"""
    loader = DocumentLoader()

    try:
        # Load documents
        chunks = loader.load_documents()

        if chunks:
            print(f"Successfully loaded {len(chunks)} chunks from documents")

            # Show first chunk as example
            first_chunk = chunks[0]
            print(f"\nExample chunk from {first_chunk.source_file}:")
            print(f"Page: {first_chunk.page_number}")
            print(f"Content preview: {first_chunk.content[:200]}...")

            # Show stats
            stats = loader.get_document_stats()
            print(f"\nDocument Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        else:
            print("No documents found or processed")

    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()
