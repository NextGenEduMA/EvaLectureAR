#!/usr/bin/env python3
"""
Analyze the content of your processed RAG documents
This will help us understand what's actually in your PDF
"""

import os
import sys
from pathlib import Path
import pickle

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def analyze_rag_content():
    """Analyze the content of processed RAG documents"""
    try:
        print("📊 Analyzing RAG Content")
        print("=" * 50)

        # Load the embeddings to see the actual content
        embeddings_file = Path("models/embeddings.pkl")
        if not embeddings_file.exists():
            print("❌ No embeddings found. Run setup_rag.py first.")
            return False

        with open(embeddings_file, 'rb') as f:
            embedded_chunks = pickle.load(f)

        print(f"📚 Total chunks: {len(embedded_chunks)}")

        # Show first few chunks to understand content
        print("\n🔍 Sample chunks from your PDF:")
        print("-" * 40)

        for i, embedded_chunk in enumerate(embedded_chunks[:10]):
            # Extract content from dictionary structure
            chunk_data = embedded_chunk['chunk_data']
            content = chunk_data.get('content', 'No content')
            page_num = chunk_data.get('page_number', 'Unknown')
            source_file = chunk_data.get('source_file', 'Unknown')

            print(f"\n📄 Chunk {i+1} (Page {page_num}):")
            print(f"Source: {source_file}")
            print(f"Content: {content[:300]}...")

            if i == 4:  # Show first 5 chunks
                break

        # Show some statistics
        print(f"\n📈 Content Statistics:")
        print(f"- Total chunks: {len(embedded_chunks)}")

        pages = set()
        chunk_sizes = []
        for chunk in embedded_chunks:
            chunk_data = chunk['chunk_data']
            pages.add(chunk_data.get('page_number', 0))
            chunk_sizes.append(len(chunk_data.get('content', '')))

        print(f"- Pages covered: {sorted(pages)}")
        if pages:
            print(f"- Page range: {min(pages)} to {max(pages)}")

        # Show chunk sizes
        if chunk_sizes:
            print(f"- Average chunk size: {sum(chunk_sizes) / len(chunk_sizes):.0f} characters")
            print(f"- Min chunk size: {min(chunk_sizes)} characters")
            print(f"- Max chunk size: {max(chunk_sizes)} characters")

        # Extract some key terms to help with queries
        print(f"\n🔑 Suggested queries based on content:")
        all_content = []
        for chunk in embedded_chunks[:20]:
            chunk_data = chunk['chunk_data']
            all_content.append(chunk_data.get('content', ''))

        combined_content = " ".join(all_content)

        # Look for common Arabic terms
        sample_text = combined_content[:1000]  # First 1000 characters
        if sample_text:
            print(f"- Sample text: {sample_text[:300]}...")

            # Based on the debug output, I can see this is about "علي" (Ali)
            print(f"\n💡 Try these queries based on your content:")
            print(f"- 'من هو علي؟' (Who is Ali?)")
            print(f"- 'صف علي' (Describe Ali)")
            print(f"- 'ما هي صفات علي؟' (What are Ali's characteristics?)")
            print(f"- 'حدثني عن الطفل' (Tell me about the child)")
            print(f"- 'ما لون عيني علي؟' (What color are Ali's eyes?)")
            print(f"- Or search for any specific words you see in the sample text above")

        return True

    except Exception as e:
        print(f"❌ Error analyzing content: {e}")
        return False

if __name__ == "__main__":
    analyze_rag_content()
