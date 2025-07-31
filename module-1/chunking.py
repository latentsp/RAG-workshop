#!/usr/bin/env python3
"""
Simple test script for the enhanced do_chunk_text function.
This script tests the function with a basic text sample.
"""
import os
import sys
# Add root directory to path to import utility functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utility import (
    do_load_document,
    do_chunk_text,
    do_create_vector_store,
    do_rag_query
)

def test_splitter_function():
    """Test the enhanced chunking function with basic functionality."""
    
    print("🧪 Testing Enhanced Text Splitter Function")
    print("=" * 50)
    
    # Sample text for testing

    book_path = "../alice_in_wonderland_book.txt"
    sample_text = do_load_document(book_path)

    # Import the function
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utility import do_chunk_text
    
    
    # Test 1: Basic recursive splitter (default)
    print("\n📝 Test 1: Basic Recursive Splitter")
    chunks = do_chunk_text(sample_text, chunk_size=200, chunk_overlap=20)
    print(f"✅ Recursive Spillter Generated {len(chunks)} chunks")
    
    # Test 2: Try character splitter
    print("\n📝 Test 2: Character Splitter")
    chunks_char = do_chunk_text(sample_text, chunk_size=200, chunk_overlap=20, splitter_type="character")
    print(f"✅ Character splitter generated {len(chunks_char)} chunks")

    # Test 3: Custom separators
    print("\n📝 Test 3: Custom Separators")

    chunks_custom = do_chunk_text(
        sample_text, 
        chunk_size=150, 
        chunk_overlap=15, 
        splitter_type="recursive",
        separators=["\n\n", ".", "!", "?", " "]
    )
    print(f"✅ Custom separators generated {len(chunks_custom)} chunks")

if __name__ == "__main__":
    print("🚀 Starting Enhanced Text Splitter Tests\n")
    
    print("\n" + "=" * 60)

    test_splitter_function()
    
    print("\n" + "=" * 60)
    print("🏁 Test session completed!") 