#!/usr/bin/env python3
"""
Simple test script for the enhanced do_chunk_text function.
This script tests the function with a basic text sample.
"""

def test_splitter_function():
    """Test the enhanced chunking function with basic functionality."""
    
    print("🧪 Testing Enhanced Text Splitter Function")
    print("=" * 50)
    
    # Sample text for testing
    sample_text = """
    Alice was beginning to get very tired of sitting by her sister on the bank, 
    and of having nothing to do. Once or twice she had peeped into the book her 
    sister was reading, but it had no pictures or conversations in it.
    
    "And what is the use of a book," thought Alice, "without pictures or conversations?"
    
    So she was considering in her own mind, as well as she could, for the hot day 
    made her feel very sleepy and stupid, whether the pleasure of making a daisy-chain 
    would be worth the trouble of getting up and picking the daisies, when suddenly 
    a White Rabbit with pink eyes ran close by her.
    """

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