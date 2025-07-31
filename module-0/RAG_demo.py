#!/usr/bin/env python3
"""
Simple RAG (Retrieval-Augmented Generation) Demo
Uses utility functions to demonstrate the complete RAG workflow.
"""

import sys
import os

# Add root directory to path to import utility functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utility import (
    do_load_document,
    do_chunk_text,
    do_create_vector_store,
    do_rag_query
)


def main():
    """Run a simple RAG demonstration."""
    print("🚀 Starting RAG Demo")
    print("="*50)

    document_path = "../alice_in_wonderland_book.txt"
    document_content = do_load_document(document_path)
    text_chunks = do_chunk_text(document_content, chunk_size=500, chunk_overlap=50)
    vector_store = do_create_vector_store(text_chunks)
    
    sample_queries = [
        "What did Alice find when she fell down the rabbit hole?",
        "Who did Alice meet at the tea party?",
        "What happened when Alice drank from the bottle?"
    ]
    
    for i, query in enumerate(sample_queries, 1):
        print(f"\n{'='*20} Query {i} {'='*20}")
        result = do_rag_query(vector_store, query)
        print("\n")
    
    print("✅ RAG Demo completed successfully!")


if __name__ == "__main__":
    main() 