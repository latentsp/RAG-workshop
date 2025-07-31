#!/usr/bin/env python3
"""
Exercise 1: Demonstrating the Importance of Context in RAG

Objectives: 
- Showcase that right context matters for accurate answers
- Get familiar with the initial RAG setup
- Compare LLM responses without context vs with proper RAG context

This exercise asks specific questions about details from Alice in Wonderland,
first without any context (pure LLM), then with proper book context via RAG.
"""

import sys
import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get your OpenAI API key from the environment
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment. Please set it in your .env file.")

# Add root directory to path to import utility functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utility import (
    do_load_document,
    do_chunk_text, 
    do_create_vector_store,
    do_rag_query,
    do_invoke_llm
)


def do_run_without_rag():
    """Run questions without any context - just pure LLM knowledge."""
    print("🚫 Running WITHOUT RAG (No Context)")
    print("="*60)

    # TODO: Add an array of questions
    questions = []
    
    results = []
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*20} Question {i} {'='*20}")
        print(f"❓ {question}")
        print("\n🤖 LLM Response (No Context):")
        
        # TODO: Invoke the LLM with the question
        response = 
        results.append({
            'question': question,
            'no_rag_response': response
        })
        print("-"*60)
    
    return results


def do_run_with_rag():
    """Run the same questions with RAG using Alice in Wonderland context."""
    print("\n✅ Running WITH RAG (Alice in Wonderland Context)")
    print("="*60)
    
    # Load and prepare the Alice in Wonderland book
    print("\n📖 Loading Alice in Wonderland...")
    # TODO: Download a book and save it as a txt file (https://www.gutenberg.org/browse/scores/top)
    # TODO: Add the path to a book
    book_path = 
    document_content = do_load_document(book_path)
    
    if not document_content:
        print("❌ Could not load the book!")
        return []
    
    # Chunk the text
    print("\n✂️ Chunking the book...")
    # TODO: fix
    chunks = do_chunk_text
    
    # Create vector store
    print("\n🗄️ Creating vector store...")
    # TODO: fix
    vector_store = do_create_vector_store
    
    # Run RAG queries
    print("\n🔍 Running RAG queries...")
    # TODO: Add an array of questions
    questions = 
    
    results = []
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*20} RAG Query {i} {'='*20}")
        result = do_rag_query(vector_store, question)
        results.append(result)
        print("-"*60)
    
    return results


def do_compare_results(no_rag_results, rag_results):
    """Compare the results from both approaches."""
    print("\n📊 COMPARISON: No RAG vs RAG")
    print("="*80)
    
    for i, (no_rag, rag) in enumerate(zip(no_rag_results, rag_results), 1):
        print(f"\n🔸 Question {i}: {no_rag['question']}")
        print("-"*80)
        
        print("🚫 WITHOUT RAG:")
        print(no_rag['no_rag_response'])
        
        print("\n✅ WITH RAG:")
        print(rag['response'])
        
        print("\n" + "="*80)


def main():
    """Main exercise function."""
    print("🎯 Exercise 1: The Importance of Context in RAG")
    print("="*80)
    print("This exercise demonstrates how proper context improves LLM responses")
    print("by asking specific questions about Alice in Wonderland.\n")
    
    # Step 1: Run questions without context
    no_rag_results = do_run_without_rag()
    
    # Step 2: Run questions with RAG context  
    rag_results = do_run_with_rag()
    
    # Step 3: Compare results
    if no_rag_results and rag_results:
        do_compare_results(no_rag_results, rag_results)
    
    print("\n🎉 Exercise 1 Complete!")
    print("💡 Key Takeaway: RAG provides specific, accurate context that pure LLM knowledge cannot match!")


if __name__ == "__main__":
    main()
