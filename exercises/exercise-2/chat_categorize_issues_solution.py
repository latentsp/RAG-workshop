#!/usr/bin/env python3
"""
Simple RAG (Retrieval-Augmented Generation) Demo
Uses utility functions to demonstrate the complete RAG workflow.
"""
from rich import print
import sys
import os

# Add root directory to path to import utility functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

from utility import (
    do_load_document,
    do_chunk_text,
    do_create_vector_store,
    do_rag_query,
    do_invoke_llm
)
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

def offline_store():
    print("🚀 Starting RAG Demo")
    print("="*50)
    global vector_store

    document_path = os.path.join(os.path.dirname(__file__), "privacy_issues.txt")
    document_content = do_load_document(document_path)
    text_chunks = do_chunk_text(document_content, chunk_size=500, chunk_overlap=50)
    vector_store = do_create_vector_store(text_chunks)

def do_categorize_user_privacy_issue():
    """Run a simple RAG demonstration."""
    
    user_privacy_issue = input("Enter a privacy issue: ")
    relevant_issues = vector_store.similarity_search(user_privacy_issue)
    
    query = f"""
    Task: Categorize the following privacy issue into one of the following severity:
    - Low
    - Medium
    - High

    Context:
    {relevant_issues}

    User Privacy Issue:
    {user_privacy_issue}

    Return only the severity of the privacy issue. Don't be chatty.
    """


    response = do_invoke_llm(query)
    print("="*50)
    print(response)
    print("="*50)

    print("✅ RAG Exercise completed successfully!")


if __name__ == "__main__":
    offline_store()
    while True:
        try:
            do_categorize_user_privacy_issue()
        except KeyboardInterrupt:
            print("\nExiting...")
            break