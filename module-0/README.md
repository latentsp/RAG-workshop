# RAG Implementation

This directory contains a modular RAG (Retrieval-Augmented Generation) implementation extracted from the Jupyter notebook.

## Files

- `utility.py` - Contains all the RAG utility functions
- `RAG.py` - Simple script demonstrating the complete RAG workflow
- `alice_in_wonderland_book.txt` - Sample document for testing

## Setup

1. Install dependencies:
   ```bash
   pip install -r ../requirements.txt
   ```

2. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

### Running the Complete RAG Demo

```bash
cd module-1
python RAG.py
```

This will:
1. Load the Alice in Wonderland document
2. Chunk the text into manageable pieces
3. Create embeddings and vector store
4. Run sample queries and display results

### Using Individual Functions

You can also import and use individual functions from `utility.py`:

```python
from utility import do_load_document, do_chunk_text, do_create_vector_store, do_rag_query

# Load your document
content = do_load_document("your_document.txt")

# Process through RAG pipeline
chunks = do_chunk_text(content)
vector_store = do_create_vector_store(chunks)
result = do_rag_query(vector_store, "Your question here")
```

## Available Functions

- `do_invoke_llm(prompt)` - Basic LLM invocation
- `do_load_document(file_path)` - Load text documents
- `do_chunk_text(text, chunk_size=500, chunk_overlap=50)` - Split text into chunks
- `do_create_vector_store(chunks)` - Create vector embeddings and store
- `do_similarity_search(vector_store, query, k=3)` - Find relevant chunks
- `do_rag_query(vector_store, query, k=3)` - Complete RAG pipeline

## Notes

- Make sure you have a valid OpenAI API key
- The script uses Chroma as the vector database
- All functions follow the `do_something()` naming convention as specified 