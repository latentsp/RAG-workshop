# Enhanced Text Splitter Guide

This guide explains the enhanced `do_chunk_text()` function that now supports multiple text splitter types from LangChain, including semantic splitters and other specialized options.

## 🚀 Quick Start

The function maintains backward compatibility, so existing code will continue to work:

```python
from utility import do_chunk_text

# Basic usage (unchanged)
chunks = do_chunk_text(text, chunk_size=500, chunk_overlap=50)

# New: Specify splitter type
chunks = do_chunk_text(text, chunk_size=500, chunk_overlap=50, splitter_type="semantic")
```

## 📋 Available Splitter Types

### 1. `recursive` (Default) ⭐
**Best for**: General text documents, articles, books
```python
chunks = do_chunk_text(text, splitter_type="recursive")
```
- Tries to split on paragraphs, then sentences, then words
- Maintains semantic coherence
- **Recommended for most use cases**

### 2. `character`
**Best for**: Simple, predictable splitting
```python
chunks = do_chunk_text(text, splitter_type="character", separator="\n\n")
```
- Splits on a single character or string
- Less intelligent than recursive

### 3. `semantic` (Experimental) 🧠
**Best for**: Documents where meaning boundaries are critical
```python
chunks = do_chunk_text(text, splitter_type="semantic")
```
- Uses AI embeddings to find semantic boundaries
- Groups semantically similar sentences together
- Requires OpenAI API key

### 4. `token`
**Best for**: Token-aware applications
```python
chunks = do_chunk_text(text, splitter_type="token", chunk_size=100)
```
- Splits based on token count (not character count)
- Useful for LLM context window management

### 5. `html_header`
**Best for**: HTML documents with clear header structure
```python
chunks = do_chunk_text(html_text, splitter_type="html_header")
```
- Splits HTML by header tags (h1, h2, h3, etc.)
- Preserves document structure

### 6. `html_section`
**Best for**: HTML documents with section-based layout
```python
chunks = do_chunk_text(html_text, splitter_type="html_section")
```
- Similar to html_header but focuses on sections
- Uses XSLT transformations

### 7. `markdown`
**Best for**: Markdown documents
```python
chunks = do_chunk_text(markdown_text, splitter_type="markdown")
```
- Splits on Markdown headers (#, ##, ###)
- Maintains document hierarchy

### 8. `python`
**Best for**: Python source code
```python
chunks = do_chunk_text(python_code, splitter_type="python")
```
- Understands Python syntax
- Splits on functions, classes, etc.

### 9. `latex`
**Best for**: LaTeX documents
```python
chunks = do_chunk_text(latex_text, splitter_type="latex")
```
- Understands LaTeX structure
- Splits on sections, subsections, etc.

### 10. `nltk`
**Best for**: Natural language processing
```python
chunks = do_chunk_text(text, splitter_type="nltk")
```
- Uses NLTK for sentence boundary detection
- More accurate sentence splitting

### 11. `spacy`
**Best for**: Advanced NLP applications
```python
chunks = do_chunk_text(text, splitter_type="spacy", pipeline="en_core_web_sm")
```
- Uses spaCy for intelligent text processing
- Supports multiple languages


Happy chunking! 🎯 