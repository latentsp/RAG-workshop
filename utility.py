import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get your OpenAI API key from the environment
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment. Please set it in your .env file.")


def do_invoke_llm(prompt):
    """Invoke LLM with a given prompt and return the response."""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    llm = ChatOpenAI(openai_api_key=openai_api_key)
    response = llm.invoke([HumanMessage(content=prompt)])
    print("LLM Response:", response.content)
    return response.content


def do_load_document(file_path):
    """Load a text document from the specified file path."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        print(f"✅ Document loaded successfully!")
        print(f"📄 Document length: {len(content)} characters")
        print(f"🔤 First 200 characters:\n{content[:200]}...")
        return content
    except FileNotFoundError:
        print(f"❌ Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"❌ Error loading document: {e}")
        return None


def do_chunk_text(text, chunk_size=500, chunk_overlap=50, splitter_type="recursive", **kwargs):
    """Split text into overlapping chunks using configurable splitter types.
    
    Args:
        text (str): The text to split into chunks
        chunk_size (int): Maximum size of each chunk
        chunk_overlap (int): Number of characters to overlap between chunks
        splitter_type (str): Type of splitter to use. Options:
            - "recursive": RecursiveCharacterTextSplitter (default, recommended for general text)
            - "character": CharacterTextSplitter (simple character-based splitting)
            - "semantic": SemanticChunker (experimental, splits by semantic similarity)
            - "token": TokenTextSplitter (splits by tokens)
            - "html_header": HTMLHeaderTextSplitter (splits HTML by headers)
            - "html_section": HTMLSectionSplitter (splits HTML by sections)
            - "markdown": MarkdownHeaderTextSplitter (splits Markdown by headers)
            - "python": PythonCodeTextSplitter (splits Python code)
            - "latex": LatexTextSplitter (splits LaTeX documents)
            - "nltk": NLTKTextSplitter (uses NLTK for sentence splitting)
            - "spacy": SpacyTextSplitter (uses spaCy for sentence splitting)
        **kwargs: Additional arguments specific to each splitter type
    
    Returns:
        list: List of text chunks
    """
    # Validate input text
    if text is None:
        print("❌ Error: Cannot chunk text - input text is None")
        print("💡 This usually means the document failed to load. Check the file path and permissions.")
        return []
    
    if not isinstance(text, str):
        print(f"❌ Error: Expected string input, got {type(text)}")
        return []
    
    if len(text.strip()) == 0:
        print("⚠️ Warning: Input text is empty")
        return []
    
    print(f"🔧 Using {splitter_type} text splitter...")
    
    try:
        if splitter_type == "recursive":
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            
            # Get custom separators if provided
            separators = kwargs.get('separators', ["\n\n", "\n", ". ", "! ", "? ", " ", ""])
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=separators
            )
            
        elif splitter_type == "character":
            from langchain.text_splitter import CharacterTextSplitter
            
            separator = kwargs.get('separator', '\n\n')
            
            text_splitter = CharacterTextSplitter(
                separator=separator,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )
            
        elif splitter_type == "semantic":
            try:
                from langchain_experimental.text_splitter import SemanticChunker
                from langchain_openai import OpenAIEmbeddings
                
                embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                breakpoint_threshold_type = kwargs.get('breakpoint_threshold_type', 'percentile')
                
                text_splitter = SemanticChunker(
                    embeddings=embeddings,
                    breakpoint_threshold_type=breakpoint_threshold_type
                )
                
            except ImportError:
                print("❌ SemanticChunker requires langchain-experimental. Installing...")
                import subprocess
                subprocess.run(["pip", "install", "langchain-experimental"], check=True)
                from langchain_experimental.text_splitter import SemanticChunker
                from langchain_openai import OpenAIEmbeddings
                
                embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                breakpoint_threshold_type = kwargs.get('breakpoint_threshold_type', 'percentile')
                
                text_splitter = SemanticChunker(
                    embeddings=embeddings,
                    breakpoint_threshold_type=breakpoint_threshold_type
                )
                
        elif splitter_type == "token":
            from langchain.text_splitter import TokenTextSplitter
            
            encoding_name = kwargs.get('encoding_name', 'gpt2')
            
            text_splitter = TokenTextSplitter(
                encoding_name=encoding_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
        elif splitter_type == "html_header":
            from langchain.text_splitter import HTMLHeaderTextSplitter
            
            headers_to_split_on = kwargs.get('headers_to_split_on', [
                ("h1", "Header 1"),
                ("h2", "Header 2"),
                ("h3", "Header 3"),
            ])
            
            text_splitter = HTMLHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on
            )
            
        elif splitter_type == "html_section":
            from langchain.text_splitter import HTMLSectionSplitter
            
            headers_to_split_on = kwargs.get('headers_to_split_on', [
                ("h1", "Header 1"),
                ("h2", "Header 2"),
            ])
            
            text_splitter = HTMLSectionSplitter(
                headers_to_split_on=headers_to_split_on
            )
            
        elif splitter_type == "markdown":
            from langchain.text_splitter import MarkdownHeaderTextSplitter
            
            headers_to_split_on = kwargs.get('headers_to_split_on', [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ])
            
            text_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on
            )
            
        elif splitter_type == "python":
            from langchain.text_splitter import PythonCodeTextSplitter
            
            text_splitter = PythonCodeTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
        elif splitter_type == "latex":
            from langchain.text_splitter import LatexTextSplitter
            
            text_splitter = LatexTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
        elif splitter_type == "nltk":
            try:
                from langchain.text_splitter import NLTKTextSplitter
                
                text_splitter = NLTKTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                
            except ImportError:
                print("❌ NLTKTextSplitter requires nltk. Installing...")
                import subprocess
                subprocess.run(["pip", "install", "nltk"], check=True)
                from langchain.text_splitter import NLTKTextSplitter
                
                text_splitter = NLTKTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                
        elif splitter_type == "spacy":
            try:
                from langchain.text_splitter import SpacyTextSplitter
                
                pipeline = kwargs.get('pipeline', 'en_core_web_sm')
                
                text_splitter = SpacyTextSplitter(
                    pipeline=pipeline,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                
            except ImportError:
                print("❌ SpacyTextSplitter requires spacy. Installing...")
                import subprocess
                subprocess.run(["pip", "install", "spacy"], check=True)
                from langchain.text_splitter import SpacyTextSplitter
                
                pipeline = kwargs.get('pipeline', 'en_core_web_sm')
                
                text_splitter = SpacyTextSplitter(
                    pipeline=pipeline,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                
        else:
            raise ValueError(f"❌ Unknown splitter_type: {splitter_type}. "
                           f"Available options: recursive, character, semantic, token, "
                           f"html_header, html_section, markdown, python, latex, nltk, spacy")
        
        # Split the text
        if splitter_type in ["html_header", "html_section", "markdown"]:
            # These splitters work differently and return Document objects
            chunks = text_splitter.split_text(text)
            if chunks and hasattr(chunks[0], 'page_content'):
                chunks = [doc.page_content for doc in chunks]
        else:
            chunks = text_splitter.split_text(text)
        
        print(f"✅ Text chunked successfully with {splitter_type} splitter!")
        print(f"📊 Number of chunks: {len(chunks)}")
        
        if chunks:
            avg_chunk_size = sum(len(chunk) for chunk in chunks) / len(chunks)
            print(f"📏 Average chunk size: {avg_chunk_size:.0f} characters")
            print(f"\n🔍 First chunk preview ({splitter_type} splitter):\n{chunks[0][:300]}...")
        
        return chunks
        
    except Exception as e:
        print(f"❌ Error with {splitter_type} splitter: {e}")
        print("🔄 Falling back to recursive character splitter...")
        
        # Fallback to recursive splitter
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        chunks = text_splitter.split_text(text)
        
        print(f"✅ Text chunked successfully with fallback recursive splitter!")
        print(f"📊 Number of chunks: {len(chunks)}")
        print(f"📏 Average chunk size: {sum(len(chunk) for chunk in chunks) / len(chunks):.0f} characters")
        print(f"\n🔍 First chunk preview:\n{chunks[0][:300]}...")
        
        return chunks


def do_create_vector_store(chunks):
    """Create embeddings and FAISS vector store from text chunks."""
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    
    # Create embeddings model
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    print("🔄 Creating embeddings and vector store...")
    print("⏳ This may take a moment...")
    
    # Create Chroma vector store
    vector_store = Chroma.from_texts(chunks, embeddings)
    
    print(f"✅ Vector store created successfully!")
    print(f"🗃️ Stored {len(chunks)} document chunks")
    print(f"🧮 Each embedding has {len(vector_store.embeddings.embed_query('test'))} dimensions")
    
    return vector_store


def do_similarity_search(vector_store, query, k=3):
    """Search for the most relevant chunks based on the query."""
    print(f"🔍 Searching for: '{query}'")
    print(f"📊 Retrieving top {k} most relevant chunks...\n")
    
    # Perform similarity search
    relevant_docs = vector_store.similarity_search(query, k=k)
    
    print(f"✅ Found {len(relevant_docs)} relevant chunks:")
    print("="*80)
    
    for i, doc in enumerate(relevant_docs, 1):
        print(f"\n🔸 Chunk {i}:")
        print(f"📝 Content: {doc.page_content[:300]}...")
        print("-"*60)
    
    return relevant_docs


def do_rag_query(vector_store, query, k=3):
    """Complete RAG pipeline: retrieve relevant chunks and generate response."""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    
    print(f"🎯 RAG Query: '{query}'")
    print("="*80)
    
    # Step 1: Retrieve relevant chunks
    print("🔍 Step 1: Retrieving relevant information...")
    relevant_docs = vector_store.similarity_search(query, k=k)
    
    # Combine retrieved content
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    print(f"✅ Retrieved {len(relevant_docs)} relevant chunks")
    print(f"📄 Total context length: {len(context)} characters")
    
    # Step 2: Generate response using LLM
    print("\n🤖 Step 2: Generating response...")
    
    # Create prompt with context
    prompt = f"""Based on the following context, please answer the question. If the answer is not in the context, say so.

            Context:
            {context}

            Question: 
            {query}

            Answer:"""
    
    # Generate response
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)
    response = llm.invoke([HumanMessage(content=prompt)])
    
    print("✅ Response generated!")
    print("="*80)
    print("🔊 RAG Response:")
    print(response.content)
    print("="*80)
    
    return {
        'query': query,
        'retrieved_chunks': relevant_docs,
        'context': context,
        'response': response.content
    } 