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


def do_chunk_text(text, chunk_size=500, chunk_overlap=50):
    """Split text into overlapping chunks for better retrieval."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )
    
    # Split the text
    chunks = text_splitter.split_text(text)
    
    print(f"✅ Text chunked successfully!")
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