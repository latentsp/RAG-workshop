"""
RAG Parameter Optimization Utility (Module 2)
=============================================

This utility focuses on measuring and comparing RAG accuracy with different parameters:
- Chunk sizes and overlaps
- Different embedders (OpenAI, Anthropic, HuggingFace)
- Various chunking strategies (Recursive, Semantic, Character, NLTK, SpaCy)
- Retrieval parameters (k values)

Key Features:
- Uses Ragas Answer Accuracy metric for evaluation
- Supports comprehensive parameter grid search
- Exports results to CSV for analysis
- Provides comparative studies across different configurations

Supported Embedders:
- openai: OpenAI embeddings (default)
- anthropic: Anthropic embeddings (requires ANTHROPIC_API_KEY)
- huggingface: HuggingFace sentence transformers

Supported Chunkers:
- recursive: RecursiveCharacterTextSplitter (default)
- semantic: SemanticChunker (requires langchain-experimental)
- character: CharacterTextSplitter
- nltk: NLTKTextSplitter (requires nltk)
- spacy: SpacyTextSplitter (requires spacy)

Example Usage:
    # Basic parameter study
    results = do_run_parameter_study()
    
    # Custom parameters
    results = do_run_parameter_study(
        chunk_sizes=[200, 500, 1000],
        chunk_overlap=100,
        k=5,
        embedder_type="anthropic",
        chunker_type="semantic"
    )
"""

import os
import asyncio
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional, Union
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Get your API keys from the environment
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

if not openai_api_key:
    print("⚠️  OPENAI_API_KEY not found in environment. Some features may not work.")
if not anthropic_api_key:
    print("⚠️  ANTHROPIC_API_KEY not found in environment. Anthropic embedder will not work.")


def do_create_test_dataset():
    """Create a test dataset with questions and reference answers for accuracy evaluation."""
    test_data = [
        {
            "question": "Who is the main character in Alice in Wonderland?",
            "reference": "The main character in Alice in Wonderland is Alice, a young girl who falls down a rabbit hole into a fantasy world."
        },
        {
            "question": "What happens to Alice at the beginning of the story?",
            "reference": "At the beginning of the story, Alice falls down a rabbit hole while chasing a white rabbit."
        },
        {
            "question": "Who does Alice follow down the rabbit hole?",
            "reference": "Alice follows a white rabbit down the rabbit hole."
        },
        {
            "question": "What kind of world does Alice discover?",
            "reference": "Alice discovers a fantasy world full of peculiar anthropomorphic creatures."
        },
        {
            "question": "What is the setting of Alice in Wonderland?",
            "reference": "The setting is a fantasy world called Wonderland, which Alice reaches by falling down a rabbit hole."
        }
    ]
    
    print(f"✅ Test dataset created with {len(test_data)} question-answer pairs")
    for i, item in enumerate(test_data, 1):
        print(f"🔸 Q{i}: {item['question']}")
    
    return test_data


def do_get_embedder(embedder_type="openai"):
    """Get the specified embedder model."""
    if embedder_type.lower() == "openai":
        if not openai_api_key:
            raise ValueError("OpenAI API key required for OpenAI embedder")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    elif embedder_type.lower() == "anthropic":
        if not anthropic_api_key:
            raise ValueError("Anthropic API key required for Anthropic embedder")
        try:
            from langchain_anthropic import AnthropicEmbeddings
            return AnthropicEmbeddings(anthropic_api_key=anthropic_api_key)
        except ImportError:
            raise ImportError("Install langchain-anthropic: pip install langchain-anthropic")
    
    elif embedder_type.lower() == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except ImportError:
            raise ImportError("Install langchain-huggingface: pip install langchain-huggingface")
    
    else:
        raise ValueError(f"Unsupported embedder type: {embedder_type}. Choose from: openai, anthropic, huggingface")


def do_create_chunker(chunker_type="recursive", chunk_size=500, chunk_overlap=50, embedder=None):
    """Create the specified text chunker."""
    
    if chunker_type.lower() == "recursive":
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
    
    elif chunker_type.lower() == "semantic":
        try:
            from langchain_experimental.text_splitter import SemanticChunker
            if embedder is None:
                embedder = do_get_embedder("openai")
            return SemanticChunker(embedder, breakpoint_threshold_type="percentile")
        except ImportError:
            raise ImportError("Install langchain-experimental: pip install langchain-experimental")
    
    elif chunker_type.lower() == "character":
        from langchain.text_splitter import CharacterTextSplitter
        return CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n\n"
        )
    
    elif chunker_type.lower() == "nltk":
        try:
            from langchain.text_splitter import NLTKTextSplitter
            return NLTKTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        except ImportError:
            raise ImportError("Install nltk: pip install nltk")
    
    elif chunker_type.lower() == "spacy":
        try:
            from langchain.text_splitter import SpacyTextSplitter
            return SpacyTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        except ImportError:
            raise ImportError("Install spacy: pip install spacy && python -m spacy download en_core_web_sm")
    
    else:
        raise ValueError(f"Unsupported chunker type: {chunker_type}. Choose from: recursive, semantic, character, nltk, spacy")


def do_chunk_text_with_params(text, chunk_size=500, chunk_overlap=50, chunker_type="recursive", embedder=None):
    """Split text into chunks with specified parameters and chunker type."""
    
    print(f"📊 Chunking with: {chunker_type} chunker, size={chunk_size}, overlap={chunk_overlap}")
    
    # Create chunker
    chunker = do_create_chunker(chunker_type, chunk_size, chunk_overlap, embedder)
    
    # Split the text
    chunks = chunker.split_text(text)
    
    print(f"✅ Created {len(chunks)} chunks")
    print(f"📏 Average chunk size: {sum(len(chunk) for chunk in chunks) / len(chunks):.0f} characters")
    
    return chunks


def do_create_vector_store_from_chunks(chunks, embedder_type="openai"):
    """Create embeddings and vector store from text chunks."""
    from langchain_community.vectorstores import Chroma
    
    # Create embeddings model
    embeddings = do_get_embedder(embedder_type)
    
    print(f"🔄 Creating vector store with {embedder_type} embedder...")
    
    # Create Chroma vector store
    vector_store = Chroma.from_texts(chunks, embeddings)
    
    print(f"✅ Vector store created with {len(chunks)} chunks")
    
    return vector_store


def do_rag_query_simple(vector_store, query, k=3):
    """Simple RAG query that returns the response for evaluation."""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    
    # Retrieve relevant chunks
    relevant_docs = vector_store.similarity_search(query, k=k)
    
    # Combine retrieved content
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
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
    
    return {
        'response': response.content,
        'retrieved_contexts': [doc.page_content for doc in relevant_docs],
        'context': context
    }


async def do_measure_accuracy_with_ragas(responses_data, test_data):
    """Measure accuracy using Ragas Answer Accuracy metric."""
    try:
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import AnswerAccuracy
        from ragas.llms import LangchainLLMWrapper
        from langchain_openai import ChatOpenAI
        
        # Create evaluator LLM
        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(openai_api_key=openai_api_key, temperature=0))
        
        # Create scorer
        scorer = AnswerAccuracy(llm=evaluator_llm)
        
        scores = []
        
        print("🔍 Evaluating responses with Ragas Answer Accuracy...")
        
        for i, (response_data, test_item) in enumerate(zip(responses_data, test_data)):
            sample = SingleTurnSample(
                user_input=test_item['question'],
                response=response_data['response'],
                reference=test_item['reference']
            )
            
            try:
                score = await scorer.single_turn_ascore(sample)
                scores.append(score)
                print(f"✅ Q{i+1} Score: {score:.3f}")
            except Exception as e:
                print(f"❌ Error evaluating Q{i+1}: {e}")
                scores.append(0.0)
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        print(f"📊 Average Answer Accuracy: {avg_score:.3f}")
        
        return {
            'individual_scores': scores,
            'average_score': avg_score,
            'total_questions': len(test_data)
        }
        
    except ImportError as e:
        print(f"❌ Ragas not installed. Install with: pip install ragas")
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"❌ Error in accuracy measurement: {e}")
        return None


def do_evaluate_with_params(document_text, chunk_size, test_data, chunk_overlap=50, k=3, 
                           embedder_type="openai", chunker_type="recursive"):
    """Evaluate RAG performance with specified parameters."""
    print(f"\n🔬 Evaluating with parameters:")
    print(f"   📏 Chunk size: {chunk_size}, overlap: {chunk_overlap}")
    print(f"   🤖 Chunker: {chunker_type}, Embedder: {embedder_type}, k: {k}")
    print("="*60)
    
    # Step 1: Chunk the document
    embedder = do_get_embedder(embedder_type) if chunker_type == "semantic" else None
    chunks = do_chunk_text_with_params(document_text, chunk_size, chunk_overlap, chunker_type, embedder)
    
    # Step 2: Create vector store
    vector_store = do_create_vector_store_from_chunks(chunks, embedder_type)
    
    # Step 3: Generate responses for all test questions
    responses_data = []
    print(f"\n🤖 Generating responses for {len(test_data)} questions...")
    
    for i, test_item in enumerate(test_data):
        print(f"🔸 Processing Q{i+1}: {test_item['question']}")
        response_data = do_rag_query_simple(vector_store, test_item['question'], k=k)
        responses_data.append(response_data)
    
    # Step 4: Measure accuracy using Ragas
    print(f"\n📏 Measuring accuracy...")
    accuracy_results = asyncio.run(do_measure_accuracy_with_ragas(responses_data, test_data))
    
    return {
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'k': k,
        'embedder_type': embedder_type,
        'chunker_type': chunker_type,
        'num_chunks': len(chunks),
        'responses': responses_data,
        'accuracy_results': accuracy_results
    }


# Backward compatibility function
def do_evaluate_with_chunk_size(document_text, chunk_size, test_data, k=3):
    """Evaluate RAG performance with a specific chunk size (backward compatibility)."""
    return do_evaluate_with_params(document_text, chunk_size, test_data, k=k)


def do_compare_rag_parameters(document_text, chunk_sizes=[200, 500, 1000, 1500], test_data=None, 
                             chunk_overlap=50, k=3, embedder_type="openai", chunker_type="recursive"):
    """Compare RAG accuracy across different parameter configurations."""
    if test_data is None:
        test_data = do_create_test_dataset()
    
    print("🏁 Starting RAG parameter comparison study")
    print(f"📊 Testing chunk sizes: {chunk_sizes}")
    print(f"🔧 Parameters: overlap={chunk_overlap}, k={k}, embedder={embedder_type}, chunker={chunker_type}")
    print(f"❓ Number of test questions: {len(test_data)}")
    print("="*80)
    
    results = []
    
    for chunk_size in chunk_sizes:
        result = do_evaluate_with_params(document_text, chunk_size, test_data, 
                                       chunk_overlap, k, embedder_type, chunker_type)
        results.append(result)
    
    # Create summary
    print("\n📈 RAG PARAMETER COMPARISON RESULTS")
    print("="*80)
    
    summary_data = []
    for result in results:
        if result['accuracy_results']:
            summary_data.append({
                'Chunk Size': result['chunk_size'],
                'Chunk Overlap': result['chunk_overlap'],
                'K': result['k'],
                'Embedder': result['embedder_type'],
                'Chunker': result['chunker_type'],
                'Number of Chunks': result['num_chunks'],
                'Average Accuracy': result['accuracy_results']['average_score'],
                'Total Questions': result['accuracy_results']['total_questions']
            })
            
            print(f"🔸 Chunk Size {result['chunk_size']:>4}: "
                  f"Accuracy = {result['accuracy_results']['average_score']:.3f}, "
                  f"Chunks = {result['num_chunks']:>3}")
    
    # Find best chunk size
    if summary_data:
        best_result = max(summary_data, key=lambda x: x['Average Accuracy'])
        print(f"\n🏆 Best performing chunk size: {best_result['Chunk Size']} "
              f"(Accuracy: {best_result['Average Accuracy']:.3f})")
    
    return {
        'results': results,
        'summary': summary_data,
        'test_data': test_data
    }


# Backward compatibility function
def do_compare_chunk_sizes(document_text, chunk_sizes=[200, 500, 1000, 1500], test_data=None, 
                          chunk_overlap=50, k=3, embedder_type="openai", chunker_type="recursive"):
    """Compare RAG accuracy across different chunk sizes (backward compatibility)."""
    return do_compare_rag_parameters(document_text, chunk_sizes, test_data, 
                                    chunk_overlap, k, embedder_type, chunker_type)


def do_compare_parameters(document_text, test_data=None, parameter_grid=None):
    """Compare RAG accuracy across different parameter combinations."""
    if test_data is None:
        test_data = do_create_test_dataset()
    
    if parameter_grid is None:
        parameter_grid = {
            'chunk_sizes': [200, 500, 1000],
            'chunk_overlaps': [50, 100],
            'k_values': [3, 5],
            'embedder_types': ['openai'],
            'chunker_types': ['recursive', 'semantic']
        }
    
    print("🔬 Starting comprehensive parameter comparison study")
    print(f"📊 Parameter grid: {parameter_grid}")
    print(f"❓ Number of test questions: {len(test_data)}")
    print("="*80)
    
    results = []
    total_combinations = (len(parameter_grid['chunk_sizes']) * 
                         len(parameter_grid['chunk_overlaps']) * 
                         len(parameter_grid['k_values']) * 
                         len(parameter_grid['embedder_types']) * 
                         len(parameter_grid['chunker_types']))
    
    print(f"🧮 Total combinations to test: {total_combinations}")
    combination_count = 0
    
    for chunk_size in parameter_grid['chunk_sizes']:
        for chunk_overlap in parameter_grid['chunk_overlaps']:
            for k in parameter_grid['k_values']:
                for embedder_type in parameter_grid['embedder_types']:
                    for chunker_type in parameter_grid['chunker_types']:
                        combination_count += 1
                        print(f"\n🔄 Testing combination {combination_count}/{total_combinations}")
                        
                        try:
                            result = do_evaluate_with_params(
                                document_text, chunk_size, test_data, 
                                chunk_overlap, k, embedder_type, chunker_type
                            )
                            results.append(result)
                        except Exception as e:
                            print(f"❌ Error with combination: {e}")
                            continue
    
    # Create summary
    print("\n📈 PARAMETER COMPARISON RESULTS")
    print("="*80)
    
    summary_data = []
    for result in results:
        if result['accuracy_results']:
            summary_data.append({
                'Chunk Size': result['chunk_size'],
                'Chunk Overlap': result['chunk_overlap'],
                'K': result['k'],
                'Embedder': result['embedder_type'],
                'Chunker': result['chunker_type'],
                'Number of Chunks': result['num_chunks'],
                'Average Accuracy': result['accuracy_results']['average_score']
            })
    
    # Sort by accuracy
    summary_data.sort(key=lambda x: x['Average Accuracy'], reverse=True)
    
    print("🏆 Top 5 configurations:")
    for i, config in enumerate(summary_data[:5], 1):
        print(f"{i}. Size:{config['Chunk Size']}, Overlap:{config['Chunk Overlap']}, "
              f"K:{config['K']}, Embedder:{config['Embedder']}, Chunker:{config['Chunker']} "
              f"→ Accuracy: {config['Average Accuracy']:.3f}")
    
    return {
        'results': results,
        'summary': summary_data,
        'test_data': test_data,
        'parameter_grid': parameter_grid
    }


def do_load_document(file_path):
    """Load a text document from the specified file path."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        print(f"✅ Document loaded successfully!")
        print(f"📄 Document length: {len(content)} characters")
        return content
    except FileNotFoundError:
        print(f"❌ Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"❌ Error loading document: {e}")
        return None


def do_export_results_to_csv(comparison_results, filename="rag_parameter_results.csv"):
    """Export comparison results to CSV file."""
    if comparison_results and comparison_results.get('summary'):
        df = pd.DataFrame(comparison_results['summary'])
        df.to_csv(filename, index=False)
        print(f"📊 Results exported to {filename}")
        return df
    else:
        print("❌ No results to export")
        return None


def do_analyze_results(comparison_results):
    """Analyze and provide insights from comparison results."""
    if not comparison_results or not comparison_results.get('summary'):
        print("❌ No results to analyze")
        return None
    
    df = pd.DataFrame(comparison_results['summary'])
    
    print("📈 RESULTS ANALYSIS")
    print("="*50)
    
    # Basic statistics
    print(f"📊 Total configurations tested: {len(df)}")
    print(f"🎯 Best accuracy: {df['Average Accuracy'].max():.3f}")
    print(f"📉 Worst accuracy: {df['Average Accuracy'].min():.3f}")
    print(f"📊 Average accuracy: {df['Average Accuracy'].mean():.3f}")
    print(f"📏 Standard deviation: {df['Average Accuracy'].std():.3f}")
    
    # Best configuration
    best_config = df.loc[df['Average Accuracy'].idxmax()]
    print(f"\n🏆 BEST CONFIGURATION:")
    for col in df.columns:
        if col != 'Average Accuracy':
            print(f"   {col}: {best_config[col]}")
    print(f"   🎯 Accuracy: {best_config['Average Accuracy']:.3f}")
    
    # Parameter impact analysis
    if len(df) > 1:
        print(f"\n🔍 PARAMETER IMPACT ANALYSIS:")
        
        # Analyze chunk size impact
        if 'Chunk Size' in df.columns and df['Chunk Size'].nunique() > 1:
            chunk_impact = df.groupby('Chunk Size')['Average Accuracy'].agg(['mean', 'std', 'count'])
            print(f"📏 Chunk Size Impact:")
            for size, stats in chunk_impact.iterrows():
                print(f"   Size {size:>4}: μ={stats['mean']:.3f}, σ={stats['std']:.3f}, n={stats['count']}")
        
        # Analyze embedder impact
        if 'Embedder' in df.columns and df['Embedder'].nunique() > 1:
            embedder_impact = df.groupby('Embedder')['Average Accuracy'].agg(['mean', 'std', 'count'])
            print(f"🤖 Embedder Impact:")
            for embedder, stats in embedder_impact.iterrows():
                print(f"   {embedder:>12}: μ={stats['mean']:.3f}, σ={stats['std']:.3f}, n={stats['count']}")
        
        # Analyze chunker impact
        if 'Chunker' in df.columns and df['Chunker'].nunique() > 1:
            chunker_impact = df.groupby('Chunker')['Average Accuracy'].agg(['mean', 'std', 'count'])
            print(f"✂️  Chunker Impact:")
            for chunker, stats in chunker_impact.iterrows():
                print(f"   {chunker:>12}: μ={stats['mean']:.3f}, σ={stats['std']:.3f}, n={stats['count']}")
    
    return {
        'statistics': {
            'count': len(df),
            'best_accuracy': df['Average Accuracy'].max(),
            'worst_accuracy': df['Average Accuracy'].min(),
            'mean_accuracy': df['Average Accuracy'].mean(),
            'std_accuracy': df['Average Accuracy'].std()
        },
        'best_config': best_config.to_dict(),
        'dataframe': df
    }


# Example usage functions
def do_run_parameter_study(document_path="../module-1/alice_in_wonderland_book.txt", 
                          chunk_sizes=[200, 500, 1000, 1500, 2000],
                          chunk_overlap=50, k=3, embedder_type="openai", chunker_type="recursive"):
    """Run a complete RAG parameter study on the specified document."""
    print("🚀 Starting RAG Parameter Optimization Study")
    print("="*80)
    
    # Load document
    document_text = do_load_document(document_path)
    if not document_text:
        return None
    
    # Create test dataset
    test_data = do_create_test_dataset()
    
    # Compare different parameters
    results = do_compare_rag_parameters(
        document_text, 
        chunk_sizes=chunk_sizes, 
        test_data=test_data,
        chunk_overlap=chunk_overlap,
        k=k,
        embedder_type=embedder_type,
        chunker_type=chunker_type
    )
    
    # Export results
    do_export_results_to_csv(results, f"rag_parameter_results_{embedder_type}_{chunker_type}.csv")
    
    return results


# Backward compatibility function
def do_run_chunk_size_study(document_path="../module-1/alice_in_wonderland_book.txt", 
                           chunk_sizes=[200, 500, 1000, 1500, 2000],
                           chunk_overlap=50, k=3, embedder_type="openai", chunker_type="recursive"):
    """Run a complete chunk size study on the specified document (backward compatibility)."""
    return do_run_parameter_study(document_path, chunk_sizes, chunk_overlap, k, embedder_type, chunker_type)


def do_run_comprehensive_study(document_path="../module-1/alice_in_wonderland_book.txt"):
    """Run a comprehensive parameter study comparing multiple configurations."""
    print("🚀 Starting Comprehensive RAG Parameter Study")
    print("="*80)
    
    # Load document
    document_text = do_load_document(document_path)
    if not document_text:
        return None
    
    # Create test dataset
    test_data = do_create_test_dataset()
    
    # Define parameter grid
    parameter_grid = {
        'chunk_sizes': [200, 500, 1000],
        'chunk_overlaps': [25, 50, 100],
        'k_values': [3, 5],
        'embedder_types': ['openai'],  # Add 'anthropic', 'huggingface' if available
        'chunker_types': ['recursive', 'character']  # Add 'semantic' if experimental package available
    }
    
    # Run comprehensive comparison
    results = do_compare_parameters(document_text, test_data, parameter_grid)
    
    # Export results
    do_export_results_to_csv(results, "comprehensive_rag_parameter_results.csv")
    
    return results


def do_run_embedder_comparison(document_path="../module-1/alice_in_wonderland_book.txt",
                              embedder_types=['openai']):
    """Compare different embedder types."""
    print("🚀 Starting Embedder Comparison Study")
    print("="*80)
    
    # Load document
    document_text = do_load_document(document_path)
    if not document_text:
        return None
    
    # Create test dataset
    test_data = do_create_test_dataset()
    
    results = []
    
    for embedder_type in embedder_types:
        print(f"\n🔍 Testing {embedder_type} embedder...")
        try:
            result = do_evaluate_with_params(
                document_text, 
                chunk_size=500, 
                test_data=test_data,
                embedder_type=embedder_type
            )
            results.append(result)
        except Exception as e:
            print(f"❌ Error with {embedder_type}: {e}")
    
    # Print comparison
    print("\n📊 EMBEDDER COMPARISON RESULTS")
    print("="*60)
    for result in results:
        if result['accuracy_results']:
            print(f"🔸 {result['embedder_type']:>12}: Accuracy = {result['accuracy_results']['average_score']:.3f}")
    
    return results


def do_run_chunker_comparison(document_path="../module-1/alice_in_wonderland_book.txt",
                             chunker_types=['recursive', 'character']):
    """Compare different chunker types."""
    print("🚀 Starting Chunker Comparison Study")
    print("="*80)
    
    # Load document
    document_text = do_load_document(document_path)
    if not document_text:
        return None
    
    # Create test dataset
    test_data = do_create_test_dataset()
    
    results = []
    
    for chunker_type in chunker_types:
        print(f"\n🔍 Testing {chunker_type} chunker...")
        try:
            result = do_evaluate_with_params(
                document_text, 
                chunk_size=500, 
                test_data=test_data,
                chunker_type=chunker_type
            )
            results.append(result)
        except Exception as e:
            print(f"❌ Error with {chunker_type}: {e}")
    
    # Print comparison
    print("\n📊 CHUNKER COMPARISON RESULTS")
    print("="*60)
    for result in results:
        if result['accuracy_results']:
            print(f"🔸 {result['chunker_type']:>12}: Accuracy = {result['accuracy_results']['average_score']:.3f}")
    
    return results


 