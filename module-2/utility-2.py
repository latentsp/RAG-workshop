import os
import asyncio
from dotenv import load_dotenv
from typing import List, Dict, Tuple
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Get your OpenAI API key from the environment
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment. Please set it in your .env file.")


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


def do_chunk_text_with_size(text, chunk_size=500, chunk_overlap=50):
    """Split text into overlapping chunks with specified size."""
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
    
    print(f"📊 Chunked text with size {chunk_size}: {len(chunks)} chunks")
    print(f"📏 Average chunk size: {sum(len(chunk) for chunk in chunks) / len(chunks):.0f} characters")
    
    return chunks


def do_create_vector_store_from_chunks(chunks):
    """Create embeddings and vector store from text chunks."""
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    
    # Create embeddings model
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    print("🔄 Creating vector store...")
    
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


def do_evaluate_with_chunk_size(document_text, chunk_size, test_data, k=3):
    """Evaluate RAG performance with a specific chunk size."""
    print(f"\n🔬 Evaluating with chunk size: {chunk_size}")
    print("="*60)
    
    # Step 1: Chunk the document
    chunks = do_chunk_text_with_size(document_text, chunk_size=chunk_size)
    
    # Step 2: Create vector store
    vector_store = do_create_vector_store_from_chunks(chunks)
    
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
        'num_chunks': len(chunks),
        'responses': responses_data,
        'accuracy_results': accuracy_results
    }


def do_compare_chunk_sizes(document_text, chunk_sizes=[200, 500, 1000, 1500], test_data=None):
    """Compare RAG accuracy across different chunk sizes."""
    if test_data is None:
        test_data = do_create_test_dataset()
    
    print("🏁 Starting chunk size comparison study")
    print(f"📊 Testing chunk sizes: {chunk_sizes}")
    print(f"❓ Number of test questions: {len(test_data)}")
    print("="*80)
    
    results = []
    
    for chunk_size in chunk_sizes:
        result = do_evaluate_with_chunk_size(document_text, chunk_size, test_data)
        results.append(result)
    
    # Create summary
    print("\n📈 CHUNK SIZE COMPARISON RESULTS")
    print("="*80)
    
    summary_data = []
    for result in results:
        if result['accuracy_results']:
            summary_data.append({
                'Chunk Size': result['chunk_size'],
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


def do_export_results_to_csv(comparison_results, filename="chunk_size_comparison.csv"):
    """Export comparison results to CSV file."""
    if comparison_results and comparison_results.get('summary'):
        df = pd.DataFrame(comparison_results['summary'])
        df.to_csv(filename, index=False)
        print(f"📊 Results exported to {filename}")
        return df
    else:
        print("❌ No results to export")
        return None


# Example usage function
def do_run_chunk_size_study(document_path="../module-1/alice_in_wonderland_book.txt"):
    """Run a complete chunk size study on the specified document."""
    print("🚀 Starting RAG Chunk Size Accuracy Study")
    print("="*80)
    
    # Load document
    document_text = do_load_document(document_path)
    if not document_text:
        return None
    
    # Create test dataset
    test_data = do_create_test_dataset()
    
    # Compare different chunk sizes
    results = do_compare_chunk_sizes(
        document_text, 
        chunk_sizes=[200, 500, 1000, 1500, 2000], 
        test_data=test_data
    )
    
    # Export results
    do_export_results_to_csv(results)
    
    return results


if __name__ == "__main__":
    # Run the study
    results = do_run_chunk_size_study() 