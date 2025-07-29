#!/usr/bin/env python3
"""
Exercise 2: Advanced RAG Evaluation with Hard Questions

Objectives: 
- Learn how to optimize RAG parameters

This exercise uses hard questions with specific details from Alice in Wonderland
and evaluates RAG responses using the RAGAS Answer Accuracy metric.
"""

import sys
import os
import json
import asyncio

# Add module-1 and module-2 to path to import utility functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'module-1'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'module-2'))

from utility import (
    do_load_document,
    do_chunk_text, 
    do_create_vector_store,
    do_rag_query
)

from utility_2 import do_measure_accuracy_with_ragas


def do_load_hard_questions():
    """Load hard questions from the JSON file."""
    questions_file = os.path.join(os.path.dirname(__file__), 'hard_quetions.json')
    
    try:
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
        
        print(f"✅ Loaded {len(questions_data)} hard questions")
        
        # Show preview of questions
        for i, q in enumerate(questions_data[:3], 1):
            print(f"🔸 Q{i}: {q['question'][:80]}...")
        
        if len(questions_data) > 3:
            print(f"   ... and {len(questions_data) - 3} more questions")
        
        return questions_data
        
    except FileNotFoundError:
        print(f"❌ Hard questions file not found: {questions_file}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON file: {e}")
        return None
    except Exception as e:
        print(f"❌ Error loading hard questions: {e}")
        return None


def do_run_with_rag():
    """Run the hard questions with RAG using Alice in Wonderland context and evaluate with RAGAS."""
    print("\n✅ Running Hard Questions with RAG (Alice in Wonderland Context)")
    print("="*80)
    
    # Load hard questions
    questions_data = do_load_hard_questions()
    if not questions_data:
        print("❌ Could not load hard questions!")
        return None
    
    # Load and prepare the Alice in Wonderland book
    print("\n📖 Loading Alice in Wonderland...")
    book_path = os.path.join(os.path.dirname(__file__), '..', 'module-1', 'alice_in_wonderland_book.txt')
    document_content = do_load_document(book_path)
    
    if not document_content:
        print("❌ Could not load the book!")
        return None
    
    # Chunk the text
    print("\n✂️ Chunking the book...")
    # TODO: fix
    chunks = 
    
    # Create vector store
    print("\n🗄️ Creating vector store...")
    # TODO: fix
    vector_store = 
    
    # Run RAG queries
    print(f"\n🔍 Running RAG queries for {len(questions_data)} hard questions...")
    
    responses_data = []
    test_data = []
    
    for i, question_item in enumerate(questions_data, 1):
        question = question_item['question']
        ground_truth = question_item['answer']
        
        print(f"\n{'='*20} RAG Query {i} {'='*20}")
        print(f"❓ Question: {question}")
        
        # Get RAG response
        result = do_rag_query(vector_store, question, k=3)
        
        # Prepare data for RAGAS evaluation
        responses_data.append({
            'response': result['response'],
            'retrieved_contexts': result.get('context', ''),
            'context': result.get('context', '')
        })
        
        test_data.append({
            'question': question,
            'reference': ground_truth  # Use the ground truth answer from JSON
        })
        
        print(f"🤖 RAG Response: {result['response']}")
        print(f"✅ Ground Truth: {ground_truth}")
        print("-"*80)
    
    # Evaluate with RAGAS
    print(f"\n📏 Evaluating {len(responses_data)} responses with RAGAS Answer Accuracy...")
    accuracy_results = asyncio.run(do_measure_accuracy_with_ragas(responses_data, test_data))
    
    if accuracy_results:
        print(f"\n📊 RAGAS EVALUATION RESULTS")
        print("="*60)
        print(f"🎯 Average Answer Accuracy: {accuracy_results['average_score']:.3f}")
        print(f"📊 Total Questions Evaluated: {accuracy_results['total_questions']}")
        
        # Show individual scores
        print(f"\n📋 Individual Question Scores:")
        for i, score in enumerate(accuracy_results['individual_scores'], 1):
            print(f"   Q{i:2d}: {score:.3f}")
        
        # Show best and worst performing questions
        scores = accuracy_results['individual_scores']
        if scores:
            best_idx = scores.index(max(scores))
            worst_idx = scores.index(min(scores))
            
            print(f"\n🏆 Best Performance:")
            print(f"   Q{best_idx + 1} (Score: {scores[best_idx]:.3f}): {questions_data[best_idx]['question'][:60]}...")
            
            print(f"\n📉 Lowest Performance:")
            print(f"   Q{worst_idx + 1} (Score: {scores[worst_idx]:.3f}): {questions_data[worst_idx]['question'][:60]}...")
    
    return {
        'questions_data': questions_data,
        'responses_data': responses_data,
        'test_data': test_data,
        'accuracy_results': accuracy_results
    }


def main():
    """Main exercise function."""
    print("🎯 Exercise 2: Advanced RAG Evaluation with Hard Questions")
    print("="*80)
    print("This exercise tests RAG performance on challenging Alice in Wonderland questions")
    print("and evaluates accuracy using the RAGAS framework with ground truth answers.\n")

    # Run RAG evaluation with hard questions
    results = do_run_with_rag()
    
    if results and results['accuracy_results']:
        accuracy = results['accuracy_results']['average_score']
        total_questions = results['accuracy_results']['total_questions']
        
        print(f"\n🎉 Exercise 2 Complete!")
        print(f"📊 Final Results: {accuracy:.3f} average accuracy on {total_questions} hard questions")
        
        # Provide performance interpretation
        if accuracy >= 1.0:
            print("🏅 Perfect! RAG system achieved flawless accuracy on all questions.")
        elif accuracy >= 0.9:
            print("🌟 Outstanding! RAG system performs exceptionally well on complex questions.")
        elif accuracy >= 0.8:
            print("✨ Excellent! RAG system performs very well on complex questions.")
        elif accuracy >= 0.7:
            print("✅ Very good performance! RAG system handles most complex questions with ease.")
        elif accuracy >= 0.6:
            print("👍 Good performance! RAG system handles most complex questions well.")
        elif accuracy >= 0.5:
            print("⚠️  Fair performance. RAG system answers about half the complex questions correctly.")
        elif accuracy >= 0.4:
            print("🔸 Moderate performance. RAG system struggles with some complex details.")
        else:
            print("📈 Room for improvement. Consider tuning chunking or retrieval parameters.")
    else:
        print("❌ Exercise completed but evaluation failed.")


if __name__ == "__main__":
    main()
