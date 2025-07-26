"""
RAG Parameter Optimization Experiment Runner
==========================================

This file runs the RAG parameter optimization experiments using the utility functions.
Choose which experiment to run by uncommenting the relevant sections.
"""

from utility_2 import *

def main():
    """Run the selected experiments."""
    
    print("🚀 RAG Parameter Optimization Experiments")
    print("="*50)
    
    # Basic parameter study
    print("🎯 Running basic parameter study...")
    results = do_run_parameter_study()
    
    # Analyze results
    if results:
        print("\n📊 Analyzing results...")
        analysis = do_analyze_results(results)
    
    # # Advanced parameter study with custom parameters
    # print("\n🎯 Running advanced parameter study...")
    # results = do_run_parameter_study(
    #     chunk_sizes=[100, 300, 600, 1200],
    #     chunk_overlap=75,
    #     k=5,
    #     embedder_type="openai",
    #     chunker_type="recursive"
    # )
    
    # # Embedder comparison (requires additional API keys)
    # print("\n🎯 Running embedder comparison...")
    # results = do_run_embedder_comparison(embedder_types=['openai', 'huggingface'])
    
    # # Chunker comparison
    # print("\n🎯 Running chunker comparison...")
    # results = do_run_chunker_comparison(chunker_types=['recursive', 'character'])
    
    # # Question filtering examples
    # print("\n🎯 Running easy questions only...")
    # results = run_easy_questions_only()
    # 
    # print("\n🎯 Running character questions study...")
    # results = run_character_questions_study()
    
    # # Comprehensive parameter study (takes longer)
    # print("\n🎯 Running comprehensive parameter study...")
    # results = do_run_comprehensive_study()
    # if results:
    #     analysis = do_analyze_results(results)


def run_quick_comparison():
    """Run a quick comparison with just a few parameters."""
    print("\n⚡ Quick Parameter Comparison")
    print("="*30)
    
    results = do_run_parameter_study(
        chunk_sizes=[200, 500, 1000],
        chunk_overlap=50,
        k=3,
        embedder_type="openai",
        chunker_type="recursive",
        max_questions=5  # Only use first 5 questions for speed
    )
    
    if results:
        do_analyze_results(results)
    
    return results


def run_easy_questions_only():
    """Test with only easy questions to see baseline performance."""
    print("\n📚 Easy Questions Only Study")
    print("="*30)
    
    results = do_run_parameter_study(
        chunk_sizes=[500, 1000],
        difficulties=["easy"],  # Only easy questions
        max_questions=None  # Use all easy questions
    )
    
    if results:
        do_analyze_results(results)
    
    return results


def run_character_questions_study():
    """Test with only character-related questions."""
    print("\n👤 Character Questions Study") 
    print("="*30)
    
    results = do_run_parameter_study(
        chunk_sizes=[200, 500, 1000, 1500],
        categories=["character"],  # Only character questions
        max_questions=None
    )
    
    if results:
        do_analyze_results(results)
    
    return results


def run_overlap_study():
    """Study the impact of different chunk overlaps."""
    print("\n🔄 Chunk Overlap Impact Study")
    print("="*30)
    
    overlap_values = [25, 50, 100, 150]
    all_results = []
    
    for overlap in overlap_values:
        print(f"\n📊 Testing overlap: {overlap}")
        results = do_run_parameter_study(
            chunk_sizes=[500],  # Fixed chunk size
            chunk_overlap=overlap,
            k=3
        )
        all_results.append(results)
    
    return all_results


def run_k_value_study():
    """Study the impact of different k values (number of retrieved chunks)."""
    print("\n🔍 K-Value Impact Study")
    print("="*30)
    
    k_values = [1, 3, 5, 7]
    all_results = []
    
    for k in k_values:
        print(f"\n📊 Testing k: {k}")
        results = do_run_parameter_study(
            chunk_sizes=[500],  # Fixed chunk size
            chunk_overlap=50,   # Fixed overlap
            k=k
        )
        all_results.append(results)
    
    return all_results


def run_embedder_study():
    """Compare different embedder types."""
    print("\n🤖 Embedder Comparison Study")
    print("="*30)
    
    # Test available embedders
    embedder_types = ['openai']
    
    # Add other embedders if API keys are available
    try:
        import os
        if os.getenv("ANTHROPIC_API_KEY"):
            embedder_types.append('anthropic')
        # embedder_types.append('huggingface')  # Usually works without API key
    except:
        pass
    
    results = do_run_embedder_comparison(embedder_types=embedder_types)
    return results


def run_chunker_study():
    """Compare different chunker types."""
    print("\n✂️  Chunker Comparison Study")
    print("="*30)
    
    # Test available chunkers
    chunker_types = ['recursive', 'character']
    
    # Add semantic chunker if experimental package is available
    try:
        import langchain_experimental
        chunker_types.append('semantic')
    except ImportError:
        print("ℹ️  Semantic chunker not available (install langchain-experimental)")
    
    results = do_run_chunker_comparison(chunker_types=chunker_types)
    return results


if __name__ == "__main__":
    # Choose which experiment to run
    
    # Option 1: Run the main comprehensive experiment
    main()
    
    # Option 2: Run specific targeted studies (uncomment to run)
    # run_quick_comparison()
    # run_overlap_study()
    # run_k_value_study() 
    # run_embedder_study()
    # run_chunker_study()
    
    # Option 3: Run question filtering studies (uncomment to run)
    # run_easy_questions_only()
    # run_character_questions_study()
    
    print("\n✅ Experiments completed!") 