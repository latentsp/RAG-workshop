# RAG Parameter Optimization Utility (Module 2)

This module focuses on measuring and comparing RAG accuracy with different parameters using the **Answer Accuracy** metric from Ragas Nvidia metrics.

## 🚀 **Enhanced Features:**

### **1. Configurable Parameters:**
- **Chunk sizes**: Any list of sizes to test
- **Chunk overlap**: Configurable overlap between chunks
- **k values**: Number of chunks to retrieve
- **Document path**: Any document can be specified

### **2. Multiple Embedders:**
- **OpenAI**: Default embeddings (requires `OPENAI_API_KEY`)
- **Anthropic**: Anthropic embeddings (requires `ANTHROPIC_API_KEY`)
- **HuggingFace**: Free sentence transformers

### **3. Various Chunking Strategies:**
- **Recursive**: RecursiveCharacterTextSplitter (default)
- **Semantic**: SemanticChunker (context-aware splitting)
- **Character**: Simple character-based splitting
- **NLTK**: Natural language toolkit splitter
- **SpaCy**: Advanced NLP-based splitting

### **4. Study Functions:**

```python
# Basic chunk size study with custom parameters
results = do_run_chunk_size_study(
    document_path="path/to/document.txt",
    chunk_sizes=[200, 500, 1000, 1500],
    chunk_overlap=100,
    k=5,
    embedder_type="anthropic",
    chunker_type="semantic"
)

# Compare different embedders
results = do_run_embedder_comparison(
    embedder_types=['openai', 'anthropic', 'huggingface']
)

# Compare different chunkers
results = do_run_chunker_comparison(
    chunker_types=['recursive', 'semantic', 'character']
)

# Comprehensive parameter grid search
results = do_run_comprehensive_study()
```

### **5. Advanced Analysis:**
- **Parameter impact analysis** showing which factors affect accuracy most
- **Statistical summaries** with mean, std, best/worst configurations
- **CSV export** with detailed results
- **Automated best configuration detection**

### **6. Key Functions:**
- `do_evaluate_with_params()` - Evaluate with full parameter control
- `do_compare_parameters()` - Grid search across parameter combinations
- `do_analyze_results()` - Statistical analysis and insights
- `do_get_embedder()` - Factory for different embedding models
- `do_create_chunker()` - Factory for different chunking strategies

## 📊 **Usage Examples:**

```python
# Test chunk overlap impact
results = do_run_chunk_size_study(
    chunk_sizes=[500],
    chunk_overlap=25,  # vs 50, 100, 150
    k=3
)

# Compare semantic vs recursive chunking
results = do_run_chunker_comparison(
    chunker_types=['recursive', 'semantic']
)

# Test retrieval parameters
results = do_evaluate_with_params(
    document_text, 
    chunk_size=500,
    test_data,
    k=7  # Retrieve more chunks
)
```

## 🛠️ **Installation Requirements:**

### Required packages:
```bash
pip install ragas langchain langchain-openai langchain-community pandas python-dotenv
```

### Optional packages for additional features:
```bash
# For Anthropic embeddings
pip install langchain-anthropic

# For HuggingFace embeddings
pip install langchain-huggingface

# For semantic chunking
pip install langchain-experimental

# For NLTK chunking
pip install nltk

# For SpaCy chunking
pip install spacy
python -m spacy download en_core_web_sm
```

## 🔧 **Setup:**

1. Create a `.env` file in your project root:
```env
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here  # Optional
```

2. Ensure you have the Alice in Wonderland document at `../module-1/alice_in_wonderland_book.txt`

## 🎯 **Quick Start:**

```python
from utility-2 import do_run_chunk_size_study

# Run basic chunk size comparison
results = do_run_chunk_size_study()

# Run with custom parameters
results = do_run_chunk_size_study(
    chunk_sizes=[200, 400, 800, 1600],
    chunk_overlap=75,
    k=5,
    embedder_type="openai",
    chunker_type="recursive"
)
```

## 📈 **Understanding Results:**

The utility will output:
- **Individual accuracy scores** for each question
- **Average accuracy** across all test questions
- **Best performing configuration** with highest accuracy
- **Parameter impact analysis** showing which factors matter most
- **CSV exports** for further analysis

Example output:
```
🏆 Best performing chunk size: 500 (Accuracy: 0.850)

📊 PARAMETER IMPACT ANALYSIS:
📏 Chunk Size Impact:
   Size  200: μ=0.760, σ=0.120, n=1
   Size  500: μ=0.850, σ=0.000, n=1
   Size 1000: μ=0.820, σ=0.000, n=1
```

## 🎓 **Workshop Applications:**

This utility demonstrates how different parameters affect RAG accuracy, showing the real impact of:
- **Chunk size optimization** - Finding the sweet spot between context and precision
- **Overlap strategies** - Balancing information retention with computational cost
- **Embedder selection** - Comparing different semantic understanding models
- **Chunking methods** - From simple splitting to semantic-aware chunking
- **Retrieval parameters** - Optimizing how much context to provide

Perfect for teaching RAG optimization concepts with concrete, measurable results! 🎯 