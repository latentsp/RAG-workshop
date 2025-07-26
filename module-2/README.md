# RAG Parameter Optimization Utility (Module 2)

This module focuses on measuring and comparing RAG accuracy with different parameters using the **Answer Accuracy** metric from Ragas Nvidia metrics.

## 📁 **File Structure:**

- **`utility-2.py`** - Core utility functions for RAG parameter optimization
- **`run_experiment.py`** - Experiment runner with predefined studies and custom experiments
- **`test_questions.json`** - External question bank with categories and difficulty levels
- **`visualize_results.ipynb`** - Jupyter notebook for visualizing experiment results
- **`README.md`** - This documentation file

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
# Basic parameter study with custom parameters
results = do_run_parameter_study(
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

### **5. External Question Bank:**
- **JSON-based question storage** for easy customization
- **Question filtering by category** (character, plot, setting)
- **Difficulty-based filtering** (easy, medium, hard)
- **Flexible question limits** (test with 5, 10, or all questions)
- **Custom question files** for different documents

### **6. Advanced Analysis:**
- **Parameter impact analysis** showing which factors affect accuracy most
- **Statistical summaries** with mean, std, best/worst configurations
- **CSV export** with detailed results
- **Automated best configuration detection**

### **7. Key Functions:**
- `do_evaluate_with_params()` - Evaluate with full parameter control
- `do_compare_rag_parameters()` - Compare different parameter configurations
- `do_compare_parameters()` - Grid search across parameter combinations
- `do_run_parameter_study()` - Run complete parameter optimization study
- `do_load_test_questions()` - Load and filter questions from external JSON file
- `do_analyze_results()` - Statistical analysis and insights
- `do_get_embedder()` - Factory for different embedding models
- `do_create_chunker()` - Factory for different chunking strategies

**Backward Compatibility:**
- `do_compare_chunk_sizes()` - Calls `do_compare_rag_parameters()`
- `do_run_chunk_size_study()` - Calls `do_run_parameter_study()`

## 📊 **Usage Examples:**

```python
# Test chunk overlap impact
results = do_run_parameter_study(
    chunk_sizes=[500],
    chunk_overlap=25,  # vs 50, 100, 150
    k=3
)

# Test with only easy questions
results = do_run_parameter_study(
    chunk_sizes=[200, 500, 1000],
    difficulties=["easy"],
    max_questions=None
)

# Test with character questions only
results = do_run_parameter_study(
    chunk_sizes=[500, 1000],
    categories=["character"],
    max_questions=5
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

3. (Optional) Customize the question file `test_questions.json`:
```json
{
  "document_info": {
    "title": "Your Document Questions",
    "description": "Custom questions for your document"
  },
  "questions": [
    {
      "id": 1,
      "question": "Your question here?",
      "reference": "Expected answer here",
      "category": "plot|character|setting|other",
      "difficulty": "easy|medium|hard"
    }
  ]
}
```

## 🎯 **Quick Start:**

### Method 1: Run Experiments (Recommended)
```bash
# Run the predefined experiments
python run_experiment.py
```

**Available experiment types in `run_experiment.py`:**
- **`main()`** - Basic parameter study with analysis
- **`run_quick_comparison()`** - Quick 3-parameter test
- **`run_overlap_study()`** - Impact of chunk overlap values
- **`run_k_value_study()`** - Impact of retrieval k values
- **`run_embedder_study()`** - Compare different embedders
- **`run_chunker_study()`** - Compare different chunking strategies
- **`run_easy_questions_only()`** - Test only with easy difficulty questions
- **`run_character_questions_study()`** - Test only with character-related questions

### Method 2: Import and Use Functions
```python
import importlib.util
import sys

# Import utility-2 module (handling hyphenated filename)
spec = importlib.util.spec_from_file_location("utility_2", "utility-2.py")
utility_2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utility_2)

# Run experiments
results = utility_2.do_run_parameter_study()

# Run with custom parameters
results = utility_2.do_run_parameter_study(
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
- **CSV exports** for further analysis (saved as `rag_parameter_results_*.csv`)

Example output:
```
🏆 Best performing chunk size: 500 (Accuracy: 0.850)

📊 PARAMETER IMPACT ANALYSIS:
📏 Chunk Size Impact:
   Size  200: μ=0.760, σ=0.120, n=1
   Size  500: μ=0.850, σ=0.000, n=1
   Size 1000: μ=0.820, σ=0.000, n=1
```

## 📊 **Results Visualization:**

Use the Jupyter notebook to create comprehensive visualizations:

```bash
jupyter notebook visualize_results.ipynb
```

**The notebook includes:**
- **Data overview and statistics** 
- **Accuracy vs chunk size plots** (line and bar charts)
- **Chunk count correlation analysis** 
- **Performance insights and recommendations**
- **Workshop-ready conclusions with expected patterns**

The visualization automatically loads your CSV results and creates professional charts perfect for workshop presentations!

## 🎓 **Workshop Applications:**

This utility demonstrates how different parameters affect RAG accuracy, showing the real impact of:
- **Chunk size optimization** - Finding the sweet spot between context and precision
- **Overlap strategies** - Balancing information retention with computational cost
- **Embedder selection** - Comparing different semantic understanding models
- **Chunking methods** - From simple splitting to semantic-aware chunking
- **Retrieval parameters** - Optimizing how much context to provide
- **Question complexity** - How difficulty affects parameter sensitivity
- **Content categories** - How different types of questions respond to optimization

## 🎯 **Benefits of External Questions:**

- **Flexibility**: Easy to add new questions without code changes
- **Categorization**: Test specific aspects (plot, characters, settings)
- **Difficulty scaling**: Compare performance across complexity levels  
- **Reusability**: Share question sets across different experiments
- **Workshop customization**: Tailor questions to specific learning objectives
- **Document adaptation**: Create question sets for different source materials

Perfect for teaching RAG optimization concepts with concrete, measurable results! 🎯 