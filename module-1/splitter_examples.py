#!/usr/bin/env python3
"""
Examples demonstrating different text splitter types in the RAG system.
This script shows how to use the enhanced do_chunk_text function with various splitters.
"""

from utility import do_load_document, do_chunk_text

def demonstrate_splitters():
    """Demonstrate different text splitter types with sample text."""
    
    # Load the Alice in Wonderland text
    print("📚 Loading Alice in Wonderland text...")
    text = do_load_document("alice_in_wonderland_book.txt")
    
    if not text:
        print("❌ Could not load document. Using sample text instead.")
        text = """
        Alice was beginning to get very tired of sitting by her sister on the bank, 
        and of having nothing to do. Once or twice she had peeped into the book her 
        sister was reading, but it had no pictures or conversations in it.
        
        "And what is the use of a book," thought Alice, "without pictures or conversations?"
        
        So she was considering in her own mind, as well as she could, for the hot day 
        made her feel very sleepy and stupid, whether the pleasure of making a daisy-chain 
        would be worth the trouble of getting up and picking the daisies, when suddenly 
        a White Rabbit with pink eyes ran close by her.
        
        There was nothing so very remarkable in that; nor did Alice think it so very much 
        out of the way to hear the Rabbit say to itself, "Oh dear! Oh dear! I shall be late!"
        """
    
    # Take just the first 2000 characters for demonstration
    sample_text = text[:2000]
    
    print("\n" + "="*80)
    print("🔧 DEMONSTRATING DIFFERENT TEXT SPLITTERS")
    print("="*80)
    
    # 1. Recursive Character Splitter (default)
    print("\n1️⃣ RECURSIVE CHARACTER SPLITTER (Default)")
    print("-" * 50)
    chunks_recursive = do_chunk_text(sample_text, chunk_size=300, chunk_overlap=50, splitter_type="recursive")
    
    # 2. Character Splitter
    print("\n2️⃣ CHARACTER SPLITTER")
    print("-" * 50)
    chunks_character = do_chunk_text(sample_text, chunk_size=300, chunk_overlap=50, splitter_type="character")
    
    # 3. Token Splitter
    print("\n3️⃣ TOKEN SPLITTER")
    print("-" * 50)
    chunks_token = do_chunk_text(sample_text, chunk_size=100, chunk_overlap=20, splitter_type="token")
    
    # 4. Semantic Splitter (experimental)
    print("\n4️⃣ SEMANTIC SPLITTER (Experimental)")
    print("-" * 50)
    try:
        chunks_semantic = do_chunk_text(sample_text, splitter_type="semantic")
        print("✅ Semantic splitter worked!")
    except Exception as e:
        print(f"⚠️ Semantic splitter failed: {e}")
        chunks_semantic = []
    
    # 5. Custom separators with recursive splitter
    print("\n5️⃣ RECURSIVE SPLITTER WITH CUSTOM SEPARATORS")
    print("-" * 50)
    custom_separators = ["\n\n", ".", "!", "?", " "]
    chunks_custom = do_chunk_text(
        sample_text, 
        chunk_size=250, 
        chunk_overlap=30, 
        splitter_type="recursive",
        separators=custom_separators
    )
    
    # Summary comparison
    print("\n" + "="*80)
    print("📊 COMPARISON SUMMARY")
    print("="*80)
    
    splitter_results = [
        ("Recursive", chunks_recursive),
        ("Character", chunks_character),
        ("Token", chunks_token),
        ("Semantic", chunks_semantic),
        ("Custom Recursive", chunks_custom)
    ]
    
    for name, chunks in splitter_results:
        if chunks:
            avg_length = sum(len(chunk) for chunk in chunks) / len(chunks)
            print(f"{name:15}: {len(chunks):2d} chunks, avg length: {avg_length:6.1f} chars")
        else:
            print(f"{name:15}: Failed or not available")


def demonstrate_html_splitter():
    """Demonstrate HTML header splitter with sample HTML."""
    
    print("\n" + "="*80)
    print("🌐 DEMONSTRATING HTML SPLITTERS")
    print("="*80)
    
    sample_html = """
    <html>
    <body>
        <h1>Chapter 1: Introduction</h1>
        <p>This is the introduction to our story. Alice was sitting by the river when she saw something unusual.</p>
        
        <h2>The White Rabbit</h2>
        <p>Suddenly, a White Rabbit with pink eyes ran close by her. This was very strange indeed!</p>
        
        <h3>Down the Rabbit Hole</h3>
        <p>Alice decided to follow the rabbit down a large rabbit hole under the hedge.</p>
        
        <h2>The Fall</h2>
        <p>The fall seemed to take a very long time. Alice had plenty of time to look about her and to wonder what was going to happen next.</p>
        
        <h1>Chapter 2: Wonderland</h1>
        <p>Finally, Alice reached the bottom and found herself in a strange new world.</p>
    </body>
    </html>
    """
    
    # HTML Header Splitter
    print("\n🏷️ HTML HEADER SPLITTER")
    print("-" * 50)
    chunks_html = do_chunk_text(sample_html, splitter_type="html_header")
    
    # Show the actual chunks with their structure
    print(f"\n📋 Generated {len(chunks_html)} chunks:")
    for i, chunk in enumerate(chunks_html, 1):
        print(f"\nChunk {i}:")
        print(f"Content: {chunk[:100]}{'...' if len(chunk) > 100 else ''}")


def demonstrate_code_splitter():
    """Demonstrate Python code splitter with sample code."""
    
    print("\n" + "="*80)
    print("🐍 DEMONSTRATING PYTHON CODE SPLITTER")
    print("="*80)
    
    sample_python_code = '''
def fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    """A simple calculator class."""
    
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        """Add two numbers."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def multiply(self, a, b):
        """Multiply two numbers."""
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result

def main():
    """Main function to demonstrate the calculator."""
    calc = Calculator()
    
    # Perform some calculations
    sum_result = calc.add(10, 20)
    product_result = calc.multiply(5, 6)
    
    print(f"Sum: {sum_result}")
    print(f"Product: {product_result}")
    print("History:", calc.history)

if __name__ == "__main__":
    main()
    '''
    
    print("\n🔧 PYTHON CODE SPLITTER")
    print("-" * 50)
    chunks_python = do_chunk_text(sample_python_code, chunk_size=400, chunk_overlap=50, splitter_type="python")
    
    print(f"\n📋 Generated {len(chunks_python)} code chunks:")
    for i, chunk in enumerate(chunks_python, 1):
        print(f"\nChunk {i}:")
        print(f"Preview: {chunk[:150]}{'...' if len(chunk) > 150 else ''}")


if __name__ == "__main__":
    print("🎯 Starting Text Splitter Demonstrations...")
    
    # Run all demonstrations
    demonstrate_splitters()
    demonstrate_html_splitter()
    demonstrate_code_splitter()
    
    print("\n" + "="*80)
    print("✅ All demonstrations completed!")
    print("💡 Try experimenting with different splitter types in your own code!")
    print("="*80) 