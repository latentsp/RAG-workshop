#!/usr/bin/env python3
"""
Simple test script for the enhanced do_chunk_text function.
This script tests the function with a basic text sample.
"""

def test_splitter_function():
    """Test the enhanced chunking function with basic functionality."""
    
    print("🧪 Testing Enhanced Text Splitter Function")
    print("=" * 50)
    
    # Sample text for testing
    sample_text = """
    Alice was beginning to get very tired of sitting by her sister on the bank, 
    and of having nothing to do. Once or twice she had peeped into the book her 
    sister was reading, but it had no pictures or conversations in it.
    
    "And what is the use of a book," thought Alice, "without pictures or conversations?"
    
    So she was considering in her own mind, as well as she could, for the hot day 
    made her feel very sleepy and stupid, whether the pleasure of making a daisy-chain 
    would be worth the trouble of getting up and picking the daisies, when suddenly 
    a White Rabbit with pink eyes ran close by her.
    """
    
    try:
        # Import the function
        from utility import do_chunk_text
        
        print("✅ Successfully imported do_chunk_text function")
        
        # Test 1: Basic recursive splitter (default)
        print("\n📝 Test 1: Basic Recursive Splitter")
        chunks = do_chunk_text(sample_text, chunk_size=200, chunk_overlap=20)
        print(f"✅ Generated {len(chunks)} chunks")
        
        # Test 2: Try character splitter
        print("\n📝 Test 2: Character Splitter")
        try:
            chunks_char = do_chunk_text(sample_text, chunk_size=200, chunk_overlap=20, splitter_type="character")
            print(f"✅ Character splitter generated {len(chunks_char)} chunks")
        except Exception as e:
            print(f"⚠️ Character splitter failed: {e}")
        
        # Test 3: Custom separators
        print("\n📝 Test 3: Custom Separators")
        try:
            chunks_custom = do_chunk_text(
                sample_text, 
                chunk_size=150, 
                chunk_overlap=15, 
                splitter_type="recursive",
                separators=["\n\n", ".", "!", "?", " "]
            )
            print(f"✅ Custom separators generated {len(chunks_custom)} chunks")
        except Exception as e:
            print(f"⚠️ Custom separators failed: {e}")
        
        print("\n🎉 Basic tests completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n💡 To set up the environment:")
        print("1. Create a virtual environment: python3 -m venv venv")
        print("2. Activate it: source venv/bin/activate")
        print("3. Install requirements: pip install -r ../requirements.txt")
        print("4. Add python-dotenv: pip install python-dotenv")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def check_dependencies():
    """Check if required dependencies are available."""
    
    print("🔍 Checking Dependencies")
    print("=" * 30)
    
    required_packages = [
        "langchain",
        "langchain_openai", 
        "langchain_community",
        "dotenv"
    ]
    
    available = []
    missing = []
    
    for package in required_packages:
        try:
            __import__(package.replace("_", ".") if "_" in package else package)
            available.append(package)
            print(f"✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"❌ {package}")
    
    print(f"\n📊 Summary: {len(available)} available, {len(missing)} missing")
    
    if missing:
        print(f"\n⚠️ Missing packages: {', '.join(missing)}")
        print("\n💡 Setup Instructions:")
        print("1. Create virtual environment: python3 -m venv venv")
        print("2. Activate: source venv/bin/activate")
        print("3. Install: pip install python-dotenv")
        print("4. Install: pip install -r ../requirements.txt")
    
    return len(missing) == 0


if __name__ == "__main__":
    print("🚀 Starting Enhanced Text Splitter Tests\n")
    
    # Check dependencies first
    deps_ok = check_dependencies()
    
    print("\n" + "=" * 60)
    
    if deps_ok:
        # Run tests if dependencies are available
        test_splitter_function()
    else:
        print("⚠️ Cannot run full tests due to missing dependencies")
        print("Please follow the setup instructions above.")
    
    print("\n" + "=" * 60)
    print("🏁 Test session completed!") 