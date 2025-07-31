def do_get_user_input():
    """Continuously get input from user and print it."""
    while True:
        try:
            user_input = input("Please enter something (or Ctrl+C to exit): ")
            do_categorize_user_privacy_issue()
        except KeyboardInterrupt:
            print("\nExiting...")
            break

def do_offline_store():
    #TODO: Store privacy_issues.json in vector DB

def do_online():
    #TODO: perform similarity search to get closest issues
    #TODO: plug similar issues into prompt and invoke an LLM to categorize
    user_input = do_get_user_input()
    task = "You are a helpful assistant that categorizes privacy issues."
    context =
    
    prompt = f"""
    {task}
    {context}
    {user_input}
    """
    category = llm.invoke(HumanMessage(content=prompt))
    print(category)

if __name__ == "__main__":
    do_offline_store()
    do_online()
