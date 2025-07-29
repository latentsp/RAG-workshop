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

def do_categorize_user_privacy_issue():
    #TODO: perform similarity search to get closest issues
    #TODO: plug similar issues into prompt and invoke an LLM to categorize
   
   category = 
    return category
if __name__ == "__main__":
    do_offline_store()
    do_get_user_input()
