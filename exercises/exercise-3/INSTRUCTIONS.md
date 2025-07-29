## Exercise 3: Chat-Based Test Result Categorization System

In this exercise, you will build an interactive chat system that leverages an internal database of existing test results to answer user queries and predict test outcomes. Follow the steps below to implement the system:

1. **Reuse the `chat_categorize_issues.py` file.**

2. **Store the `privacy_issues.json` data in a vector database.**

3. **When a user submits a query:**
   - Search the vector database for relevant test results that match or closely resemble the requested features.

4. **Construct a prompt for the LLM that includes:**
   - The retrieved relevant test results,
   - The user's query,
   - An instruction to categorize the issue as either Low, Medium, or High.

5. **Invoke the LLM with the constructed prompt.**

6. **Display the categorization result to the user by printing it.**

