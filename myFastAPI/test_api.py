import requests
import json
import sys

def test_chat_endpoint(user_question, type_of_question):
    """
    Test the chat endpoint with the given parameters.
    
    Args:
        user_question (str): The question to ask
        type_of_question (str): The type of question
        
    Returns:
        None: Prints the response from the API
    """
    url = "http://localhost:8000/chat"
    
    payload = {
        "user_question": user_question,
        "type_of_question": type_of_question
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            print("\nAPI Response:")
            print("-" * 80)
            print(result["response"])
            print("-" * 80)
        else:
            print(f"Error: {response.status_code} - {response.text}")
            
    except requests.RequestException as e:
        print(f"Request error: {e}")
        
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    # Check if command line arguments are provided
    if len(sys.argv) >= 3:
        user_question = sys.argv[1]
        type_of_question = sys.argv[2]
    else:
        # Default test case
        user_question = "Tell me a joke."
        type_of_question = "small_talk"
        print("No arguments provided. Using default test case.")
    
    print(f"Testing chat endpoint with:")
    print(f"User Question: {user_question}")
    print(f"Type of Question: {type_of_question}")
    
    test_chat_endpoint(user_question, type_of_question)