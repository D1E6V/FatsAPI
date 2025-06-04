import requests
import json
import argparse
import sys
from typing import Optional

# Set stdout to handle Unicode characters
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # For Python < 3.7
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def chat_with_llm(
    user_question: str,
    type_of_question: str,
    provider: Optional[str] = None,
    url: str = "http://localhost:8000/chat"
) -> None:
    """
    Send a chat request to the API and print the response.
    
    Args:
        user_question (str): The question to ask
        type_of_question (str): The type of question
        provider (Optional[str]): The LLM provider to use (default: None)
        url (str): The API endpoint URL (default: http://localhost:8000/chat)
    """
    # Prepare the payload
    payload = {
        "user_question": user_question,
        "type_of_question": type_of_question
    }
    
    # Add provider if specified
    if provider:
        payload["provider"] = provider
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        # Send the request
        print(f"\nSending request to {url}...")
        print(f"Question: {user_question}")
        print(f"Type: {type_of_question}")
        if provider:
            print(f"Provider: {provider}")
        
        response = requests.post(url, json=payload, headers=headers)
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            print("\n" + "=" * 80)
            print("LLM RESPONSE:")
            print("=" * 80)
            print(result["response"])
            print("=" * 80)
            
            if "provider" in result:
                print(f"Provider: {result['provider']}")
            if "question_type" in result:
                print(f"Question Type: {result['question_type']}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
            
    except requests.RequestException as e:
        print(f"Request error: {e}")
        
    except Exception as e:
        print(f"Unexpected error: {e}")

def main():
    """Parse command line arguments and call the chat function."""
    parser = argparse.ArgumentParser(description="Chat with an LLM via the FastAPI backend")
    
    parser.add_argument("question", help="The question to ask the LLM")
    parser.add_argument("--type", "-t", default="small_talk", 
                        help="The type of question (default: small_talk)")
    parser.add_argument("--provider", "-p", 
                        help="The LLM provider to use (openai, google, anthropic)")
    parser.add_argument("--url", "-u", default="http://localhost:8000/chat",
                        help="The API endpoint URL (default: http://localhost:8000/chat)")
    
    args = parser.parse_args()
    
    chat_with_llm(
        user_question=args.question,
        type_of_question=args.type,
        provider=args.provider,
        url=args.url
    )

if __name__ == "__main__":
    main()