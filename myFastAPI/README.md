# FastAPI Chat Application

A Python web application using the FastAPI framework that serves as a backend for a chatbot. It includes a single API endpoint designed for chat interactions.

## Features

- Single API endpoint (`/chat`) for processing chat requests
- Forwards user questions to a Large Language Model (LLM)
- Customizes prompts based on the type of question
- Returns raw LLM responses

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file based on the `.env.example` template and add your LLM API key(s)

## Usage

1. Start the server:
   ```
   uvicorn main:app --reload
   ```
   
2. The API will be available at `http://localhost:8000`

3. You can access the interactive API documentation at `http://localhost:8000/docs`

## API Endpoint

### POST /chat

Process a chat request and return a response from an LLM.

#### Request Body

```json
{
  "user_question": "string",
  "type_of_question": "string"
}
```

- `user_question`: The primary question or message from the user
- `type_of_question`: A pre-classified category or type of the user's question (e.g., "story_generator", "code_explainer", "recipe_suggester", "schedule_lookup", "small_talk")

#### Response

```json
{
  "response": "string"
}
```

- `response`: The raw response from the LLM

## Examples

### Example 1: Story Generation

Request:
```json
{
  "user_question": "Tell me a short story about a brave knight named Lancelot who fights a giant in the land of Avalon.",
  "type_of_question": "story_generator"
}
```

### Example 2: Code Explanation

Request:
```json
{
  "user_question": "Explain the following Python code snippet:\n\n```python\ndef factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)\n```",
  "type_of_question": "code_explainer"
}
```

### Example 3: Recipe Suggestion

Request:
```json
{
  "user_question": "I have chicken, pasta, and tomatoes. Suggest a simple recipe I can make.",
  "type_of_question": "recipe_suggester"
}
```

### Example 4: Schedule Lookup

Request:
```json
{
  "user_question": "What's on BBC One at 8 PM?",
  "type_of_question": "schedule_lookup"
}
```

### Example 5: Small Talk

Request:
```json
{
  "user_question": "Tell me a joke.",
  "type_of_question": "small_talk"
}