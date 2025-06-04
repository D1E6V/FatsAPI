from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os
from typing import Optional
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Chat API",
    description="A FastAPI application that serves as a backend for a chatbot",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define request model
class ChatRequest(BaseModel):
    user_question: str
    type_of_question: str

# Define response model
class ChatResponse(BaseModel):
    response: str

# LLM API configuration
# API_KEY should be stored in environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCsgiiaFhc2YKUg7ACJmZFZG_S2nqjCWWA")  # Using the provided Gemini API key
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# Define system prompts based on question type
SYSTEM_PROMPTS = {
    "story_generator": "You are a creative storyteller. Create an engaging and imaginative story based on the user's request.",
    "code_explainer": "You are a programming expert. Explain the provided code in a clear and educational manner.",
    "recipe_suggester": "You are a culinary expert. Suggest recipes based on the ingredients mentioned by the user.",
    "schedule_lookup": "You are a TV guide assistant. Provide information about TV schedules based on the user's query.",
    "small_talk": "You are a friendly conversational assistant. Engage in casual conversation with the user.",
    # Default prompt for any other type
    "default": "You are a helpful assistant. Provide a relevant and informative response to the user's question."
}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a chat request and return a response from an LLM.
    
    Parameters:
    - user_question: The primary question or message from the user
    - type_of_question: A pre-classified category or type of the user's question
    
    Returns:
    - The raw response from the LLM
    """
    try:
        # Get the appropriate system prompt based on question type
        system_prompt = SYSTEM_PROMPTS.get(
            request.type_of_question, 
            SYSTEM_PROMPTS["default"]
        )
        
        # Log the incoming request
        logger.info(f"Received chat request - Type: {request.type_of_question}")
        
        # Prepare the payload for the LLM API
        payload = {
            "model": "gpt-3.5-turbo",  # Can be configured based on requirements
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.user_question}
            ],
            "temperature": 0.7
        }
        
        # Call the Google Gemini API
        async with httpx.AsyncClient() as client:
            # Prepare the payload for Gemini API
            gemini_payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": f"{system_prompt}\n\nUser question: {request.user_question}"}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.7,
                    "topP": 0.8,
                    "topK": 40,
                    "maxOutputTokens": 1024
                }
            }
            
            # Add API key as query parameter
            url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
            
            headers = {
                "Content-Type": "application/json"
            }
            
            try:
                response = await client.post(
                    url,
                    json=gemini_payload,
                    headers=headers,
                    timeout=30.0  # Adjust timeout as needed
                )
                
                # Check if the request was successful
                if response.status_code == 200:
                    result = response.json()
                    # Extract the content from the Gemini response
                    llm_response = result["candidates"][0]["content"]["parts"][0]["text"]
                    # This line is unreachable, removing it
                else:
                    # Log the error
                    logger.error(f"Gemini API error: {response.status_code} - {response.text}")
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Gemini API error: {response.text}"
                    )
            except Exception as e:
                logger.error(f"Error calling Gemini API: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error calling Gemini API: {str(e)}"
                )
        
        return {"response": llm_response}
                
    except httpx.RequestError as e:
        # Handle network errors
        logger.error(f"Network error when calling LLM API: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Network error when calling LLM API: {str(e)}"
        )
    
    except Exception as e:
        # Handle any other errors
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )

# Run the application using Uvicorn when executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)