from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from enum import Enum
import httpx
import os
import logging
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from functools import lru_cache

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Advanced Chat API",
    description="A FastAPI application that serves as a backend for a chatbot with multiple LLM provider support",
    version="1.0.0"
)

# Define LLM providers
class LLMProvider(str, Enum):
    OPENAI = "openai"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"

# Define question types
class QuestionType(str, Enum):
    STORY_GENERATOR = "story_generator"
    CODE_EXPLAINER = "code_explainer"
    RECIPE_SUGGESTER = "recipe_suggester"
    SCHEDULE_LOOKUP = "schedule_lookup"
    SMALL_TALK = "small_talk"
    OTHER = "other"

# Define request model
class ChatRequest(BaseModel):
    user_question: str = Field(..., description="The primary question or message from the user")
    type_of_question: QuestionType = Field(..., description="A pre-classified category or type of the user's question")
    provider: Optional[LLMProvider] = Field(default=LLMProvider.OPENAI, description="The LLM provider to use")

# Define response model
class ChatResponse(BaseModel):
    response: str
    provider: LLMProvider
    question_type: QuestionType

# Configuration class
class Settings:
    # API Keys (should be stored in environment variables)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY_HERE")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "YOUR_ANTHROPIC_API_KEY_HERE")
    
    # API URLs
    OPENAI_API_URL: str = "https://api.openai.com/v1/chat/completions"
    GOOGLE_API_URL: str = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
    ANTHROPIC_API_URL: str = "https://api.anthropic.com/v1/messages"
    
    # Default models
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    GOOGLE_MODEL: str = "gemini-pro"
    ANTHROPIC_MODEL: str = "claude-3-sonnet-20240229"

@lru_cache()
def get_settings():
    return Settings()

# Define system prompts based on question type
SYSTEM_PROMPTS = {
    QuestionType.STORY_GENERATOR: "You are a creative storyteller. Create an engaging and imaginative story based on the user's request.",
    QuestionType.CODE_EXPLAINER: "You are a programming expert. Explain the provided code in a clear and educational manner.",
    QuestionType.RECIPE_SUGGESTER: "You are a culinary expert. Suggest recipes based on the ingredients mentioned by the user.",
    QuestionType.SCHEDULE_LOOKUP: "You are a TV guide assistant. Provide information about TV schedules based on the user's query.",
    QuestionType.SMALL_TALK: "You are a friendly conversational assistant. Engage in casual conversation with the user.",
    QuestionType.OTHER: "You are a helpful assistant. Provide a relevant and informative response to the user's question."
}

# LLM API client class
class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def close(self):
        await self.client.aclose()
    
    async def query_openai(self, system_prompt: str, user_question: str) -> str:
        """Query the OpenAI API and return the response."""
        payload = {
            "model": self.settings.OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ],
            "temperature": 0.7
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.settings.OPENAI_API_KEY}"
        }
        
        response = await self.client.post(
            self.settings.OPENAI_API_URL,
            json=payload,
            headers=headers
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"OpenAI API error: {response.text}"
            )
    
    async def query_google(self, system_prompt: str, user_question: str) -> str:
        """Query the Google Gemini API and return the response."""
        # Combine system prompt and user question for Gemini
        combined_prompt = f"{system_prompt}\n\nUser question: {user_question}"
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": combined_prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.7
            }
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Add API key as query parameter
        url = f"{self.settings.GOOGLE_API_URL}?key={self.settings.GOOGLE_API_KEY}"
        
        response = await self.client.post(
            url,
            json=payload,
            headers=headers
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Google API error: {response.text}"
            )
    
    async def query_anthropic(self, system_prompt: str, user_question: str) -> str:
        """Query the Anthropic API and return the response."""
        payload = {
            "model": self.settings.ANTHROPIC_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ],
            "max_tokens": 1000
        }
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.settings.ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01"
        }
        
        response = await self.client.post(
            self.settings.ANTHROPIC_API_URL,
            json=payload,
            headers=headers
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["content"][0]["text"]
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Anthropic API error: {response.text}"
            )
    
    async def query_llm(self, provider: LLMProvider, system_prompt: str, user_question: str) -> str:
        """Query the specified LLM provider and return the response."""
        if provider == LLMProvider.OPENAI:
            return await self.query_openai(system_prompt, user_question)
        elif provider == LLMProvider.GOOGLE:
            return await self.query_google(system_prompt, user_question)
        elif provider == LLMProvider.ANTHROPIC:
            return await self.query_anthropic(system_prompt, user_question)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported LLM provider: {provider}"
            )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, settings: Settings = Depends(get_settings)):
    """
    Process a chat request and return a response from an LLM.
    
    Parameters:
    - user_question: The primary question or message from the user
    - type_of_question: A pre-classified category or type of the user's question
    - provider: (Optional) The LLM provider to use (default: openai)
    
    Returns:
    - The raw response from the LLM
    """
    try:
        # Get the appropriate system prompt based on question type
        system_prompt = SYSTEM_PROMPTS.get(
            request.type_of_question, 
            SYSTEM_PROMPTS[QuestionType.OTHER]
        )
        
        # Log the incoming request
        logger.info(f"Received chat request - Type: {request.type_of_question}, Provider: {request.provider}")
        
        # Create LLM client
        llm_client = LLMClient(settings)
        
        try:
            # Query the LLM
            llm_response = await llm_client.query_llm(
                request.provider,
                system_prompt,
                request.user_question
            )
            
            return {
                "response": llm_response,
                "provider": request.provider,
                "question_type": request.type_of_question
            }
            
        finally:
            # Close the client
            await llm_client.close()
                
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
    uvicorn.run("advanced_main:app", host="0.0.0.0", port=8000, reload=True)