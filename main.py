import os
import re
from typing import Any, Dict, List, Optional

import aiohttp
import orjson
import logging
from aiohttp import ClientError, ClientSession, TCPConnector
from aiohttp_retry import ExponentialRetry, RetryClient
from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure standard logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("my-logger")

# Configuration using BaseModel with environment variables
class Settings(BaseModel):
    CLOUDFLARE_AI_GATEWAY_URL: str = os.getenv("CLOUDFLARE_AI_GATEWAY_URL")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    PERSON_ENRICHMENT_API_KEY: str = os.getenv("PERSON_ENRICHMENT_API_KEY")

    class Config:
        case_sensitive = False
        env_file = ".env"  # Ensure you have a .env file with your environment variables

settings = Settings()

# Initialize FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Pydantic model for form validation
class LifeStoryForm(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    location: Optional[str] = Field(None, max_length=100)

# Ensure logger is properly shut down on application shutdown
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down application.")

# Serve robots.txt and ads.txt
@app.get("/robots.txt")
async def read_robots():
    logger.info("Serving robots.txt")
    return FileResponse("static/robots.txt")

@app.get("/ads.txt")
async def read_ads():
    logger.info("Serving ads.txt")
    return FileResponse("static/ads.txt")

@app.get("/", response_class=HTMLResponse)
async def home_get(request: Request):
    logger.info("Serving home page.")
    return templates.TemplateResponse("home.html", {"request": request, "story": "", "error": ""})

# Dependency to get HTTP client session
async def get_client_session() -> ClientSession:
    logger.info("Creating new ClientSession.")
    connector = TCPConnector(limit=100)
    session = ClientSession(connector=connector)
    try:
        yield session
    finally:
        logger.info("Closing ClientSession.")
        await session.close()

# Repository interface for Person Enrichment API
class PersonEnrichmentRepositoryInterface:
    async def enrich_person(self, params: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

# Repository implementation for Person Enrichment API
class PersonEnrichmentRepository(PersonEnrichmentRepositoryInterface):
    def __init__(self, session: ClientSession):
        self.session = session
        self.retry_options = ExponentialRetry(attempts=3)
        logger.info("PersonEnrichmentRepository initialized.")

    async def enrich_person(self, params: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Enriching person with params: {params}")
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": settings.PERSON_ENRICHMENT_API_KEY  # Corrected header key
        }
        url = "https://api.peopledatalabs.com/v5/person/enrich"  # Ensure this is the correct endpoint
        retry_client = RetryClient(client_session=self.session, retry_options=self.retry_options)

        try:
            async with retry_client.post(url, headers=headers, json=params, ssl=True) as response:
                response_text = await response.text()
                logger.info(f"Response status: {response.status}, Response text: {response_text}")
                if response.status >= 400:
                    logger.error(f"HTTP error {response.status}: {response_text}")
                    raise HTTPException(status_code=response.status, detail="Error fetching person data.")
                response_json = await response.json(loads=orjson.loads)
                logger.info(f"Person enrichment response: {response_json}")
                return response_json
        except ClientError as e:
            logger.error(f"HTTP ClientError: {str(e)}")
            raise HTTPException(status_code=500, detail="Error communicating with Person Enrichment API.")

# Repository interface for LLM service
class LLMServiceRepositoryInterface:
    async def generate_story(self, data: Dict[str, Any]) -> str:
        raise NotImplementedError

# Repository implementation for LLM service
class LLMServiceRepository(LLMServiceRepositoryInterface):
    def __init__(self, session: ClientSession):
        self.session = session
        self.retry_options = ExponentialRetry(attempts=3)
        logger.info("LLMServiceRepository initialized.")

    async def generate_story(self, data: Dict[str, Any]) -> str:
        logger.info(f"Generating story with data: {data}")
        headers = {
            "Content-Type": "application/json",
            "authorization": f"Bearer {settings.OPENAI_API_KEY}"
        }
        url = f"{settings.CLOUDFLARE_AI_GATEWAY_URL}/openai/chat/completions"  # Adjust endpoint as needed
        payload = {
            "model": "chatgpt-4o-latest",
            "messages": self._create_story_messages(data),
            "temperature": 0.7,
            "max_tokens": 1000
        }
        retry_client = RetryClient(client_session=self.session, retry_options=self.retry_options)

        try:
            async with retry_client.post(url, headers=headers, json=payload, ssl=True) as response:
                response_text = await response.text()
                logger.info(f"Response status: {response.status}, Response text: {response_text}")
                if response.status >= 400:
                    logger.error(f"LLM HTTP error {response.status}: {response_text}")
                    raise HTTPException(status_code=response.status, detail="Error generating story.")
                response_json = await response.json(loads=orjson.loads)
                story = response_json["choices"][0]["message"]["content"].strip()
                logger.info(f"Generated story: {story}")
                return story
        except ClientError as e:
            logger.error(f"LLM ClientError: {str(e)}")
            raise HTTPException(status_code=500, detail="Error communicating with LLM service.")

    def _create_story_messages(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        # Craft the prompt based on enriched person data
        system_prompt = (
            "You are a creative writer tasked with creating a whimsical and engaging children's book-style life story. "
            "Use the following personal data to craft a narrative that is imaginative, heartwarming, and suitable for children."
        )
        # Serialize the person data for the prompt
        user_prompt = f"Create a children's book-style life story for the following person:\n\n{data}"
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

# Story generation service interface
class StoryGenerationServiceInterface:
    async def create_story(self, form_data: LifeStoryForm) -> str:
        raise NotImplementedError

# Story generation service implementation
class StoryGenerationService(StoryGenerationServiceInterface):
    def __init__(self, enrichment_repo: PersonEnrichmentRepositoryInterface, llm_repo: LLMServiceRepositoryInterface):
        self.enrichment_repo = enrichment_repo
        self.llm_repo = llm_repo
        logger.info("StoryGenerationService initialized.")

    async def create_story(self, form_data: LifeStoryForm) -> str:
        logger.info(f"Creating story for form data: {form_data}")
        # Prepare parameters for Person Enrichment API
        params = {
            "name": form_data.name,
        }
        if form_data.location:
            params["location"] = form_data.location

        # Fetch enriched person data
        enrichment_response = await self.enrichment_repo.enrich_person(params)

        logger.info(f"Enrichment response: {enrichment_response}")

        if enrichment_response.get("status") != 200 or not enrichment_response.get("data"):
            logger.error("Person not found or insufficient data.")
            raise HTTPException(status_code=404, detail="Person not found or insufficient data.")

        likelihood = enrichment_response.get("likelihood", 0)
        person_data = enrichment_response["data"]

        if likelihood < 5:
            logger.error("Low confidence in match. Please provide more information for better results.")
            raise HTTPException(status_code=400, detail="Low confidence in match. Please provide more information for better results.")

        # Generate life story using LLM
        story = await self.llm_repo.generate_story(person_data)

        logger.info(f"Generated story: {story}")
        return story

# Dependency to get Person Enrichment repository
async def get_person_enrichment_repository(session: ClientSession = Depends(get_client_session)) -> PersonEnrichmentRepositoryInterface:
    logger.info("Getting PersonEnrichmentRepository.")
    return PersonEnrichmentRepository(session)

# Dependency to get LLM service repository
async def get_llm_service_repository(session: ClientSession = Depends(get_client_session)) -> LLMServiceRepositoryInterface:
    logger.info("Getting LLMServiceRepository.")
    return LLMServiceRepository(session)

# Dependency to get story generation service
async def get_story_generation_service(
    enrichment_repo: PersonEnrichmentRepositoryInterface = Depends(get_person_enrichment_repository),
    llm_repo: LLMServiceRepositoryInterface = Depends(get_llm_service_repository)
) -> StoryGenerationServiceInterface:
    logger.info("Getting StoryGenerationService.")
    return StoryGenerationService(enrichment_repo, llm_repo)

# POST endpoint for generating life story
@app.post("/", response_class=HTMLResponse)
async def generate_life_story(
    request: Request,
    name: str = Form(..., min_length=1, max_length=100),
    location: Optional[str] = Form(None, max_length=100),
    service: StoryGenerationServiceInterface = Depends(get_story_generation_service),
):
    try:
        logger.info(f"Received request to generate life story for name: {name}, location: {location}")
        if not name:
            error = "Name input is required."
            logger.error(error)
            return templates.TemplateResponse(
                "home.html", {"request": request, "story": "", "error": error}
            )

        # Generate the life story
        form_data = LifeStoryForm(name=name, location=location)
        story = await service.create_story(form_data)

        logger.info("Life story generated successfully.")
        return templates.TemplateResponse("home.html", {"request": request, "story": story, "error": ""})

    except HTTPException as e:
        logger.error(f"HTTPException: {e.detail}")
        return templates.TemplateResponse(
            "home.html", {"request": request, "story": "", "error": e.detail}
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return templates.TemplateResponse(
            "home.html", {"request": request, "story": "", "error": "An unexpected error occurred."}
        )
# Run the application locally if the script is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
