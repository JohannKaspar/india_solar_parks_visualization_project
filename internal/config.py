"""\
This module contains the Config class which is a singleton class that loads the environment variables from the .env file.
Usage: 
"""

from typing import List, Union
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    LLM_MODEL: str
    LLM_TEMP: str
    EMBEDDING_MODEL: str
    SUPABASE_URL: str
    SUPABASE_KEY: str
    # SERPENT_API_KEY: str
    # GOOGLE_API_KEY: str
    # GOOGLE_CSE_ID: str
    # SERPER_API_KEY: str
    PHOENIX_COLLECTOR_ENDPOINT: str
    BRAVE_SEARCH_API_KEY: str

    class Config:
        case_sensitive = True
        env_file = "app/.env"


settings = Settings()
