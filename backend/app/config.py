from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # MongoDB Settings
    MONGO_URL: str
    DB_NAME: str = "live_db"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

