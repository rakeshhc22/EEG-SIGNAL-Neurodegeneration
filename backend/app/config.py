# backend/app/config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file (if present)
load_dotenv()

class Settings:
    PROJECT_NAME: str = "NeuroDetect API"
    VERSION: str = "1.0.0"

    # Database URL (SQLite by default, but can be swapped to Postgres/MySQL easily)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./neurodetect.db")

    # CORS origins (restrict in production)
    ALLOWED_ORIGINS: list = ["*"]

settings = Settings()

