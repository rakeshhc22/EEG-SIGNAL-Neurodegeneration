# backend/app/schemas.py

from pydantic import BaseModel
from typing import Dict, Any, Optional

class AnalysisResponse(BaseModel):
    message: str
    file: str
    results: Dict[str, Any]

class PredictionResult(BaseModel):
    predicted_class: str
    confidence: float
    accuracy: Optional[float] = None
    probabilities: Optional[list] = None
