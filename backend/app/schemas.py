# backend/app/schemas.py

from pydantic import BaseModel
from typing import List, Dict, Optional

class BandPowers(BaseModel):
    Delta_Waves: float
    Theta_Waves: float
    Alpha_Waves: float
    Beta_Waves: float
    Gamma_Waves: float

class Statistics(BaseModel):
    mean_amplitude: float
    signal_variance: float
    standard_deviation: float
    kurtosis: float
    skewness: float
    peak_amplitude: float
    rms_amplitude: float

class Temporal(BaseModel):
    zero_crossings: int
    signal_energy: float
    peak_frequency: float
    dominant_frequency: float

class FileInfo(BaseModel):
    filename: str
    file_size: int
    file_extension: str

class Features(BaseModel):
    band_powers: BandPowers
    statistics: Statistics
    temporal: Temporal
    file_info: FileInfo
    extraction_status: str

class ModelResult(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: List[float]
    feature_count: Optional[int] = None

class AnalysisResults(BaseModel):
    QDA: ModelResult
    TabNet: ModelResult
    features_used: Features

class AnalysisResponse(BaseModel):
    message: str
    file: str
    results: AnalysisResults
