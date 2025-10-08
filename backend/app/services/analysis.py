# backend/app/routes/analysis.py

"""
EEG Analysis API Route - Integrated with QDA and TabNet models
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict
import os
import shutil
from ..services.training_service import training_service

router = APIRouter()

@router.post("/api/analysis")
async def analyze_eeg(file: UploadFile = File(...)) -> Dict:
    """
    Analyze uploaded EEG CSV file using QDA and TabNet models
    
    Args:
        file: Uploaded CSV file with 179 features
        
    Returns:
        dict: Analysis results from both models
    """
    try:
        # Save uploaded file temporarily
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Run prediction using training_service
        results = training_service.predict(file_path)
        
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Format response for frontend
        response = {
            "success": True,
            "models": {
                "QDA": {
                    "predicted_class": results["QDA"].get("predicted_class", "Error"),
                    "confidence": results["QDA"].get("confidence", 0),
                    "accuracy": results["QDA"].get("accuracy", 0),
                    "probabilities": results["QDA"].get("probability_details", {}),
                    "method": results["QDA"].get("method", "QDA")
                },
                "TabNet": {
                    "predicted_class": results["TabNet"].get("predicted_class", "Error"),
                    "confidence": results["TabNet"].get("confidence", 0),
                    "accuracy": results["TabNet"].get("accuracy", 0),
                    "probabilities": results["TabNet"].get("probability_details", {}),
                    "method": results["TabNet"].get("method", "TabNet")
                }
            },
            "ensemble": {
                "predicted_class": results["ensemble"].get("predicted_class", "Unknown"),
                "confidence": results["ensemble"].get("confidence", 0),
                "method": results["ensemble"].get("method", "Ensemble")
            }
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
