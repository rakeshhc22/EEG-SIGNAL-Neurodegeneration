# backend/app/routes/analysis_routes.py

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
import os
import traceback
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger("analysis_routes")
logging.basicConfig(level=logging.INFO)

from app.services.training_service import training_service
from app.services.utils import save_uploaded_file
from app.schemas import AnalysisResponse, PredictionResult

router = APIRouter(prefix="/api", tags=["Analysis"])


@router.post("/analysis")
async def analyze_signal(
    file: UploadFile = File(...),
    patient_id: Optional[int] = Form(None),
    patient_name: Optional[str] = Form(None),
    patient_age: Optional[str] = Form(None)
):
    """
    Main EEG analysis endpoint.
    
    Returns:
        {
            "message": "...",
            "file": "filename.csv",
            "results": {
                "QDA": {...},
                "TabNet": {...},
                "ensemble": {...}  # ALWAYS included
            }
        }
    """
    try:
        # Save uploaded file
        file_path = await save_uploaded_file(file, upload_dir="uploads")
        logger.info(f"📁 File saved: {os.path.basename(file_path)}")
        
        # Run ML predictions
        raw_results = training_service.predict(file_path)
        
        # ✅ FIXED: Include ensemble results
        response_data = {
            "message": "EEG analysis completed successfully",
            "file": os.path.basename(file_path),
            "timestamp": datetime.now().isoformat(),
            "results": {
                "QDA": {
                    "predicted_class": raw_results["QDA"]["predicted_class"],
                    "confidence": raw_results["QDA"]["confidence"],
                    "probabilities": raw_results["QDA"].get("probabilities", []),
                    "status": raw_results["QDA"].get("status", "unknown")
                },
                "TabNet": {
                    "predicted_class": raw_results["TabNet"]["predicted_class"],
                    "confidence": raw_results["TabNet"]["confidence"],
                    "probabilities": raw_results["TabNet"].get("probabilities", []),
                    "status": raw_results["TabNet"].get("status", "unknown")
                },
                "ensemble": {
                    "predicted_class": raw_results["ensemble"]["predicted_class"],
                    "confidence": raw_results["ensemble"]["confidence"],
                    "method": raw_results["ensemble"]["method"]
                }
            }
        }
        
        # ✅ CRITICAL: Log the confidence values
        logger.info(f"📊 Response Confidence Values:")
        logger.info(f"   QDA: {response_data['results']['QDA']['confidence']}")
        logger.info(f"   TabNet: {response_data['results']['TabNet']['confidence']}")
        logger.info(f"   Ensemble: {response_data['results']['ensemble']['confidence']}")
        
        return JSONResponse(content=response_data, status_code=200)
        
    except Exception as e:
        logger.error(f"❌ Analysis endpoint error: {str(e)}")
        traceback.print_exc()
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "type": type(e).__name__,
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "NeuroDetect API",
        "models": {
            "qda": training_service.qda is not None,
            "tabnet": training_service.tabnet is not None
        },
        "features": 62,
        "timestamp": datetime.now().isoformat()
    }
