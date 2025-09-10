# backend/app/routes/analysis_routes.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import traceback
from datetime import datetime
from app.services.training_service import training_service
from app.services.utils import save_uploaded_file
from app.schemas import AnalysisResponse

router = APIRouter(prefix="/api", tags=["Analysis"])

@router.post("/analysis", response_model=AnalysisResponse)
async def analyze_signal(file: UploadFile = File(...), patient_id: int = 1):
    try:
        # Save uploaded file
        file_path = await save_uploaded_file(file, upload_dir="uploads")
        
        # Run ML predictions
        results = training_service.predict(file_path)
        
        return AnalysisResponse(
            message="EEG analysis completed",
            file=os.path.basename(file_path),
            results=results
        )
        
    except Exception as e:
        # Add detailed error logging
        print(f"❌ Error in analysis endpoint:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        print(f"   Full traceback:")
        traceback.print_exc()
        
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
