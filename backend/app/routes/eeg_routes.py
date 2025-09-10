# backend/app/routes/eeg_routes.py

import os
from pathlib import Path
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from app.services.utils import save_uploaded_file

router = APIRouter(prefix="/api", tags=["EEG Upload"])

@router.post("/upload")
async def upload_eeg(file: UploadFile = File(...)):
    try:
        # ✅ Use the centralized save function with await
        file_path = await save_uploaded_file(file, upload_dir="uploads")
        
        return JSONResponse(content={
            "message": "File uploaded successfully", 
            "filename": file.filename,
            "file_path": str(file_path)
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={"error": f"Upload failed: {str(e)}"}
        )
