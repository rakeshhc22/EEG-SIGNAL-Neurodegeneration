# backend/app/services/utils.py

"""
Enhanced Utility Functions for EEG Analysis System
-------------------------------------------------
- File upload handling with validation
- EEG file format support 
- Error handling and logging
"""

import os
import shutil
import uuid
from typing import Optional, Union
from fastapi import UploadFile, HTTPException
import logging
import asyncio

logger = logging.getLogger(__name__)

# Supported file extensions for EEG data
SUPPORTED_EXTENSIONS = {'.txt', '.edf', '.csv', '.dat', '.fif', '.set'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB

async def save_uploaded_file(upload_file: UploadFile, upload_dir: str = "uploads") -> str:
    """
    Save uploaded EEG file with validation and error handling.
    
    Args:
        upload_file: FastAPI UploadFile object
        upload_dir: Directory to save the file (default: "uploads")
        
    Returns:
        str: Path to saved file
        
    Raises:
        HTTPException: If file validation fails or save operation fails
    """
    try:
        # Validate file
        await _validate_upload_file(upload_file)
        
        # Create directory if it doesn't exist
        os.makedirs(upload_dir, exist_ok=True)
        
        # Generate unique filename while preserving original extension
        original_filename = upload_file.filename or "unknown_file"
        file_extension = os.path.splitext(original_filename)[1].lower()
        
        if not file_extension:
            file_extension = ".txt"  # Default extension for EEG files
            
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(upload_dir, unique_filename)
        
        # Save file content
        with open(file_path, "wb") as buffer:
            # Reset file pointer to beginning
            await upload_file.seek(0)
            content = await upload_file.read()
            buffer.write(content)
        
        # Verify file was saved correctly
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            raise HTTPException(status_code=500, detail="Failed to save file completely")
            
        logger.info(f"✅ File saved successfully: {file_path} ({os.path.getsize(file_path)} bytes)")
        return file_path
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"❌ Error saving uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

async def _validate_upload_file(upload_file: UploadFile) -> None:
    """Validate uploaded file before processing."""
    if not upload_file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file extension
    file_extension = os.path.splitext(upload_file.filename)[1].lower()
    if file_extension and file_extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file_extension}. Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    
    # Check file size (read a portion to estimate)
    try:
        await upload_file.seek(0)
        content_sample = await upload_file.read(1024)  # Read first 1KB
        await upload_file.seek(0)  # Reset for actual saving
        
        if len(content_sample) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
            
    except Exception as e:
        logger.error(f"File validation error: {e}")
        raise HTTPException(status_code=400, detail="Invalid file format")

def cleanup_file(file_path: str) -> bool:
    """
    Safely remove file from filesystem.
    
    Args:
        file_path: Path to file to remove
        
    Returns:
        bool: True if file was removed, False otherwise
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"🗑️ File removed: {file_path}")
            return True
        else:
            logger.warning(f"⚠️ File not found for cleanup: {file_path}")
            return False
    except Exception as e:
        logger.error(f"❌ Error removing file {file_path}: {e}")
        return False

def get_file_info(file_path: str) -> dict:
    """
    Get comprehensive file information.
    
    Args:
        file_path: Path to file
        
    Returns:
        dict: File information including size, type, and status
    """
    try:
        if not os.path.exists(file_path):
            return {"error": "File not found", "exists": False}
            
        stat_info = os.stat(file_path)
        file_extension = os.path.splitext(file_path)[1].lower()
        
        return {
            "path": file_path,
            "filename": os.path.basename(file_path),
            "size_bytes": stat_info.st_size,
            "size_mb": round(stat_info.st_size / (1024 * 1024), 2),
            "extension": file_extension,
            "supported": file_extension in SUPPORTED_EXTENSIONS,
            "exists": True,
            "created": stat_info.st_ctime,
            "modified": stat_info.st_mtime
        }
    except Exception as e:
        logger.error(f"Error getting file info for {file_path}: {e}")
        return {"error": f"Could not get file info: {str(e)}", "exists": False}

def create_upload_directory(directory: str = "uploads") -> str:
    """
    Create upload directory if it doesn't exist.
    
    Args:
        directory: Directory path to create
        
    Returns:
        str: Absolute path to created directory
    """
    try:
        abs_path = os.path.abspath(directory)
        os.makedirs(abs_path, exist_ok=True)
        logger.info(f"📁 Upload directory ready: {abs_path}")
        return abs_path
    except Exception as e:
        logger.error(f"Failed to create upload directory {directory}: {e}")
        raise Exception(f"Could not create upload directory: {str(e)}")

def validate_eeg_file_content(file_path: str) -> dict:
    """
    Validate EEG file content and extract basic information.
    
    Args:
        file_path: Path to EEG file
        
    Returns:
        dict: Validation results and file info
    """
    try:
        file_info = get_file_info(file_path)
        if "error" in file_info:
            return file_info
            
        validation_result = {
            "valid": False,
            "file_type": "unknown",
            "estimated_channels": 0,
            "estimated_samples": 0,
            "warnings": []
        }
        
        # Basic content validation based on file extension
        extension = file_info["extension"]
        
        if extension == ".txt":
            # Try to read as text and estimate structure
            try:
                with open(file_path, 'r') as f:
                    first_lines = [f.readline().strip() for _ in range(5)]
                    
                # Check if it looks like numeric data
                numeric_lines = 0
                max_columns = 0
                
                for line in first_lines:
                    if line:
                        try:
                            # Try to split and convert to float
                            values = [float(x) for x in line.split()]
                            numeric_lines += 1
                            max_columns = max(max_columns, len(values))
                        except ValueError:
                            continue
                            
                if numeric_lines >= 2 and max_columns > 0:
                    validation_result.update({
                        "valid": True,
                        "file_type": "text_numeric",
                        "estimated_channels": max_columns,
                        "estimated_samples": max(0, file_info["size_bytes"] // 50)  # Rough estimate
                    })
                else:
                    validation_result["warnings"].append("File does not appear to contain numeric EEG data")
                    
            except Exception as e:
                validation_result["warnings"].append(f"Could not read file content: {str(e)}")
                
        elif extension in [".edf", ".fif", ".set"]:
            validation_result.update({
                "valid": True,
                "file_type": f"binary_{extension[1:]}",
                "estimated_channels": "unknown",
                "estimated_samples": "unknown"
            })
            validation_result["warnings"].append("Binary EEG format - content validation requires specialized libraries")
            
        elif extension == ".csv":
            try:
                # Quick CSV validation
                with open(file_path, 'r') as f:
                    first_line = f.readline().strip()
                    if ',' in first_line:
                        columns = len(first_line.split(','))
                        validation_result.update({
                            "valid": True,
                            "file_type": "csv",
                            "estimated_channels": columns,
                            "estimated_samples": "unknown"
                        })
                    else:
                        validation_result["warnings"].append("CSV file does not contain comma-separated values")
            except Exception as e:
                validation_result["warnings"].append(f"Could not validate CSV: {str(e)}")
        
        return {
            **file_info,
            "validation": validation_result
        }
        
    except Exception as e:
        logger.error(f"EEG file validation failed for {file_path}: {e}")
        return {"error": f"Validation failed: {str(e)}", "valid": False}

# Initialize upload directory on import
try:
    create_upload_directory()
except Exception as e:
    logger.warning(f"Could not initialize default upload directory: {e}")
