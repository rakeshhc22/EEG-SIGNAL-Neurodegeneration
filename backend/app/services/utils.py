# backend/app/services/utils.py

"""
Utility functions for EEG project
---------------------------------
- File handling
- Dataset preparation
- Metrics formatting
"""

import os
import uuid
from typing import Dict, List
import numpy as np
from fastapi import UploadFile

async def save_uploaded_file(file: UploadFile, upload_dir: str = "uploads") -> str:
    """
    Save uploaded EEG file to disk.
    Args:
        file (UploadFile): File from FastAPI request.
        upload_dir (str): Directory to save file.
    Returns:
        str: Path to saved file.
    """
    # Ensure upload directory exists
    os.makedirs(upload_dir, exist_ok=True)
    
    # Generate unique filename with original extension
    file_ext = os.path.splitext(file.filename)[-1] if file.filename else ".tmp"
    filename = f"{uuid.uuid4().hex}{file_ext}"
    file_path = os.path.join(upload_dir, filename)
    
    try:
        # Read file content asynchronously
        content = await file.read()
        
        # Write file to disk
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        # Reset file position for potential reuse
        await file.seek(0)
        
        # Return absolute path for consistency
        return os.path.abspath(file_path)
        
    except Exception as e:
        # Clean up partial file if error occurs
        if os.path.exists(file_path):
            os.remove(file_path)
        raise Exception(f"Failed to save file: {str(e)}")

def flatten_features(features: Dict) -> np.ndarray:
    """
    Convert nested feature dict (band_powers + statistics) into flat NumPy array.
    Args:
        features (dict): EEG features.
    Returns:
        np.ndarray: 1D feature vector.
    """
    feats = []
    
    # Handle band powers
    if "band_powers" in features and isinstance(features["band_powers"], dict):
        for key, value in features["band_powers"].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                feats.append(float(value))
            else:
                feats.append(0.0)  # Default for invalid values
    
    # Handle statistics
    if "statistics" in features and isinstance(features["statistics"], dict):
        for key, value in features["statistics"].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                feats.append(float(value))
            else:
                feats.append(0.0)  # Default for invalid values
    
    return np.array(feats)

def build_dataset(samples: List[Dict]) -> tuple:
    """
    Build feature matrix (X) and labels (y) from dataset.
    Args:
        samples (list): Each sample = {"features": {...}, "label": 0/1}
    Returns:
        tuple: (X, y) NumPy arrays for training models
    """
    X = []
    y = []
    
    for sample in samples:
        if "features" in sample and "label" in sample:
            try:
                feature_vector = flatten_features(sample["features"])
                if len(feature_vector) > 0:
                    X.append(feature_vector)
                    y.append(int(sample["label"]))
            except Exception as e:
                print(f"Warning: Skipping invalid sample: {e}")
                continue
    
    return np.array(X), np.array(y)

def format_metrics(y_true, y_pred) -> Dict:
    """
    Generate accuracy and confusion matrix from predictions.
    Args:
        y_true (list/array): True labels
        y_pred (list/array): Predicted labels
    Returns:
        dict: Metrics
    """
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    
    try:
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred).tolist()
        
        return {
            "accuracy": round(float(acc) * 100, 2),
            "confusion_matrix": cm,
            "samples": len(y_true),
            "classification_report": classification_report(y_true, y_pred, output_dict=True)
        }
    except Exception as e:
        return {
            "accuracy": 0.0,
            "confusion_matrix": [],
            "samples": 0,
            "error": str(e)
        }

def validate_file_path(file_path: str) -> bool:
    """
    Validate if file path exists and is accessible.
    Args:
        file_path (str): Path to file
    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(file_path, str):
        return False
    
    return os.path.exists(file_path) and os.path.isfile(file_path)

def get_file_info(file_path: str) -> Dict:
    """
    Get information about uploaded file.
    Args:
        file_path (str): Path to file
    Returns:
        dict: File information
    """
    if not validate_file_path(file_path):
        return {"error": "Invalid file path"}
    
    try:
        stat = os.stat(file_path)
        return {
            "filename": os.path.basename(file_path),
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "extension": os.path.splitext(file_path)[1],
            "absolute_path": os.path.abspath(file_path)
        }
    except Exception as e:
        return {"error": f"Failed to get file info: {str(e)}"}
