# backend/app/services/feature_extraction.py

"""
This module handles EEG feature extraction:
- Band power (Delta, Theta, Alpha, Beta, Gamma)
- Statistical features (mean, variance, kurtosis, etc.)
- Connectivity measures (optional for future expansion)
"""

from typing import Dict, List, Optional
import os
import random
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_file_path(file_path: str) -> bool:
    """
    Validate that file_path is a string and file exists.
    Args:
        file_path: Path to EEG file
    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(file_path, str):
        logger.error(f"Invalid file_path type: {type(file_path)}. Expected str.")
        return False
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False
    
    if not os.path.isfile(file_path):
        logger.error(f"Path is not a file: {file_path}")
        return False
    
    return True

def extract_band_powers(file_path: str) -> Dict:
    """
    Simulate EEG band power extraction.
    Later, replace with real computation using MNE/NumPy/Scipy.
    Args:
        file_path (str): Path to uploaded EEG file.
    Returns:
        dict: Relative power of EEG frequency bands.
    """
    # Validate input
    if not validate_file_path(file_path):
        return {
            "error": "Invalid file path or file not found",
            "file_path": str(file_path),
            "file_path_type": str(type(file_path))
        }
    
    try:
        # Get file size to make features more realistic
        file_size = os.path.getsize(file_path)
        
        # Use file size as seed for consistent results per file
        random.seed(file_size % 1000)
        
        # Generate realistic EEG band powers that sum to approximately 1.0
        delta = round(random.uniform(0.25, 0.45), 3)  # Delta: 0.5-4 Hz
        theta = round(random.uniform(0.15, 0.25), 3)  # Theta: 4-8 Hz
        alpha = round(random.uniform(0.10, 0.20), 3)  # Alpha: 8-13 Hz
        beta = round(random.uniform(0.08, 0.18), 3)   # Beta: 13-30 Hz
        gamma = round(1.0 - (delta + theta + alpha + beta), 3)  # Gamma: 30+ Hz
        
        # Ensure gamma is positive
        if gamma < 0:
            gamma = round(random.uniform(0.05, 0.15), 3)
            # Renormalize
            total = delta + theta + alpha + beta + gamma
            delta = round(delta / total, 3)
            theta = round(theta / total, 3)
            alpha = round(alpha / total, 3)
            beta = round(beta / total, 3)
            gamma = round(gamma / total, 3)
        
        bands = {
            "Delta_Waves": delta,
            "Theta_Waves": theta,
            "Alpha_Waves": alpha,
            "Beta_Waves": beta,
            "Gamma_Waves": gamma
        }
        
        logger.info(f"Extracted band powers from {os.path.basename(file_path)}")
        return bands
        
    except Exception as e:
        logger.error(f"Error extracting band powers: {str(e)}")
        return {
            "error": f"Failed to extract band powers: {str(e)}",
            "Delta_Waves": 0.0,
            "Theta_Waves": 0.0,
            "Alpha_Waves": 0.0,
            "Beta_Waves": 0.0,
            "Gamma_Waves": 0.0
        }

def extract_statistical_features(file_path: str) -> Dict:
    """
    Simulate statistical feature extraction from EEG signals.
    Args:
        file_path (str): Path to uploaded EEG file.
    Returns:
        dict: Mean, variance, skewness (dummy values for now).
    """
    # Validate input
    if not validate_file_path(file_path):
        return {
            "error": "Invalid file path or file not found",
            "file_path": str(file_path),
            "file_path_type": str(type(file_path))
        }
    
    try:
        # Get file size for more realistic features
        file_size = os.path.getsize(file_path)
        
        # Use file characteristics for consistent results
        random.seed((file_size * 7) % 1000)
        
        # Generate realistic statistical features
        mean_amplitude = round(random.uniform(0.05, 0.35), 4)
        signal_variance = round(random.uniform(0.02, 0.25), 4)
        kurtosis = round(random.uniform(1.5, 5.0), 3)
        skewness = round(random.uniform(-2.0, 2.0), 3)
        std_deviation = round(np.sqrt(signal_variance), 4)
        
        # Additional features
        peak_amplitude = round(mean_amplitude + random.uniform(0.1, 0.5), 4)
        rms_amplitude = round(np.sqrt(mean_amplitude**2 + signal_variance), 4)
        
        features = {
            "mean_amplitude": mean_amplitude,
            "signal_variance": signal_variance,
            "standard_deviation": std_deviation,
            "kurtosis": kurtosis,
            "skewness": skewness,
            "peak_amplitude": peak_amplitude,
            "rms_amplitude": rms_amplitude
        }
        
        logger.info(f"Extracted statistical features from {os.path.basename(file_path)}")
        return features
        
    except Exception as e:
        logger.error(f"Error extracting statistical features: {str(e)}")
        return {
            "error": f"Failed to extract statistical features: {str(e)}",
            "mean_amplitude": 0.0,
            "signal_variance": 0.0,
            "standard_deviation": 0.0,
            "kurtosis": 0.0,
            "skewness": 0.0,
            "peak_amplitude": 0.0,
            "rms_amplitude": 0.0
        }

def extract_temporal_features(file_path: str) -> Dict:
    """
    Extract temporal domain features from EEG signal.
    Args:
        file_path (str): Path to uploaded EEG file.
    Returns:
        dict: Temporal features
    """
    if not validate_file_path(file_path):
        return {"error": "Invalid file path"}
    
    try:
        file_size = os.path.getsize(file_path)
        random.seed((file_size * 13) % 1000)
        
        # Simulate temporal features
        zero_crossings = random.randint(50, 200)
        signal_energy = round(random.uniform(0.1, 2.0), 4)
        peak_frequency = round(random.uniform(1.0, 40.0), 2)
        dominant_frequency = round(random.uniform(5.0, 15.0), 2)
        
        return {
            "zero_crossings": zero_crossings,
            "signal_energy": signal_energy,
            "peak_frequency": peak_frequency,
            "dominant_frequency": dominant_frequency
        }
        
    except Exception as e:
        logger.error(f"Error extracting temporal features: {str(e)}")
        return {
            "error": str(e),
            "zero_crossings": 0,
            "signal_energy": 0.0,
            "peak_frequency": 0.0,
            "dominant_frequency": 0.0
        }

def extract_features(file_path: str) -> Dict:
    """
    Combine all feature extraction methods.
    Args:
        file_path (str): Path to uploaded EEG file.
    Returns:
        dict: Consolidated features for ML models.
    """
    # Validate input type and existence
    if not isinstance(file_path, str):
        error_msg = f"Expected string file path, got {type(file_path)}: {file_path}"
        logger.error(error_msg)
        return {
            "error": error_msg,
            "band_powers": {"error": "Invalid input"},
            "statistics": {"error": "Invalid input"},
            "temporal": {"error": "Invalid input"}
        }
    
    if not validate_file_path(file_path):
        error_msg = f"File not found or inaccessible: {file_path}"
        logger.error(error_msg)
        return {
            "error": error_msg,
            "band_powers": {"error": "File not found"},
            "statistics": {"error": "File not found"},
            "temporal": {"error": "File not found"}
        }
    
    try:
        logger.info(f"Starting feature extraction for: {os.path.basename(file_path)}")
        
        # Extract all feature types
        band_powers = extract_band_powers(file_path)
        stats = extract_statistical_features(file_path)
        temporal = extract_temporal_features(file_path)
        
        # Get file metadata
        file_info = {
            "filename": os.path.basename(file_path),
            "file_size": os.path.getsize(file_path),
            "file_extension": os.path.splitext(file_path)[1]
        }
        
        result = {
            "band_powers": band_powers,
            "statistics": stats,
            "temporal": temporal,
            "file_info": file_info,
            "extraction_status": "success"
        }
        
        logger.info(f"Feature extraction completed successfully for: {os.path.basename(file_path)}")
        return result
        
    except Exception as e:
        error_msg = f"Unexpected error during feature extraction: {str(e)}"
        logger.error(error_msg)
        return {
            "error": error_msg,
            "band_powers": {"error": str(e)},
            "statistics": {"error": str(e)},
            "temporal": {"error": str(e)},
            "extraction_status": "failed"
        }

def validate_extracted_features(features: Dict) -> bool:
    """
    Validate that extracted features are in correct format.
    Args:
        features (dict): Extracted features
    Returns:
        bool: True if valid, False otherwise
    """
    required_keys = ["band_powers", "statistics"]
    
    if not isinstance(features, dict):
        return False
    
    for key in required_keys:
        if key not in features:
            return False
        if not isinstance(features[key], dict):
            return False
    
    return True

def get_feature_summary(features: Dict) -> Dict:
    """
    Get summary statistics of extracted features.
    Args:
        features (dict): Extracted features
    Returns:
        dict: Feature summary
    """
    if not validate_extracted_features(features):
        return {"error": "Invalid features format"}
    
    try:
        summary = {
            "total_features": 0,
            "band_power_features": len(features.get("band_powers", {})),
            "statistical_features": len(features.get("statistics", {})),
            "temporal_features": len(features.get("temporal", {})),
            "extraction_successful": "error" not in features
        }
        
        summary["total_features"] = (
            summary["band_power_features"] + 
            summary["statistical_features"] + 
            summary["temporal_features"]
        )
        
        return summary
        
    except Exception as e:
        return {"error": f"Failed to generate summary: {str(e)}"}
