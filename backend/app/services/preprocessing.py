# backend/app/services/preprocessing.py
"""
This module handles EEG preprocessing:
- Filtering
- Normalization
- Feature extraction (dummy for now)
"""

from typing import Dict
import os

def extract_features(file_path: str) -> Dict:
    """
    Dummy EEG feature extraction function.
    Later, replace with real EEG libraries (e.g., mne, numpy).

    Args:
        file_path (str): Path to uploaded EEG file.

    Returns:
        dict: Extracted features (dummy values for now).
    """

    if not os.path.exists(file_path):
        return {"error": "File not found"}

    # Dummy features (later compute from EEG)
    features = {
        "mean_amplitude": 0.23,
        "signal_variance": 0.12,
        "peak_frequency": 8.5,  # Hz
    }

    return features
