# backend/app/services/training_service.py
"""
Training Service for EEG Models
-------------------------------
- Loads EEG dataset
- Extracts features
- Builds dataset
- Trains QDA & TabNet models
- Returns metrics
"""

from typing import Dict, List
from . import feature_extraction, utils, model_qda, model_tabnet


class TrainingService:
    def __init__(self):
        self.qda = model_qda.qda_model
        self.tabnet = model_tabnet.tabnet_model

    def prepare_dataset(self, eeg_files: List[Dict]) -> List[Dict]:
        """
        Extract features from EEG files and create labeled dataset.

        Args:
            eeg_files (list of dict): [{"file_path": str, "label": int}, ...]

        Returns:
            list: [{"features": {...}, "label": int}]
        """
        dataset = []
        for sample in eeg_files:
            file_path = sample["file_path"]
            label = sample["label"]

            features = feature_extraction.extract_features(file_path)
            dataset.append({"features": features, "label": label})

        return dataset

    def train_qda(self, dataset: List[Dict]) -> Dict:
        """
        Train QDA model and return metrics.
        """
        metrics = self.qda.train(dataset)
        return {"model": "QDA", **metrics}

    def train_tabnet(self, dataset: List[Dict]) -> Dict:
        """
        Train TabNet model and return metrics.
        """
        metrics = self.tabnet.train(dataset)
        return {"model": "TabNet", **metrics}

    def train_all(self, eeg_files: List[Dict]) -> Dict:
        """
        Train both QDA & TabNet models on dataset.

        Args:
            eeg_files (list): [{"file_path": str, "label": int}]

        Returns:
            dict: Training metrics for both models
        """
        dataset = self.prepare_dataset(eeg_files)

        results = {
            "QDA": self.train_qda(dataset),
            "TabNet": self.train_tabnet(dataset),
        }

        return results

    def predict(self, file_path: str) -> Dict:
        """
        Run prediction on a single EEG file using trained models.

        Args:
            file_path (str): Path to EEG file.

        Returns:
            dict: Predictions from both QDA and TabNet.
        """
        features = feature_extraction.extract_features(file_path)

        qda_pred = self.qda.predict(features)
        tabnet_pred = self.tabnet.predict(features)

        return {
            "QDA": qda_pred,
            "TabNet": tabnet_pred,
            "features_used": features,
        }


# ✅ Helper instance
training_service = TrainingService()
