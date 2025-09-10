# backend/app/services/model_qda.py

"""
QDA Model for EEG Classification
--------------------------------
- Uses scikit-learn's QuadraticDiscriminantAnalysis
- Trains on EEG features (e.g., band powers, statistics)
- Provides prediction (Seizure vs Normal) with confidence
- Loads trained model from disk automatically
"""

import numpy as np
import pickle
import os
import logging
from typing import Dict, List, Tuple
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QDAModel:
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.model_path = "app/ml_models/qda_model.pkl"
        
        # Auto-load model on initialization
        self.load_model()

    def load_model(self):
        """
        Load trained QDA model from disk if it exists.
        """
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.is_trained = True
                logger.info(f"✅ QDA model loaded successfully from {self.model_path}")
                print(f"✅ QDA model loaded successfully from {self.model_path}")
                return True
            except Exception as e:
                logger.error(f"❌ Failed to load QDA model: {e}")
                print(f"❌ Failed to load QDA model: {e}")
                self.model = QuadraticDiscriminantAnalysis()
                self.is_trained = False
                return False
        else:
            logger.warning(f"⚠️ QDA model file not found at {self.model_path}")
            print(f"⚠️ QDA model file not found at {self.model_path}")
            self.model = QuadraticDiscriminantAnalysis()
            self.is_trained = False
            return False

    def save_model(self):
        """
        Save trained QDA model to disk.
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            logger.info(f"✅ QDA model saved to {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save QDA model: {e}")
            return False

    def prepare_data(self, dataset: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert dataset into feature matrix (X) and labels (y).
        Args:
            dataset (list of dict): Each dict should have 'features' and 'label'.
        Returns:
            (X, y) -> NumPy arrays for training/testing
        """
        X = []
        y = []
        
        for sample in dataset:
            try:
                feats = []
                
                # Flatten band powers
                if "band_powers" in sample["features"] and isinstance(sample["features"]["band_powers"], dict):
                    for val in sample["features"]["band_powers"].values():
                        if isinstance(val, (int, float)) and not isinstance(val, bool):
                            feats.append(float(val))
                        else:
                            feats.append(0.0)
                
                # Flatten statistics
                if "statistics" in sample["features"] and isinstance(sample["features"]["statistics"], dict):
                    for val in sample["features"]["statistics"].values():
                        if isinstance(val, (int, float)) and not isinstance(val, bool):
                            feats.append(float(val))
                        else:
                            feats.append(0.0)
                
                if len(feats) > 0:
                    X.append(feats)
                    y.append(int(sample["label"]))  # 0 = Normal, 1 = Seizure
                    
            except Exception as e:
                logger.warning(f"Skipping invalid sample: {e}")
                continue
        
        return np.array(X), np.array(y)

    def train(self, dataset: List[Dict]) -> Dict:
        """
        Train the QDA model on given dataset.
        Args:
            dataset (list): List of samples with 'features' and 'label'.
        Returns:
            dict: Training performance metrics.
        """
        try:
            X, y = self.prepare_data(dataset)
            
            if len(X) == 0:
                raise ValueError("No valid samples in dataset")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Initialize fresh model for training
            self.model = QuadraticDiscriminantAnalysis()
            self.model.fit(X_train, y_train)
            self.is_trained = True
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred).tolist()
            
            # Save model automatically after training
            self.save_model()
            
            results = {
                "accuracy": round(float(acc) * 100, 2),
                "confusion_matrix": cm,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "total_features": X.shape[1] if len(X) > 0 else 0
            }
            
            logger.info(f"QDA training completed: {results['accuracy']}% accuracy")
            return results
            
        except Exception as e:
            logger.error(f"QDA training failed: {e}")
            self.is_trained = False
            return {
                "error": str(e),
                "accuracy": 0.0,
                "confusion_matrix": [],
                "train_samples": 0,
                "test_samples": 0
            }

    def predict(self, features: Dict) -> Dict:
        """
        Predict seizure probability for given features.
        Args:
            features (dict): Extracted EEG features.
        Returns:
            dict: Predicted class and confidence.
        """
        # Check if model is trained and loaded
        if not self.is_trained or self.model is None:
            error_msg = "QDA model is not trained yet. Please train the model first or check if model file exists."
            logger.error(error_msg)
            return {
                "error": error_msg,
                "predicted_class": "Unknown",
                "confidence": 0.0,
                "probabilities": [0.5, 0.5]
            }
        
        try:
            feats = []
            
            # Flatten band powers
            if "band_powers" in features and isinstance(features["band_powers"], dict):
                for val in features["band_powers"].values():
                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                        feats.append(float(val))
                    else:
                        feats.append(0.0)
            
            # Flatten statistics
            if "statistics" in features and isinstance(features["statistics"], dict):
                for val in features["statistics"].values():
                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                        feats.append(float(val))
                    else:
                        feats.append(0.0)
            
            if len(feats) == 0:
                raise ValueError("No valid features provided")
            
            X = np.array([feats])
            proba = self.model.predict_proba(X)[0]
            predicted = int(np.argmax(proba))
            
            result = {
                "predicted_class": "Seizure Detected" if predicted == 1 else "Normal",
                "confidence": round(float(np.max(proba) * 100), 2),
                "probabilities": [round(float(p), 4) for p in proba],
                "feature_count": len(feats)
            }
            
            logger.info(f"QDA prediction: {result['predicted_class']} ({result['confidence']}%)")
            return result
            
        except Exception as e:
            error_msg = f"QDA prediction failed: {str(e)}"
            logger.error(error_msg)
            return {
                "error": error_msg,
                "predicted_class": "Error",
                "confidence": 0.0,
                "probabilities": [0.5, 0.5]
            }

    def get_model_info(self) -> Dict:
        """
        Get information about the current model state.
        """
        return {
            "is_trained": self.is_trained,
            "model_path": self.model_path,
            "model_exists": os.path.exists(self.model_path),
            "model_type": "QuadraticDiscriminantAnalysis",
            "model_loaded": self.model is not None
        }

# ✅ Create the global instance
qda_model = QDAModel()
