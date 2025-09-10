# backend/app/ml_models/train_qda.py

"""
QDA Model Training Script
-------------------------
- Loads EEG dataset with features and labels
- Trains QuadraticDiscriminantAnalysis model
- Saves trained model to pickle file
- Handles proper data preparation and validation
"""

import os
import pickle
import logging
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import make_classification

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def generate_dummy_eeg_dataset(n_samples=1000):
    """
    Generate dummy EEG-like dataset for training.
    Replace this with your actual EEG data loading function.
    """
    logger.info(f"Generating dummy EEG dataset with {n_samples} samples...")
    
    # Generate synthetic classification data
    X_raw, y = make_classification(
        n_samples=n_samples,
        n_features=12,  # 5 band powers + 7 statistical features
        n_classes=2,
        n_redundant=2,
        n_informative=8,
        random_state=42,
        class_sep=0.8
    )
    
    # Convert to EEG-like feature format
    dataset = []
    for i in range(n_samples):
        # Split features into band powers and statistics
        features = {
            "band_powers": {
                "Delta_Waves": float(abs(X_raw[i][0])),
                "Theta_Waves": float(abs(X_raw[i][1])),
                "Alpha_Waves": float(abs(X_raw[i][2])),
                "Beta_Waves": float(abs(X_raw[i][3])),
                "Gamma_Waves": float(abs(X_raw[i][4]))
            },
            "statistics": {
                "mean_amplitude": float(X_raw[i][5]),
                "signal_variance": float(abs(X_raw[i][6])),
                "standard_deviation": float(abs(X_raw[i][7])),
                "kurtosis": float(X_raw[i][8]),
                "skewness": float(X_raw[i][9]),
                "peak_amplitude": float(abs(X_raw[i][10])),
                "rms_amplitude": float(abs(X_raw[i][11]))
            }
        }
        
        sample = {
            "features": features,
            "label": int(y[i])  # 0 = Normal, 1 = Seizure
        }
        dataset.append(sample)
    
    # Verify class distribution
    class_counts = {}
    for sample in dataset:
        label = sample["label"]
        class_counts[label] = class_counts.get(label, 0) + 1
    
    logger.info(f"Dataset class distribution: {class_counts}")
    return dataset

def prepare_data_for_qda(dataset):
    """
    Convert dataset into feature matrix (X) and labels (y) for QDA training.
    """
    logger.info("Preparing data for QDA training...")
    
    X = []
    y = []
    
    for sample in dataset:
        try:
            feats = []
            
            # Flatten band powers
            if "band_powers" in sample["features"]:
                for val in sample["features"]["band_powers"].values():
                    feats.append(float(val))
            
            # Flatten statistics
            if "statistics" in sample["features"]:
                for val in sample["features"]["statistics"].values():
                    feats.append(float(val))
            
            if len(feats) > 0:
                X.append(feats)
                y.append(int(sample["label"]))
                
        except Exception as e:
            logger.warning(f"Skipping invalid sample: {e}")
            continue
    
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"Prepared {len(X)} samples with {X.shape[1]} features")
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Label distribution: {np.bincount(y)}")
    
    return X, y

def train_and_save_qda():
    """
    Train QDA model and save to disk.
    """
    try:
        logger.info("🚀 Starting QDA model training...")
        
        # Load dataset (replace with your actual data loading)
        dataset = generate_dummy_eeg_dataset(n_samples=2000)
        
        # Prepare data
        X, y = prepare_data_for_qda(dataset)
        
        # Check if we have enough samples and classes
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            raise ValueError(f"Need at least 2 classes for training, got {len(unique_classes)}")
        
        if len(X) < 10:
            raise ValueError(f"Need at least 10 samples for training, got {len(X)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Train QDA model
        logger.info("Training QuadraticDiscriminantAnalysis model...")
        model = QuadraticDiscriminantAnalysis()
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        logger.info(f"✅ QDA training completed!")
        logger.info(f"📊 Test Accuracy: {accuracy * 100:.2f}%")
        logger.info(f"📊 Confusion Matrix:\n{cm}")
        
        # Print detailed classification report
        report = classification_report(y_test, y_pred, target_names=['Normal', 'Seizure'])
        logger.info(f"📊 Classification Report:\n{report}")
        
        # Save model to disk
        model_dir = "app/ml_models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "qda_model.pkl")
        
        logger.info(f"💾 Saving QDA model to {model_path}...")
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Verify save
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            logger.info(f"✅ QDA model saved successfully ({file_size} bytes)")
        else:
            raise Exception("Failed to save model file")
            
        # Test loading the saved model
        logger.info("🔍 Testing model loading...")
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        # Quick prediction test
        test_pred = loaded_model.predict(X_test[:5])
        logger.info(f"✅ Model loading test successful. Sample predictions: {test_pred}")
        
        return {
            "model_path": model_path,
            "accuracy": accuracy * 100,
            "confusion_matrix": cm.tolist(),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_count": X.shape[1]
        }
        
    except Exception as e:
        logger.error(f"❌ QDA training failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        results = train_and_save_qda()
        print("\n" + "="*50)
        print("🎉 QDA TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Model saved to: {results['model_path']}")
        print(f"Accuracy: {results['accuracy']:.2f}%")
        print(f"Training samples: {results['train_samples']}")
        print(f"Test samples: {results['test_samples']}")
        print(f"Feature count: {results['feature_count']}")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        exit(1)
