"""
TabNet Model Training Script for 3-Class EEG Dataset
Optimized for 178-feature format with Normal, Seizure, and Neurodegeneration classes
"""

import numpy as np
import pandas as pd
import pickle
import os
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """Data loader for EEG CSV format"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def load_eeg_data(self, filepath):
        """Load 3-class EEG dataset directly"""
        print(f"Loading EEG dataset from {filepath}...")
        df = pd.read_csv(filepath)
        
        print(f"Raw dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()[:5]}... (showing first 5)")
        
        # Check if there's an unnamed index column
        if 'Unnamed: 0' in df.columns:
            print("Removing 'Unnamed: 0' index column...")
            df = df.drop(columns=['Unnamed: 0'])
        
        # Separate features and labels
        # Last column should be 'y' (label)
        if 'y' in df.columns:
            X = df.drop(columns=['y']).values
            y = df['y'].values
            feature_names = df.drop(columns=['y']).columns.tolist()
        else:
            # Assume last column is label
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            feature_names = df.columns[:-1].tolist()
        
        # Convert to numeric, forcing any non-numeric to NaN
        print("\nConverting data to numeric format...")
        X = pd.DataFrame(X).apply(pd.to_numeric, errors='coerce').values
        y = pd.Series(y).apply(pd.to_numeric, errors='coerce').values
        
        # Handle any NaN values created during conversion
        if np.any(np.isnan(X)):
            print(f"Warning: Found {np.sum(np.isnan(X))} NaN values after conversion, filling with 0")
            X = np.nan_to_num(X, nan=0.0)
        
        if np.any(np.isnan(y)):
            print(f"Warning: Found {np.sum(np.isnan(y))} NaN labels, this is a data quality issue!")
            y = np.nan_to_num(y, nan=0)
        
        print(f"\nFinal dataset shape: {X.shape}")
        print(f"Number of features: {X.shape[1]}")
        print(f"Classes: {np.unique(y)}")
        print(f"\nClass distribution:")
        for class_id, count in pd.Series(y).value_counts().sort_index().items():
            class_name = {0: 'Normal', 1: 'Seizure', 2: 'Neurodegeneration'}.get(int(class_id), 'Unknown')
            print(f"  Class {int(class_id)} ({class_name}): {count} samples")
        
        return X, y.astype(int), feature_names

class TabNetTrainer:
    """TabNet trainer for 3-class EEG classification"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.class_mapping = {
            0: 'Normal',
            1: 'Seizure',
            2: 'Neurodegeneration'
        }
    
    def prepare_data(self, X, y):
        """Prepare data with proper scaling and encoding"""
        print("\nPreparing data for training...")
        
        # Ensure X is float array
        X = X.astype(np.float32)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Final check for invalid values
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print("Warning: Found NaN or Inf values in final check, cleaning...")
            X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        
        print(f"Prepared dataset shape: {X.shape}")
        print(f"Label encoding: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        return X.astype(np.float32), y_encoded
    
    def train(self, X, y):
        """Train TabNet model with optimal hyperparameters"""
        print("\n" + "="*60)
        print("Training TabNet Model for 3-Class EEG Classification")
        print("="*60)
        
        # Split data (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTrain set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Scale features
        print("\nScaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply SMOTE for class balancing
        print("\nChecking class balance...")
        train_class_counts = pd.Series(y_train).value_counts()
        print(f"Training set class distribution:\n{train_class_counts}")
        
        # Only apply SMOTE if classes are imbalanced
        if train_class_counts.min() < train_class_counts.max() * 0.8:
            print("\nApplying SMOTE for class balance...")
            try:
                smote = SMOTE(random_state=42, k_neighbors=3)
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
                print(f"After SMOTE: {X_train_balanced.shape}")
                print(f"Balanced class distribution:\n{pd.Series(y_train_balanced).value_counts()}")
            except Exception as e:
                print(f"SMOTE not applied: {e}")
                X_train_balanced, y_train_balanced = X_train_scaled, y_train
        else:
            print("Classes are already balanced, skipping SMOTE")
            X_train_balanced, y_train_balanced = X_train_scaled, y_train
        
        # Ensure data types are correct for TabNet
        X_train_balanced = X_train_balanced.astype(np.float32)
        X_test_scaled = X_test_scaled.astype(np.float32)
        
        # Create validation set
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train_balanced, y_train_balanced, test_size=0.15,
            random_state=42, stratify=y_train_balanced
        )
        
        print(f"\nFinal train: {X_train_final.shape}")
        print(f"Validation: {X_val.shape}")
        print(f"Test: {X_test_scaled.shape}")
        
        # Configure TabNet model - optimized for 179 features
        print("\n" + "-"*60)
        print("Configuring TabNet model...")
        self.model = TabNetClassifier(
            n_d=64,                    # Decision prediction layer width
            n_a=64,                    # Attention embedding width
            n_steps=5,                 # Number of sequential attention steps
            gamma=1.3,                 # Coefficient for feature reusage
            lambda_sparse=1e-3,        # Sparsity regularization
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=0.02),
            mask_type='entmax',        # Attention mechanism type
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params=dict(step_size=20, gamma=0.9),
            verbose=1,
            device_name='auto',
            seed=42
        )
        
        print("TabNet Configuration:")
        print(f"  - Decision layer width: 64")
        print(f"  - Attention layer width: 64")
        print(f"  - Sequential steps: 5")
        print(f"  - Learning rate: 0.02")
        print(f"  - Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        
        # Train model
        print("\n" + "-"*60)
        print("Training TabNet...")
        print("-"*60)
        
        try:
            self.model.fit(
                X_train_final, y_train_final,
                eval_set=[(X_val, y_val)],
                eval_name=['validation'],
                eval_metric=['accuracy'],
                max_epochs=100,
                patience=20,
                batch_size=256,
                virtual_batch_size=128,
                num_workers=0,
                drop_last=False
            )
        except Exception as e:
            print(f"Training encountered an issue: {e}")
            print("Continuing with model evaluation...")
        
        # Evaluate on training data
        train_pred = self.model.predict(X_train_balanced)
        train_acc = accuracy_score(y_train_balanced, train_pred)
        
        # Evaluate on test data
        test_pred = self.model.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, test_pred)
        
        print("\n" + "-"*60)
        print(f"Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        # Detailed classification report
        print("\n" + "="*60)
        print("Classification Report (Test Set)")
        print("="*60)
        target_names = [self.class_mapping[i] for i in sorted(self.class_mapping.keys())]
        print(classification_report(y_test, test_pred, target_names=target_names, digits=4))
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, test_pred)
        print(cm)
        print("\nConfusion Matrix Format:")
        print("           Predicted")
        print("           Normal  Seizure  Neurodegeneration")
        print(f"Normal       {cm[0,0]:>4}    {cm[0,1]:>4}       {cm[0,2]:>4}")
        print(f"Seizure      {cm[1,0]:>4}    {cm[1,1]:>4}       {cm[1,2]:>4}")
        print(f"Neurodegenr. {cm[2,0]:>4}    {cm[2,1]:>4}       {cm[2,2]:>4}")
        
        # Feature importance
        print("\n" + "-"*60)
        print("Top 15 Most Important Features:")
        print("-"*60)
        feature_importances = self.model.feature_importances_
        top_indices = np.argsort(feature_importances)[-15:][::-1]
        for rank, idx in enumerate(top_indices, 1):
            feature_name = self.feature_names[idx] if self.feature_names else f"Feature {idx}"
            print(f"  {rank:>2}. {feature_name}: {feature_importances[idx]:.6f}")
        
        return self.model, test_acc
    
    def save_model(self, filepath):
        """Save trained TabNet model"""
        if self.model is None:
            print("Error: No model to save")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Get actual feature count
        n_features = len(self.feature_names) if self.feature_names else 178
        
        # Save TabNet model (keep existing naming convention)
        tabnet_path = filepath.replace('.pkl', '_tabnet')
        self.model.save_model(tabnet_path)
        
        # Save additional components
        model_components = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'class_mapping': self.class_mapping,
            'n_features': n_features,
            'feature_importances': self.model.feature_importances_,
            'model_type': 'TabNet',
            'classes': list(self.class_mapping.values())
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_components, f)
        
        print(f"\n✅ TabNet model saved successfully!")
        print(f"   Model weights: {tabnet_path}")
        print(f"   Model components: {filepath}")
        print(f"   Model type: TabNet")
        print(f"   Features: {n_features}")
        print(f"   Classes: {list(self.class_mapping.values())}")

def main():
    """Main training function"""
    print("\n" + "="*60)
    print("🚀 TabNet Model Training for 3-Class EEG Classification")
    print("   Normal | Seizure | Neurodegeneration")
    print("="*60)
    
    # Initialize components
    loader = DataLoader()
    trainer = TabNetTrainer()
    
    # File path - matches your folder structure
    data_file = "Data/Normal/3class_eeg_balanced_178features.csv"
    
    try:
        # Check if file exists
        if not os.path.exists(data_file):
            raise FileNotFoundError(
                f"Dataset not found at: {data_file}\n"
                f"Please ensure '3class_eeg_balanced_178features.csv' is in Data/Normal/ folder"
            )
        
        # Load data
        X, y, feature_names = loader.load_eeg_data(data_file)
        trainer.feature_names = feature_names
        
        # Prepare data
        X_prepared, y_prepared = trainer.prepare_data(X, y)
        
        # Train model
        model, accuracy = trainer.train(X_prepared, y_prepared)
        
        # Save model - matches your existing path structure
        model_path = 'ml_models/trained_models/tabnet_model.pkl'
        trainer.save_model(model_path)
        
        print("\n" + "="*60)
        print("✅ TabNet Training Complete!")
        print(f"   Final Test Accuracy: {accuracy:.2%}")
        if accuracy >= 0.90:
            print("   🎯 TARGET ACHIEVED: ≥90% accuracy!")
        print(f"   Model saved at: {model_path}")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\n❌ File Error: {e}")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
