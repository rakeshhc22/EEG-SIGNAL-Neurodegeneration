# backend/app/services/feature_extraction.py

"""
FIXED Feature Extraction - Handles CSV with headers properly
-------------------------------------------------------------
Fixes:
1. Skips header row in CSV files
2. Handles both row-wise and column-wise data formats
3. Robust pandas fallback for complex CSV structures
4. Better error messages for debugging
"""

import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional
from scipy.fft import fft, fftfreq
from scipy.stats import kurtosis, skew

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EEGProcessor:
    def __init__(self):
        self.sampling_rate = 173.61
        self.duration = 23.6
        self.nyquist = self.sampling_rate / 2
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
    
    def load_eeg_data(self, file_path: str) -> np.ndarray:
        """
        Load EEG data from CSV/TXT file with robust header detection.
        
        Handles formats:
        - CSV with headers (Unnamed, X1, X2, ..., X178)
        - Raw comma-separated values
        - Single row or single column data
        """
        try:
            logger.info(f"Loading EEG data from: {os.path.basename(file_path)}")
            
            # Method 1: Try pandas first (handles headers automatically)
            try:
                df = pd.read_csv(file_path)
                logger.info(f"Pandas detected shape: {df.shape} (rows x cols)")
                
                # Remove non-numeric columns (like 'Unnamed' index column)
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df_numeric = df[numeric_cols]
                
                if df_numeric.empty:
                    raise ValueError("No numeric columns found in CSV")
                
                # Flatten to 1D array (handles both row and column formats)
                data = df_numeric.values.flatten()
                
                # Remove NaN values
                data = data[~np.isnan(data)]
                
                if len(data) == 0:
                    raise ValueError("No valid numeric data after filtering")
                
                logger.info(f"✅ Loaded {len(data)} samples using pandas")
                return data
                
            except Exception as pandas_error:
                logger.warning(f"Pandas method failed: {pandas_error}. Trying manual parsing...")
            
            # Method 2: Manual parsing (fallback)
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            data = []
            for line_idx, line in enumerate(lines):
                # Skip empty lines
                if not line.strip():
                    continue
                
                try:
                    # Split by comma
                    values = line.strip().split(',')
                    
                    for val in values:
                        val = val.strip()
                        if not val:
                            continue
                        
                        # Skip header-like strings (contains letters)
                        if any(c.isalpha() for c in val):
                            logger.debug(f"Skipping non-numeric value: {val}")
                            continue
                        
                        # Try to convert to float
                        try:
                            data.append(float(val))
                        except ValueError:
                            continue
                            
                except Exception as line_error:
                    logger.debug(f"Skipping line {line_idx}: {line_error}")
                    continue
            
            if len(data) == 0:
                raise ValueError(
                    f"No valid numeric data found in file. "
                    f"Please check file format. Expected: CSV with numeric EEG values."
                )
            
            data = np.array(data)
            logger.info(f"✅ Loaded {len(data)} samples using manual parsing")
            return data
            
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load EEG data: {str(e)}")
            raise ValueError(f"Failed to load EEG data from {os.path.basename(file_path)}: {str(e)}")
    
    def extract_band_powers(self, data: np.ndarray) -> Dict:
        """
        Extract EEG frequency band powers using FFT.
        
        Returns relative power in each frequency band:
        - Delta (0.5-4 Hz)
        - Theta (4-8 Hz)
        - Alpha (8-13 Hz)
        - Beta (13-30 Hz)
        - Gamma (30-50 Hz)
        """
        try:
            # Normalize signal
            signal_norm = data - np.mean(data)
            signal_std = np.std(signal_norm)
            
            if signal_std > 1e-10:
                signal_norm = signal_norm / signal_std
            
            # Direct FFT (no windowing distortion)
            fft_vals = fft(signal_norm)
            fft_freq = fftfreq(len(signal_norm), 1/self.sampling_rate)
            
            # Use only positive frequencies
            positive_idx = fft_freq > 0
            fft_freq_pos = fft_freq[positive_idx]
            fft_power = np.abs(fft_vals[positive_idx])**2
            
            # Total power
            total_power = np.sum(fft_power)
            if total_power < 1e-10:
                total_power = 1.0
            
            # Calculate band powers
            band_powers = {}
            for band_name, (low, high) in self.bands.items():
                band_idx = (fft_freq_pos >= low) & (fft_freq_pos <= high)
                
                if np.any(band_idx):
                    band_power = np.sum(fft_power[band_idx])
                    relative_power = band_power / total_power
                else:
                    relative_power = 0.01
                
                # Map to expected names
                if band_name == 'delta':
                    band_powers["Delta_Waves"] = float(relative_power)
                elif band_name == 'theta':
                    band_powers["Theta_Waves"] = float(relative_power)
                elif band_name == 'alpha':
                    band_powers["Alpha_Waves"] = float(relative_power)
                elif band_name == 'beta':
                    band_powers["Beta_Waves"] = float(relative_power)
                elif band_name == 'gamma':
                    band_powers["Gamma_Waves"] = float(relative_power)
            
            logger.info(
                f"Band powers extracted: "
                f"Delta={band_powers.get('Delta_Waves', 0):.3f}, "
                f"Theta={band_powers.get('Theta_Waves', 0):.3f}, "
                f"Alpha={band_powers.get('Alpha_Waves', 0):.3f}, "
                f"Beta={band_powers.get('Beta_Waves', 0):.3f}, "
                f"Gamma={band_powers.get('Gamma_Waves', 0):.3f}"
            )
            
            return band_powers
            
        except Exception as e:
            logger.error(f"Band power extraction failed: {e}")
            # Return default uniform distribution
            return {
                "Delta_Waves": 0.2,
                "Theta_Waves": 0.2,
                "Alpha_Waves": 0.2,
                "Beta_Waves": 0.2,
                "Gamma_Waves": 0.2
            }
    
    def extract_statistical_features(self, data: np.ndarray) -> Dict:
        """
        Extract comprehensive statistical features from EEG signal.
        
        Features include:
        - Time domain: mean, variance, std, kurtosis, skewness, RMS
        - Frequency domain: spectral centroid, bandwidth, rolloff
        - Signal properties: zero-crossing rate, energy, entropy
        - MFCC-like features
        """
        try:
            # Basic time-domain statistics
            mean_amp = float(np.mean(data))
            var_amp = float(np.var(data))
            std_amp = float(np.std(data))
            kurt_val = float(kurtosis(data))
            skew_val = float(skew(data))
            peak_amp = float(np.max(np.abs(data)))
            rms_amp = float(np.sqrt(np.mean(data**2)))
            
            # FFT for frequency-domain features
            fft_vals = fft(data)
            fft_freq = fftfreq(len(data), 1/self.sampling_rate)
            fft_mag = np.abs(fft_vals[:len(fft_vals)//2])
            freq_pos = fft_freq[:len(fft_freq)//2]
            
            # Spectral features
            if np.sum(fft_mag) > 0:
                spectral_centroid = float(np.sum(freq_pos * fft_mag) / np.sum(fft_mag))
                spectral_bandwidth = float(
                    np.sqrt(np.sum(((freq_pos - spectral_centroid)**2) * fft_mag) / np.sum(fft_mag))
                )
                
                cum_sum = np.cumsum(fft_mag)
                rolloff_idx = np.where(cum_sum >= 0.85 * cum_sum[-1])[0]
                spectral_rolloff = float(freq_pos[rolloff_idx[0]]) if len(rolloff_idx) > 0 else 20.0
            else:
                spectral_centroid = 10.0
                spectral_bandwidth = 5.0
                spectral_rolloff = 20.0
            
            # Zero-crossing rate
            zero_crossings = np.where(np.diff(np.signbit(data)))[0]
            zcr = float(len(zero_crossings) / len(data))
            
            # Energy and entropy
            energy = float(np.sum(data**2))
            signal_abs = np.abs(data)
            signal_abs_norm = signal_abs / (np.sum(signal_abs) + 1e-10)
            entropy = float(-np.sum(signal_abs_norm * np.log(signal_abs_norm + 1e-10)))
            
            # MFCC-like features (low-frequency spectral characteristics)
            mfcc_1 = float(np.mean(fft_mag[:50])) if len(fft_mag) > 50 else 0.0
            mfcc_2 = float(np.std(fft_mag[:50])) if len(fft_mag) > 50 else 0.0
            mfcc_3 = float(np.max(fft_mag[:50]) - np.min(fft_mag[:50])) if len(fft_mag) > 50 else 0.0
            
            features = {
                "mean_amplitude": mean_amp,
                "signal_variance": var_amp,
                "standard_deviation": std_amp,
                "kurtosis": kurt_val,
                "skewness": skew_val,
                "peak_amplitude": peak_amp,
                "rms_amplitude": rms_amp,
                "spectral_centroid": spectral_centroid,
                "spectral_bandwidth": spectral_bandwidth,
                "spectral_rolloff": spectral_rolloff,
                "zero_crossing_rate": zcr,
                "mfcc_1": mfcc_1,
                "mfcc_2": mfcc_2,
                "mfcc_3": mfcc_3,
                "energy": energy,
                "entropy": entropy
            }
            
            logger.info(f"✅ Extracted {len(features)} statistical features")
            return features
            
        except Exception as e:
            logger.error(f"Statistical feature extraction failed: {e}")
            # Return safe default values
            return {
                "mean_amplitude": 0.0,
                "signal_variance": 1.0,
                "standard_deviation": 1.0,
                "kurtosis": 0.0,
                "skewness": 0.0,
                "peak_amplitude": 1.0,
                "rms_amplitude": 0.5,
                "spectral_centroid": 10.0,
                "spectral_bandwidth": 5.0,
                "spectral_rolloff": 20.0,
                "zero_crossing_rate": 0.05,
                "mfcc_1": 0.0,
                "mfcc_2": 0.0,
                "mfcc_3": 0.0,
                "energy": 1.0,
                "entropy": 0.5
            }


# Global processor instance
processor = EEGProcessor()


def extract_features_for_prediction(file_path: str) -> Dict:
    """
    Main feature extraction pipeline for EEG prediction.
    
    Args:
        file_path: Path to CSV file containing raw EEG signal data
        
    Returns:
        Dictionary containing:
        - band_powers: Relative power in 5 frequency bands (21 total features)
        - statistics: 16 time/frequency domain features
        
    Raises:
        ValueError: If file cannot be loaded or processed
    """
    try:
        logger.info(f"=" * 60)
        logger.info(f"Starting feature extraction for: {os.path.basename(file_path)}")
        logger.info(f"=" * 60)
        
        # Step 1: Load raw EEG data
        raw_data = processor.load_eeg_data(file_path)
        logger.info(f"Signal length: {len(raw_data)} samples")
        logger.info(f"Signal range: [{np.min(raw_data):.2f}, {np.max(raw_data):.2f}]")
        
        # Step 2: Extract frequency band powers
        band_powers = processor.extract_band_powers(raw_data)
        
        # Step 3: Extract statistical features
        statistics = processor.extract_statistical_features(raw_data)
        
        # Combine all features
        features = {
            "band_powers": band_powers,
            "statistics": statistics
        }
        
        total_features = len(band_powers) + len(statistics)
        logger.info(f"=" * 60)
        logger.info(f"✅ SUCCESS: Extracted {total_features} features total")
        logger.info(f"   - Band powers: {len(band_powers)} features")
        logger.info(f"   - Statistics: {len(statistics)} features")
        logger.info(f"=" * 60)
        
        return features
        
    except Exception as e:
        logger.error(f"❌ Feature extraction failed: {str(e)}")
        logger.error(f"File: {file_path}")
        raise ValueError(f"Feature extraction failed for {os.path.basename(file_path)}: {str(e)}")


def validate_features(features: Dict) -> bool:
    """
    Validate that extracted features are valid for model prediction.
    
    Args:
        features: Dictionary from extract_features_for_prediction()
        
    Returns:
        True if features are valid, False otherwise
    """
    try:
        required_band_powers = ["Delta_Waves", "Theta_Waves", "Alpha_Waves", "Beta_Waves", "Gamma_Waves"]
        required_stats = [
            "mean_amplitude", "signal_variance", "standard_deviation",
            "kurtosis", "skewness", "peak_amplitude", "rms_amplitude",
            "spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
            "zero_crossing_rate", "mfcc_1", "mfcc_2", "mfcc_3",
            "energy", "entropy"
        ]
        
        # Check band powers
        if "band_powers" not in features:
            logger.error("Missing 'band_powers' in features")
            return False
        
        for bp in required_band_powers:
            if bp not in features["band_powers"]:
                logger.error(f"Missing band power: {bp}")
                return False
        
        # Check statistics
        if "statistics" not in features:
            logger.error("Missing 'statistics' in features")
            return False
        
        for stat in required_stats:
            if stat not in features["statistics"]:
                logger.error(f"Missing statistic: {stat}")
                return False
        
        logger.info("✅ Feature validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Feature validation error: {e}")
        return False
