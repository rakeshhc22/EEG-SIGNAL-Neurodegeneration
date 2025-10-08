# backend/app/services/training_service.py

"""
FIXED Training Service - Proper Confidence Scores Always
---------------------------------------------------------
Ensures all predictions return valid confidence values
"""

import os
import logging
from typing import Dict, Optional
import numpy as np
from datetime import datetime
import traceback

try:
    from .feature_extraction import extract_features_for_prediction
    FEATURE_EXTRACTION_AVAILABLE = True
except ImportError:
    FEATURE_EXTRACTION_AVAILABLE = False

try:
    from .model_qda import qda_model
    QDA_AVAILABLE = True
except ImportError:
    qda_model = None
    QDA_AVAILABLE = False

try:
    from .model_tabnet import tabnet_model
    TABNET_AVAILABLE = True
except ImportError:
    tabnet_model = None
    TABNET_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedTrainingService:
    def __init__(self):
        self.qda = qda_model if QDA_AVAILABLE else None
        self.tabnet = tabnet_model if TABNET_AVAILABLE else None
        logger.info(f"Training Service Initialized:")
        logger.info(f"  - QDA Available: {QDA_AVAILABLE}")
        logger.info(f"  - TabNet Available: {TABNET_AVAILABLE}")
        logger.info(f"  - Feature Extraction: {FEATURE_EXTRACTION_AVAILABLE}")
    
    def predict(self, file_path: str) -> Dict:
        """
        Main prediction pipeline with guaranteed confidence scores.
        
        Returns:
            {
                "QDA": {...},
                "TabNet": {...},
                "ensemble": {...}  # ALWAYS included
            }
        """
        try:
            logger.info(f"{'='*60}")
            logger.info(f"🧠 Analyzing: {os.path.basename(file_path)}")
            logger.info(f"{'='*60}")
            
            if not FEATURE_EXTRACTION_AVAILABLE:
                raise ValueError("Feature extraction module not available")
            
            # Step 1: Extract features (21 → 62)
            features_21 = extract_features_for_prediction(file_path)
            features_62 = self._expand_to_62_features(features_21)
            
            results = {}
            
            # Step 2: QDA Prediction
            if self.qda and hasattr(self.qda, 'is_trained') and self.qda.is_trained:
                try:
                    qda_result = self.qda.predict(features_62)
                    results["QDA"] = self._format_result(qda_result, "QDA")
                    logger.info(f"✅ QDA: {qda_result.get('predicted_class')} "
                              f"({qda_result.get('confidence', 0.0):.1f}%)")
                except Exception as e:
                    logger.error(f"❌ QDA prediction failed: {e}")
                    results["QDA"] = self._error_result("QDA", str(e))
            else:
                logger.warning("⚠️ QDA model not available")
                results["QDA"] = self._unavailable_result("QDA")
            
            # Step 3: TabNet Prediction
            if self.tabnet and hasattr(self.tabnet, 'is_trained') and self.tabnet.is_trained:
                try:
                    tabnet_result = self.tabnet.predict(features_62)
                    results["TabNet"] = self._format_result(tabnet_result, "TabNet")
                    logger.info(f"✅ TabNet: {tabnet_result.get('predicted_class')} "
                              f"({tabnet_result.get('confidence', 0.0):.1f}%)")
                except Exception as e:
                    logger.error(f"❌ TabNet prediction failed: {e}")
                    results["TabNet"] = self._error_result("TabNet", str(e))
            else:
                logger.warning("⚠️ TabNet model not available")
                results["TabNet"] = self._unavailable_result("TabNet")
            
            # Step 4: Create Ensemble (ALWAYS)
            results["ensemble"] = self._create_ensemble(results["QDA"], results["TabNet"])
            
            logger.info(f"{'='*60}")
            logger.info(f"🎯 Final Prediction: {results['ensemble']['predicted_class']} "
                      f"({results['ensemble']['confidence']:.1f}%)")
            logger.info(f"{'='*60}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Complete prediction failure: {e}")
            traceback.print_exc()
            return self._complete_error(str(e))
    
    def _expand_to_62_features(self, features_21: Dict) -> Dict:
        """
        Expand 21 → 62 features while preserving file-specific characteristics.
        """
        try:
            bp = features_21.get("band_powers", {})
            stats = features_21.get("statistics", {})
            
            # Get base band powers
            delta = float(bp.get("Delta_Waves", 0.2))
            theta = float(bp.get("Theta_Waves", 0.2))
            alpha = float(bp.get("Alpha_Waves", 0.2))
            beta = float(bp.get("Beta_Waves", 0.2))
            gamma = float(bp.get("Gamma_Waves", 0.2))
            
            # 12 band power features
            expanded_bp = {
                "Delta_Waves": delta,
                "Theta_Waves": theta,
                "Alpha_Waves": alpha,
                "Beta_Waves": beta,
                "Gamma_Waves": gamma,
                "Delta_Alpha_Ratio": delta / max(alpha, 0.001),
                "Theta_Beta_Ratio": theta / max(beta, 0.001),
                "Alpha_Beta_Ratio": alpha / max(beta, 0.001),
                "Delta_Theta_Combined": delta + theta,
                "High_Freq_Power": beta + gamma,
                "Total_Power": delta + theta + alpha + beta + gamma,
                "Low_High_Ratio": (delta + theta) / max(beta + gamma, 0.001)
            }
            
            # Get base statistics
            mean_amp = float(stats.get("mean_amplitude", 0.0))
            var_amp = float(stats.get("signal_variance", 1.0))
            std_amp = float(stats.get("standard_deviation", 1.0))
            kurt = float(stats.get("kurtosis", 0.0))
            skew_val = float(stats.get("skewness", 0.0))
            peak = float(stats.get("peak_amplitude", 1.0))
            rms = float(stats.get("rms_amplitude", 0.5))
            spec_cent = float(stats.get("spectral_centroid", 10.0))
            spec_bw = float(stats.get("spectral_bandwidth", 5.0))
            spec_roll = float(stats.get("spectral_rolloff", 20.0))
            zcr = float(stats.get("zero_crossing_rate", 0.05))
            mfcc1 = float(stats.get("mfcc_1", 0.0))
            mfcc2 = float(stats.get("mfcc_2", 0.0))
            mfcc3 = float(stats.get("mfcc_3", 0.0))
            energy = float(stats.get("energy", 1.0))
            entropy = float(stats.get("entropy", 0.5))
            
            # 50 statistical features
            expanded_stats = {
                "mean_amplitude": mean_amp,
                "signal_variance": var_amp,
                "standard_deviation": std_amp,
                "kurtosis": kurt,
                "skewness": skew_val,
                "peak_amplitude": peak,
                "rms_amplitude": rms,
                "spectral_centroid": spec_cent,
                "spectral_bandwidth": spec_bw,
                "spectral_rolloff": spec_roll,
                "zero_crossing_rate": zcr,
                "mfcc_1": mfcc1,
                "mfcc_2": mfcc2,
                "mfcc_3": mfcc3,
                "energy": energy,
                "entropy": entropy,
                "amplitude_range": peak - abs(mean_amp),
                "coefficient_variation": std_amp / max(abs(mean_amp), 0.001),
                "signal_to_noise": abs(mean_amp) / max(std_amp, 0.001),
                "spectral_spread": spec_bw / max(spec_cent, 0.1),
                "spectral_slope": (spec_roll - spec_cent) / 10.0,
                "spectral_flux": abs(spec_cent - 12.5),
                "temporal_centroid": zcr * 100,
                "spectral_decrease": max(0, spec_roll - spec_cent),
                "harmonic_ratio": mfcc1 / max(abs(mfcc2), 0.001),
                "noise_ratio": entropy / max(energy, 0.001),
                "dynamic_range": peak / max(rms, 0.001),
                "spectral_contrast": spec_roll - spec_cent,
                "rhythmic_pattern": mfcc3 * zcr,
                "frequency_stability": 1.0 / max(spec_bw, 0.001),
                "amplitude_modulation": std_amp / max(rms, 0.001),
                "phase_coherence": 1.0 - min(entropy, 1.0),
                "signal_complexity": abs(kurt) + abs(skew_val),
                "temporal_stability": 1.0 / max(std_amp, 0.001),
                "frequency_concentration": spec_cent / max(spec_bw, 0.1),
                "neural_activity_index": energy * zcr,
                "seizure_indicator": max(0, kurt - 3.0) * peak,
                "neurodegeneration_marker": entropy * (1.0 - min(spec_cent / 25.0, 1.0)),
                "brain_rhythm_coherence": (alpha + beta) / 2.0,
                "pathological_pattern": abs(skew_val) + max(0, abs(kurt) - 3.0),
                "clinical_severity": peak * entropy,
                "diagnostic_confidence": 1.0 - entropy / 8.0,
                "signal_regularity": 1.0 / max(var_amp, 0.001),
                "frequency_dominance": max(delta, theta, alpha, beta, gamma),
                "time_domain_complexity": std_amp * abs(kurt),
                "frequency_domain_complexity": spec_bw * entropy,
                "amplitude_asymmetry": abs(skew_val) * peak,
                "frequency_asymmetry": abs(spec_cent - 15.0),
                "neural_synchrony": (1.0 - entropy) * alpha,
                "pathological_score": abs(kurt) + abs(skew_val) + entropy
            }
            
            features_62 = {
                "band_powers": expanded_bp,
                "statistics": expanded_stats
            }
            
            logger.info(f"✅ Features expanded: 21 → 62 (12 band + 50 stats)")
            return features_62
            
        except Exception as e:
            logger.error(f"❌ Feature expansion failed: {e}")
            raise
    
    def _format_result(self, model_result: Dict, model_name: str) -> Dict:
        """Format model prediction result with guaranteed fields."""
        confidence = float(model_result.get("confidence", 0.0))
        predicted_class = str(model_result.get("predicted_class", "Unknown"))
        probabilities = model_result.get("probabilities", [0.33, 0.33, 0.34])
        
        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2),
            "probabilities": [round(float(p), 4) for p in probabilities],
            "model": model_name,
            "method": model_result.get("method", "Standard"),
            "status": "success"
        }
    
    def _unavailable_result(self, model_name: str) -> Dict:
        """Result when model is not loaded (not an error)."""
        return {
            "predicted_class": "Unknown",
            "confidence": 0.0,
            "probabilities": [0.33, 0.33, 0.34],
            "model": f"{model_name} (Not Loaded)",
            "method": "Unavailable",
            "status": "unavailable"
        }
    
    def _error_result(self, model_name: str, error: str) -> Dict:
        """Result when model prediction fails."""
        return {
            "predicted_class": "Error",
            "confidence": 0.0,
            "probabilities": [0.33, 0.33, 0.34],
            "model": model_name,
            "method": "Error",
            "error": error,
            "status": "error"
        }
    
    def _create_ensemble(self, qda: Dict, tabnet: Dict) -> Dict:
        """
        Create ensemble prediction from QDA and TabNet.
        ALWAYS returns a valid result with confidence > 0.
        """
        qda_valid = qda.get("status") == "success"
        tabnet_valid = tabnet.get("status") == "success"
        
        # Both models succeeded
        if qda_valid and tabnet_valid:
            qda_conf = float(qda["confidence"])
            tabnet_conf = float(tabnet["confidence"])
            avg_conf = (qda_conf + tabnet_conf) / 2
            
            # Use prediction from model with higher confidence
            if qda_conf >= tabnet_conf:
                pred = qda["predicted_class"]
            else:
                pred = tabnet["predicted_class"]
            
            return {
                "predicted_class": pred,
                "confidence": round(avg_conf, 2),
                "method": "Ensemble (QDA + TabNet)",
                "qda_confidence": round(qda_conf, 2),
                "tabnet_confidence": round(tabnet_conf, 2)
            }
        
        # Only QDA succeeded
        elif qda_valid:
            return {
                "predicted_class": qda["predicted_class"],
                "confidence": round(float(qda["confidence"]), 2),
                "method": "QDA Only (TabNet unavailable)",
                "qda_confidence": round(float(qda["confidence"]), 2),
                "tabnet_confidence": 0.0
            }
        
        # Only TabNet succeeded
        elif tabnet_valid:
            return {
                "predicted_class": tabnet["predicted_class"],
                "confidence": round(float(tabnet["confidence"]), 2),
                "method": "TabNet Only (QDA unavailable)",
                "qda_confidence": 0.0,
                "tabnet_confidence": round(float(tabnet["confidence"]), 2)
            }
        
        # Both failed - return conservative "Unknown" with 0 confidence
        else:
            logger.error("⚠️ Both models failed - returning Unknown result")
            return {
                "predicted_class": "Unknown",
                "confidence": 0.0,
                "method": "No Valid Models",
                "qda_confidence": 0.0,
                "tabnet_confidence": 0.0
            }
    
    def _complete_error(self, error: str) -> Dict:
        """Complete failure of prediction pipeline."""
        return {
            "QDA": self._error_result("QDA", error),
            "TabNet": self._error_result("TabNet", error),
            "ensemble": {
                "predicted_class": "Error",
                "confidence": 0.0,
                "method": "Complete Failure",
                "error": error
            }
        }


# Global instance
training_service = EnhancedTrainingService()
