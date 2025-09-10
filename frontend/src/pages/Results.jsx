// src/pages/Results.jsx
import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "../css/results.css";
import { FiArrowLeft, FiDownload, FiUser, FiCalendar, FiFileText } from "react-icons/fi";

export default function Results() {
  const navigate = useNavigate();
  const [analysisData, setAnalysisData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Load the latest analysis from localStorage
    const latestAnalysis = localStorage.getItem('latestAnalysis');
    if (latestAnalysis) {
      setAnalysisData(JSON.parse(latestAnalysis));
    }
    setLoading(false);
  }, []);

  const formatConfidence = (confidence) => {
    return `${confidence.toFixed(1)}%`;
  };

  const getResultColor = (predictedClass, confidence) => {
    if (confidence < 60) return '#f39c12';
    return predictedClass === 'Seizure Detected' ? '#e74c3c' : '#27ae60';
  };

  const handleDownloadReport = () => {
    if (!analysisData) return;
    
    const reportData = {
      patient: analysisData.patientInfo,
      analysis: analysisData.results,
      timestamp: analysisData.timestamp,
      fileName: analysisData.fileName
    };
    
    const dataStr = JSON.stringify(reportData, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `EEG_Analysis_Report_${analysisData.patientInfo.name}_${new Date().toLocaleDateString()}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  if (loading) {
    return (
      <div className="results-page">
        <div className="loading-container">
          <div className="loader"></div>
          <p>Loading results...</p>
        </div>
      </div>
    );
  }

  if (!analysisData) {
    return (
      <div className="results-page">
        <div className="no-data-container">
          <h2>No Analysis Results Available</h2>
          <p>Please upload an EEG file first to see analysis results.</p>
          <button 
            className="btn-primary"
            onClick={() => navigate('/upload')}
          >
            Go to Upload
          </button>
        </div>
      </div>
    );
  }

  const { results, patientInfo, timestamp, fileName } = analysisData;

  return (
    <div className="results-page">
      <div className="results-header">
        <div className="header-actions">
          <button 
            className="btn-back"
            onClick={() => navigate('/upload')}
          >
            <FiArrowLeft /> Back to Upload
          </button>
          <button 
            className="btn-download"
            onClick={handleDownloadReport}
          >
            <FiDownload /> Download Report
          </button>
        </div>
        <h1>Analysis Results</h1>
        <p className="processed-time">
          <FiCalendar /> Processed: {new Date(timestamp).toLocaleString()}
        </p>
      </div>

      {/* Patient Information Card */}
      <div className="info-card">
        <h3><FiUser /> Patient Information</h3>
        <div className="patient-details">
          <div className="detail-item">
            <strong>Name:</strong> {patientInfo.name}
          </div>
          {patientInfo.age && (
            <div className="detail-item">
              <strong>Age:</strong> {patientInfo.age} years
            </div>
          )}
          {patientInfo.medicalId && (
            <div className="detail-item">
              <strong>Medical ID:</strong> {patientInfo.medicalId}
            </div>
          )}
          <div className="detail-item">
            <strong>File:</strong> {fileName}
          </div>
        </div>
      </div>

      {/* Primary Result Alert */}
      <div className={`result-alert ${results.QDA.predicted_class === 'Seizure Detected' ? 'alert-danger' : 'alert-success'}`}>
        <div className="alert-content">
          <h2>{results.QDA.predicted_class}</h2>
          <p>Confidence: {formatConfidence(results.QDA.confidence)}</p>
          <small>Model: QDA Classifier</small>
        </div>
      </div>

      {/* Model Comparison */}
      <div className="comparison-section">
        <h3>Model Comparison</h3>
        <div className="models-grid">
          <div className="model-card">
            <h4>QDA</h4>
            <div className="model-metrics">
              <div className="metric-item">
                <span className="metric-label">Accuracy</span>
                <span className="metric-value">87.6%</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Sensitivity</span>
                <span className="metric-value">89.2%</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Specificity</span>
                <span className="metric-value">86.1%</span>
              </div>
            </div>
            <div 
              className="prediction-result"
              style={{ color: getResultColor(results.QDA.predicted_class, results.QDA.confidence) }}
            >
              <div className="prediction-class">{results.QDA.predicted_class}</div>
              <div className="prediction-confidence">{formatConfidence(results.QDA.confidence)}</div>
            </div>
          </div>

          <div className="model-card">
            <h4>TabNet</h4>
            <div className="model-metrics">
              <div className="metric-item">
                <span className="metric-label">Accuracy</span>
                <span className="metric-value">94.2%</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Sensitivity</span>
                <span className="metric-value">91.5%</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Specificity</span>
                <span className="metric-value">96.8%</span>
              </div>
            </div>
            <div 
              className="prediction-result"
              style={{ color: getResultColor(results.TabNet.predicted_class, results.TabNet.confidence) }}
            >
              <div className="prediction-class">{results.TabNet.predicted_class}</div>
              <div className="prediction-confidence">{formatConfidence(results.TabNet.confidence)}</div>
            </div>
          </div>
        </div>
      </div>

      {/* Feature Importance */}
      <div className="features-section">
        <h3>Feature Importance</h3>
        <div className="features-grid">
          <div className="feature-item">
            <strong>File Size:</strong> {(results.features_used.file_info.file_size / (1024*1024)).toFixed(1)} MB
          </div>
          <div className="feature-item">
            <strong>Dominant Frequency:</strong> {results.features_used.temporal.dominant_frequency} Hz
          </div>
          <div className="feature-item">
            <strong>Peak Frequency:</strong> {results.features_used.temporal.peak_frequency} Hz
          </div>
          <div className="feature-item">
            <strong>Signal Energy:</strong> {results.features_used.temporal.signal_energy}
          </div>
        </div>
      </div>

      {/* Additional Notes */}
      {patientInfo.notes && (
        <div className="notes-section">
          <h3><FiFileText /> Additional Notes</h3>
          <p>{patientInfo.notes}</p>
        </div>
      )}
    </div>
  );
}
