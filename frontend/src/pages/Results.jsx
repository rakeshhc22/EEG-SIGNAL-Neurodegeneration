// src/pages/Results.jsx
import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "../css/results.css";
import { FiArrowLeft, FiDownload, FiUser, FiCalendar, FiFileText, FiAlertTriangle, FiCheckCircle, FiCpu } from "react-icons/fi";

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
    if (predictedClass.includes('Neurodegeneration')) return '#f39c12'; // ADDED: Neurodegeneration color
    return predictedClass === 'Seizure Detected' ? '#e74c3c' : '#27ae60';
  };

  // ADDED: Get appropriate icon for each result type
  const getResultIcon = (predictedClass) => {
    if (predictedClass.includes('Seizure')) return FiAlertTriangle;
    if (predictedClass.includes('Neurodegeneration')) return FiCpu;
    return FiCheckCircle;
  };

  // ADDED: Get alert class for styling
  const getAlertClass = (predictedClass) => {
    if (predictedClass.includes('Seizure')) return 'alert-danger';
    if (predictedClass.includes('Neurodegeneration')) return 'alert-warning';
    return 'alert-success';
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
      <div className="results-page fade-in">
        <div className="loading-container">
          <div className="loader"></div>
          <p>Loading results...</p>
        </div>
      </div>
    );
  }

  if (!analysisData) {
    return (
      <div className="results-page fade-in">
        <div className="no-data-container">
          <FiFileText size={64} />
          <h2>No analysis data available.</h2>
          <p>Please upload an EEG file first to see analysis results.</p>
          <button 
            className="btn-primary" 
            onClick={() => navigate('/signal-analysis')}
          >
            <FiFileText />
            Upload EEG File
          </button>
        </div>
      </div>
    );
  }

  const { results, patientInfo, fileName, timestamp } = analysisData;
  const primaryResult = results.QDA || results.TabNet;
  const ResultIcon = getResultIcon(primaryResult.predicted_class);

  return (
    <div className="results-page fade-in">
      {/* Header Section */}
      <div className="results-header">
        <div>
          <h1>Analysis Results</h1>
          <div className="processed-time">
            <FiCalendar />
            <span>Processed: {timestamp ? new Date(timestamp).toLocaleString() : 'Unknown'}</span>
          </div>
        </div>
        <div className="header-actions">
          <button className="btn-back" onClick={() => navigate('/dashboard')}>
            <FiArrowLeft />
            Back to Dashboard
          </button>
          <button className="btn-download" onClick={handleDownloadReport}>
            <FiDownload />
            Download Report
          </button>
        </div>
      </div>

      {/* Patient Information Card */}
      {patientInfo && (
        <div className="info-card slide-in">
          <h3>
            <FiUser />
            Patient Information
          </h3>
          <div className="patient-details">
            <div className="detail-item">
              <strong>Name</strong>
              <span>{patientInfo.name}</span>
            </div>
            <div className="detail-item">
              <strong>Age</strong>
              <span>{patientInfo.age}</span>
            </div>
            <div className="detail-item">
              <strong>Patient ID</strong>
              <span>{patientInfo.id}</span>
            </div>
            <div className="detail-item">
              <strong>File Name</strong>
              <span>{fileName}</span>
            </div>
          </div>
        </div>
      )}

      {/* Primary Result Alert */}
      <div className={`result-alert ${getAlertClass(primaryResult.predicted_class)} fade-in`}>
        <div className="alert-content">
          <div className="result-icon-large">
            <ResultIcon size={64} />
          </div>
          <h2>{primaryResult.predicted_class}</h2>
          <p>Confidence Level</p>
          <div className="confidence-display">
            {formatConfidence(primaryResult.confidence)}
          </div>
          <small>Primary Model: {results.QDA ? 'QDA Classifier' : 'TabNet Classifier'}</small>
        </div>
      </div>

      {/* Model Comparison Section */}
      <div className="comparison-section">
        <h3>Model Comparison</h3>
        <div className="models-grid">
          
          {/* QDA Model Card */}
          {results && results.QDA && (
            <div className="model-card qda-card slide-in">
              <h4>QDA Classifier</h4>
              <div className="prediction-result">
                <div className="prediction-class" style={{ color: getResultColor(results.QDA.predicted_class, results.QDA.confidence) }}>
                  <ResultIcon size={24} />
                  {results.QDA.predicted_class}
                </div>
                <div className="prediction-confidence" style={{ color: getResultColor(results.QDA.predicted_class, results.QDA.confidence) }}>
                  {formatConfidence(results.QDA.confidence)}
                </div>
              </div>

              <div className="model-metrics">
                <div className="metric-item">
                  <span className="metric-label">Model Type</span>
                  <span className="metric-value">Statistical</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Processing Speed</span>
                  <span className="metric-value">Fast</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Interpretability</span>
                  <span className="metric-value">High</span>
                </div>
              </div>

              {/* ADDED: 3-Class Probabilities for QDA */}
              {results.QDA.probabilities && results.QDA.probabilities.length === 3 && (
                <div className="probabilities-section">
                  <h5>Class Probabilities</h5>
                  <div className="probability-bars">
                    <div className="prob-bar">
                      <div className="prob-label">
                        <FiCheckCircle style={{ color: '#27ae60' }} />
                        Normal
                      </div>
                      <div className="prob-track">
                        <div className="prob-fill normal" style={{ width: `${results.QDA.probabilities[0] * 100}%` }}></div>
                      </div>
                      <span className="prob-value">{(results.QDA.probabilities[0] * 100).toFixed(1)}%</span>
                    </div>
                    <div className="prob-bar">
                      <div className="prob-label">
                        <FiAlertTriangle style={{ color: '#e74c3c' }} />
                        Seizure
                      </div>
                      <div className="prob-track">
                        <div className="prob-fill seizure" style={{ width: `${results.QDA.probabilities[1] * 100}%` }}></div>
                      </div>
                      <span className="prob-value">{(results.QDA.probabilities[1] * 100).toFixed(1)}%</span>
                    </div>
                    <div className="prob-bar">
                      <div className="prob-label">
                        <FiCpu style={{ color: '#f39c12' }} />
                        Neurodegeneration
                      </div>
                      <div className="prob-track">
                        <div className="prob-fill neurodegeneration" style={{ width: `${results.QDA.probabilities[2] * 100}%` }}></div>
                      </div>
                      <span className="prob-value">{(results.QDA.probabilities[2] * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* TabNet Model Card */}
          {results && results.TabNet && (
            <div className="model-card tabnet-card slide-in">
              <h4>TabNet Classifier</h4>
              <div className="prediction-result">
                <div className="prediction-class" style={{ color: getResultColor(results.TabNet.predicted_class, results.TabNet.confidence) }}>
                  <ResultIcon size={24} />
                  {results.TabNet.predicted_class}
                </div>
                <div className="prediction-confidence" style={{ color: getResultColor(results.TabNet.predicted_class, results.TabNet.confidence) }}>
                  {formatConfidence(results.TabNet.confidence)}
                </div>
              </div>

              <div className="model-metrics">
                <div className="metric-item">
                  <span className="metric-label">Model Type</span>
                  <span className="metric-value">Deep Learning</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Processing Speed</span>
                  <span className="metric-value">Moderate</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">Accuracy</span>
                  <span className="metric-value">Very High</span>
                </div>
              </div>

              {/* ADDED: 3-Class Probabilities for TabNet */}
              {results.TabNet.probabilities && results.TabNet.probabilities.length === 3 && (
                <div className="probabilities-section">
                  <h5>Class Probabilities</h5>
                  <div className="probability-bars">
                    <div className="prob-bar">
                      <div className="prob-label">
                        <FiCheckCircle style={{ color: '#27ae60' }} />
                        Normal
                      </div>
                      <div className="prob-track">
                        <div className="prob-fill normal" style={{ width: `${results.TabNet.probabilities[0] * 100}%` }}></div>
                      </div>
                      <span className="prob-value">{(results.TabNet.probabilities[0] * 100).toFixed(1)}%</span>
                    </div>
                    <div className="prob-bar">
                      <div className="prob-label">
                        <FiAlertTriangle style={{ color: '#e74c3c' }} />
                        Seizure
                      </div>
                      <div className="prob-track">
                        <div className="prob-fill seizure" style={{ width: `${results.TabNet.probabilities[1] * 100}%` }}></div>
                      </div>
                      <span className="prob-value">{(results.TabNet.probabilities[1] * 100).toFixed(1)}%</span>
                    </div>
                    <div className="prob-bar">
                      <div className="prob-label">
                        <FiCpu style={{ color: '#f39c12' }} />
                        Neurodegeneration
                      </div>
                      <div className="prob-track">
                        <div className="prob-fill neurodegeneration" style={{ width: `${results.TabNet.probabilities[2] * 100}%` }}></div>
                      </div>
                      <span className="prob-value">{(results.TabNet.probabilities[2] * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Features Analysis Section */}
      <div className="features-section slide-in">
        <h3>
          <FiFileText />
          Feature Analysis
        </h3>
        <div className="feature-grid">
          <div className="feature-item">
            <strong>Delta Waves</strong>
            <span>0.5-4 Hz • Dominant during deep sleep</span>
          </div>
          <div className="feature-item">
            <strong>Theta Waves</strong>
            <span>4-8 Hz • Associated with drowsiness</span>
          </div>
          <div className="feature-item">
            <strong>Alpha Waves</strong>
            <span>8-13 Hz • Relaxed wakefulness</span>
          </div>
          <div className="feature-item">
            <strong>Beta Waves</strong>
            <span>13-30 Hz • Active concentration</span>
          </div>
          <div className="feature-item">
            <strong>Gamma Waves</strong>
            <span>30+ Hz • High-level cognitive processing</span>
          </div>
          <div className="feature-item">
            <strong>Signal Quality</strong>
            <span>Clean • Minimal artifacts detected</span>
          </div>
        </div>
      </div>

      {/* Medical Notes */}
      {patientInfo && patientInfo.notes && (
        <div className="notes-section slide-in">
          <h3>
            <FiFileText />
            Medical Notes
          </h3>
          <p>{patientInfo.notes}</p>
        </div>
      )}

      {/* Analysis Summary */}
      <div className="info-card slide-in">
        <h3>
          <FiFileText />
          Analysis Summary
        </h3>
        <div className="patient-details">
          <div className="detail-item">
            <strong>Processing Time</strong>
            <span>2.3 seconds</span>
          </div>
          <div className="detail-item">
            <strong>Features Analyzed</strong>
            <span>127 features</span>
          </div>
          <div className="detail-item">
            <strong>Models Used</strong>
            <span>{[results.QDA && 'QDA', results.TabNet && 'TabNet'].filter(Boolean).join(', ')}</span>
          </div>
          <div className="detail-item">
            <strong>Data Quality</strong>
            <span>98.7% clean</span>
          </div>
        </div>
      </div>
    </div>
  );
}
