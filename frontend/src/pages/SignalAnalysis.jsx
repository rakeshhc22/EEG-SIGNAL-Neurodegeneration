// src/pages/SignalAnalysis.jsx
import React, { useState, useEffect, useRef } from "react";
import {
  FiZoomIn,
  FiZoomOut,
  FiRotateCcw,
  FiUpload,
  FiCheckCircle,
  FiLoader,
  FiActivity,
  FiBarChart2,
  FiSettings
} from "react-icons/fi";
import "../css/signalAnalysis.css";

export default function SignalAnalysis() {
  const [zoomLevel, setZoomLevel] = useState(1);
  const [selectedFile, setSelectedFile] = useState(null);
  const [processingStatus, setProcessingStatus] = useState({
    fileUpload: true,
    preprocessing: true,
    featureExtraction: false,
    tabnetAnalysis: false,
    qdaClassification: false
  });
  
  const [eegData, setEegData] = useState(generateEEGData());
  const [frequencyData] = useState([
    { band: 'Delta (0.5-4 Hz)', percentage: 12.5, color: '#e74c3c' },
    { band: 'Theta (4-8 Hz)', percentage: 8.3, color: '#f39c12' },
    { band: 'Alpha (8-12 Hz)', percentage: 15.2, color: '#f1c40f' },
    { band: 'Beta (12-30 Hz)', percentage: 22.1, color: '#27ae60' },
    { band: 'Gamma (30+ Hz)', percentage: 18.4, color: '#3498db' }
  ]);

  // Generate synthetic EEG data for visualization
  function generateEEGData() {
    const channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4'];
    const timePoints = 1000;
    const data = [];

    channels.forEach((channel, channelIndex) => {
      const channelData = [];
      for (let t = 0; t < timePoints; t++) {
        const time = (t / timePoints) * 10; // 10 seconds
        let amplitude = 0;
        
        // Add multiple frequency components
        amplitude += Math.sin(2 * Math.PI * 10 * time) * (Math.random() * 2 + 1); // Alpha
        amplitude += Math.sin(2 * Math.PI * 20 * time) * (Math.random() * 1 + 0.5); // Beta
        amplitude += Math.sin(2 * Math.PI * 5 * time) * (Math.random() * 1.5 + 0.5); // Theta
        
        // Add noise
        amplitude += (Math.random() - 0.5) * 0.5;
        
        // Offset each channel
        amplitude += channelIndex * 2;
        
        channelData.push({ time, amplitude });
      }
      data.push({ channel, data: channelData, color: getChannelColor(channelIndex) });
    });
    
    return data;
  }

  function getChannelColor(index) {
    const colors = [
      '#e74c3c', '#f39c12', '#f1c40f', '#27ae60',
      '#3498db', '#9b59b6', '#e67e22', '#95a5a6'
    ];
    return colors[index % colors.length];
  }

  const handleZoomIn = () => {
    setZoomLevel(prev => Math.min(prev * 1.5, 5));
  };

  const handleZoomOut = () => {
    setZoomLevel(prev => Math.max(prev / 1.5, 0.5));
  };

  const handleReset = () => {
    setZoomLevel(1);
    setEegData(generateEEGData());
  };

  const ProcessingStep = ({ icon: Icon, title, status, isActive }) => {
    const getStatusIcon = () => {
      if (status) return <FiCheckCircle className="status-complete" />;
      if (isActive) return <FiLoader className="status-loading" />;
      return <div className="status-pending"></div>;
    };

    return (
      <div className={`processing-step ${status ? 'completed' : ''} ${isActive ? 'active' : ''}`}>
        <div className="step-icon">
          <Icon size={20} />
        </div>
        <div className="step-content">
          <span className="step-title">{title}</span>
          <div className="step-status">
            {getStatusIcon()}
          </div>
        </div>
      </div>
    );
  };

  const FrequencyBar = ({ band, percentage, color }) => (
    <div className="frequency-bar">
      <div className="frequency-label">{band}</div>
      <div className="frequency-progress">
        <div 
          className="frequency-fill"
          style={{ 
            width: `${percentage * 3}px`, 
            background: color 
          }}
        ></div>
        <span className="frequency-percentage">{percentage}%</span>
      </div>
    </div>
  );

  const EEGVisualization = () => (
    <div className="eeg-visualization">
      <div className="chart-controls">
        <button className="control-btn" onClick={handleZoomIn}>
          <FiZoomIn /> Zoom In
        </button>
        <button className="control-btn" onClick={handleZoomOut}>
          <FiZoomOut /> Zoom Out
        </button>
        <button className="control-btn" onClick={handleReset}>
          <FiRotateCcw /> Reset
        </button>
      </div>

      <div className="eeg-chart-container">
        <svg 
          width="100%" 
          height="400" 
          viewBox={`0 0 ${800 * zoomLevel} 400`}
          className="eeg-chart"
        >
          {/* Grid lines */}
          {Array.from({length: Math.ceil(10 * zoomLevel)}, (_, i) => (
            <line
              key={`vline-${i}`}
              x1={i * 80}
              y1={0}
              x2={i * 80}
              y2={400}
              stroke="#e9ecef"
              strokeWidth="1"
            />
          ))}
          
          {Array.from({length: 9}, (_, i) => (
            <line
              key={`hline-${i}`}
              x1={0}
              y1={i * 50}
              x2={800 * zoomLevel}
              y2={i * 50}
              stroke="#e9ecef"
              strokeWidth="1"
            />
          ))}

          {/* EEG Channel Data */}
          {eegData.map((channelData, channelIndex) => {
            const yOffset = channelIndex * 45 + 25;
            
            const pathData = channelData.data
              .map((point, index) => {
                const x = (point.time / 10) * 800 * zoomLevel;
                const y = yOffset + (point.amplitude - channelIndex * 2) * 5;
                return `${index === 0 ? 'M' : 'L'} ${x} ${y}`;
              })
              .join(' ');

            return (
              <g key={channelData.channel}>
                {/* Channel label */}
                <text
                  x={10}
                  y={yOffset + 5}
                  fill={channelData.color}
                  fontSize="12"
                  fontWeight="600"
                >
                  {channelData.channel}
                </text>
                
                {/* Signal path */}
                <path
                  d={pathData}
                  fill="none"
                  stroke={channelData.color}
                  strokeWidth="1.5"
                  opacity="0.8"
                />
              </g>
            );
          })}

          {/* Time axis */}
          <text x={400 * zoomLevel} y={390} textAnchor="middle" fill="#6c757d" fontSize="14" fontWeight="500">
            Time (seconds)
          </text>
        </svg>

        {/* Channel Legend */}
        <div className="channel-legend">
          {eegData.map((channelData, index) => (
            <div key={channelData.channel} className="legend-item">
              <div 
                className="legend-color" 
                style={{ backgroundColor: channelData.color }}
              ></div>
              <span>{channelData.channel}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  useEffect(() => {
    // Simulate processing steps
    const steps = ['preprocessing', 'featureExtraction', 'tabnetAnalysis', 'qdaClassification'];
    let currentStep = 0;
    
    const interval = setInterval(() => {
      if (currentStep < steps.length) {
        setProcessingStatus(prev => ({
          ...prev,
          [steps[currentStep]]: true
        }));
        currentStep++;
      } else {
        clearInterval(interval);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="signal-analysis-page">
      <div className="analysis-header">
        <h1>Signal Analysis</h1>
        <p>Real-time EEG signal processing and analysis</p>
      </div>

      <div className="analysis-content">
        {/* Main EEG Visualization */}
        <div className="visualization-section">
          <div className="section-card">
            <h3>
              <FiActivity /> EEG Signal Visualization
            </h3>
            <EEGVisualization />
          </div>
        </div>

        {/* Right Sidebar */}
        <div className="analysis-sidebar">
          {/* Processing Status */}
          <div className="section-card">
            <h3>
              <FiSettings /> Processing Status
            </h3>
            <div className="processing-steps">
              <ProcessingStep
                icon={FiUpload}
                title="File Upload"
                status={processingStatus.fileUpload}
              />
              <ProcessingStep
                icon={FiSettings}
                title="Preprocessing"
                status={processingStatus.preprocessing}
                isActive={!processingStatus.preprocessing}
              />
              <ProcessingStep
                icon={FiBarChart2}
                title="Feature Extraction"
                status={processingStatus.featureExtraction}
                isActive={processingStatus.preprocessing && !processingStatus.featureExtraction}
              />
              <ProcessingStep
                icon={FiActivity}
                title="TabNet Analysis"
                status={processingStatus.tabnetAnalysis}
                isActive={processingStatus.featureExtraction && !processingStatus.tabnetAnalysis}
              />
              <ProcessingStep
                icon={FiCheckCircle}
                title="QDA Classification"
                status={processingStatus.qdaClassification}
                isActive={processingStatus.tabnetAnalysis && !processingStatus.qdaClassification}
              />
            </div>
          </div>

          {/* Frequency Analysis */}
          <div className="section-card">
            <h3>
              <FiBarChart2/> Frequency Analysis
            </h3>
            <div className="frequency-analysis">
              {frequencyData.map((freq, index) => (
                <FrequencyBar
                  key={index}
                  band={freq.band}
                  percentage={freq.percentage}
                  color={freq.color}
                />
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
