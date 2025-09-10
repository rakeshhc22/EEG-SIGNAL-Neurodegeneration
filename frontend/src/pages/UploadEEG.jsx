// src/pages/UploadEEG.jsx
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "../css/uploadEEG.css";
import Loader from "../components/Loader";
import { FiUploadCloud, FiUser, FiFileText, FiCalendar } from "react-icons/fi";
import { analyzeEEG } from "../services/api";

export default function UploadEEG() {
  const navigate = useNavigate();
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  // Patient form data
  const [patientData, setPatientData] = useState({
    name: "",
    age: "",
    medicalId: "",
    notes: ""
  });

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setMessage("");
    setError("");

    if (selectedFile) {
      const allowedTypes = ['.edf', '.csv', '.txt', '.mat'];
      const fileExtension = selectedFile.name.toLowerCase().substring(selectedFile.name.lastIndexOf('.'));
      
      if (!allowedTypes.includes(fileExtension)) {
        setError("Please upload a valid EEG file (.edf, .csv, .txt, .mat)");
        setFile(null);
        return;
      }

      if (selectedFile.size > 100 * 1024 * 1024) {
        setError("File size must be less than 100MB");
        setFile(null);
        return;
      }
    }
  };

  const handlePatientDataChange = (e) => {
    const { name, value } = e.target;
    setPatientData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      handleFileChange({ target: { files: [droppedFile] } });
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!file) {
      setError("Please select an EEG file");
      return;
    }

    if (!patientData.name.trim()) {
      setError("Please enter patient name");
      return;
    }

    setUploading(true);
    setMessage("");
    setError("");

    try {
      console.log("Starting EEG analysis for:", patientData);
      
      const response = await analyzeEEG(file);
      
      // Store analysis results and patient data in localStorage for Results page
      const analysisData = {
        ...response,
        patientInfo: patientData,
        timestamp: new Date().toISOString(),
        fileName: file.name
      };
      
      localStorage.setItem('latestAnalysis', JSON.stringify(analysisData));
      
      // Also store patient data for Patients page
      const existingPatients = JSON.parse(localStorage.getItem('patients') || '[]');
      const newPatient = {
        id: `MED-2024-${String(existingPatients.length + 1).padStart(3, '0')}`,
        ...patientData,
        lastTest: new Date().toLocaleDateString(),
        status: response.results.QDA.predicted_class,
        riskLevel: response.results.QDA.predicted_class === 'Seizure Detected' ? 'High Risk' : 'Low Risk',
        timestamp: new Date().toISOString()
      };
      
      existingPatients.push(newPatient);
      localStorage.setItem('patients', JSON.stringify(existingPatients));
      
      setMessage("✅ Analysis completed successfully! Redirecting to results...");
      
      // Navigate to results page after 2 seconds
      setTimeout(() => {
        navigate('/results');
      }, 2000);
      
    } catch (err) {
      console.error("Error analyzing EEG:", err);
      setError(`❌ Analysis failed: ${err.message}`);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="upload-page">
      <div className="upload-container">
        <div className="upload-header">
          <h1>Upload EEG Signal</h1>
          <p>Upload EEG files for seizure detection analysis</p>
        </div>

        <form onSubmit={handleSubmit} className="upload-form">
          {/* File Upload Section */}
          <div 
            className={`upload-box ${file ? 'has-file' : ''}`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
          >
            <div className="upload-content">
              <FiUploadCloud className="upload-icon" size={48} />
              <h3>Drag and drop your EEG files here</h3>
              <p>or</p>
              <button 
                type="button" 
                className="browse-btn"
                onClick={() => document.getElementById('fileInput').click()}
                disabled={uploading}
              >
                Browse Files
              </button>
              
              <input
                id="fileInput"
                type="file"
                accept=".edf,.csv,.txt,.mat"
                onChange={handleFileChange}
                className="file-input"
                style={{ display: 'none' }}
                disabled={uploading}
              />
              
              <div className="file-info">
                <p>Supported formats: <strong>.edf, .csv, .txt, .mat</strong> (Max 100MB)</p>
              </div>
            </div>
          </div>

          {/* Selected File Display */}
          {file && (
            <div className="selected-file">
              <FiFileText className="file-icon" />
              <span>{file.name}</span>
              <span className="file-size">({(file.size / (1024*1024)).toFixed(2)} MB)</span>
            </div>
          )}

          {/* Patient Information Form */}
          <div className="patient-info-section">
            <h3><FiUser /> Patient Information</h3>
            
            <div className="form-row">
              <div className="form-group">
                <label htmlFor="patientName">Patient Name *</label>
                <input
                  id="patientName"
                  type="text"
                  name="name"
                  placeholder="Enter patient name"
                  value={patientData.name}
                  onChange={handlePatientDataChange}
                  required
                  disabled={uploading}
                />
              </div>
              
              <div className="form-group">
                <label htmlFor="patientAge">Age</label>
                <input
                  id="patientAge"
                  type="number"
                  name="age"
                  placeholder="Age"
                  value={patientData.age}
                  onChange={handlePatientDataChange}
                  min="1"
                  max="120"
                  disabled={uploading}
                />
              </div>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label htmlFor="medicalId">Medical ID</label>
                <input
                  id="medicalId"
                  type="text"
                  name="medicalId"
                  placeholder="MED-2024-XXX"
                  value={patientData.medicalId}
                  onChange={handlePatientDataChange}
                  disabled={uploading}
                />
              </div>
            </div>

            <div className="form-group">
              <label htmlFor="notes">Additional Notes</label>
              <textarea
                id="notes"
                name="notes"
                placeholder="Additional notes..."
                value={patientData.notes}
                onChange={handlePatientDataChange}
                rows="3"
                disabled={uploading}
              />
            </div>
          </div>

          {/* Submit Button */}
          <button 
            type="submit"
            className={`analyze-btn ${uploading ? 'uploading' : ''}`}
            disabled={uploading || !file || !patientData.name.trim()}
          >
            {uploading ? (
              <>
                <Loader />
                Analyzing EEG Signal...
              </>
            ) : (
              'Analyze EEG Signal'
            )}
          </button>
        </form>

        {/* Messages */}
        {message && (
          <div className="message success">
            {message}
          </div>
        )}

        {error && (
          <div className="message error">
            {error}
          </div>
        )}
      </div>
    </div>
  );
}
