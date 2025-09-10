// src/pages/Patients.jsx
import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "../css/patients.css";
import { FiSearch, FiEye, FiPlus, FiFilter, FiDownload } from "react-icons/fi";

export default function Patients() {
  const navigate = useNavigate();
  const [patients, setPatients] = useState([]);
  const [filteredPatients, setFilteredPatients] = useState([]);
  const [searchTerm, setSearchTerm] = useState("");
  const [filterStatus, setFilterStatus] = useState("all");

  useEffect(() => {
    // Load patients from localStorage
    const storedPatients = JSON.parse(localStorage.getItem('patients') || '[]');
    
    // Add some demo patients if none exist
    if (storedPatients.length === 0) {
      const demoPatients = [
        {
          id: "MED-2024-001",
          name: "John Smith",
          age: "45",
          lastTest: "Aug 15, 2024",
          status: "Seizure Detected",
          riskLevel: "High Risk",
          medicalId: "MED-2024-001",
          timestamp: "2024-08-15T10:30:00.000Z"
        },
        {
          id: "MED-2024-002",
          name: "Sarah Johnson",
          age: "34",
          lastTest: "Aug 20, 2024",
          status: "Normal",
          riskLevel: "Low Risk",
          medicalId: "MED-2024-002",
          timestamp: "2024-08-20T14:15:00.000Z"
        },
        {
          id: "MED-2024-003",
          name: "Robert Wilson",
          age: "67",
          lastTest: "Aug 18, 2024",
          status: "Neurodegeneration Risk",
          riskLevel: "Medium Risk",
          medicalId: "MED-2024-003",
          timestamp: "2024-08-18T09:45:00.000Z"
        }
      ];
      setPatients(demoPatients);
      setFilteredPatients(demoPatients);
    } else {
      setPatients(storedPatients);
      setFilteredPatients(storedPatients);
    }
  }, []);

  useEffect(() => {
    // Filter patients based on search term and status filter
    let filtered = patients;

    if (searchTerm) {
      filtered = filtered.filter(patient =>
        patient.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        patient.id.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    if (filterStatus !== "all") {
      filtered = filtered.filter(patient =>
        patient.status.toLowerCase().includes(filterStatus.toLowerCase())
      );
    }

    setFilteredPatients(filtered);
  }, [patients, searchTerm, filterStatus]);

  const handleViewPatient = (patientId) => {
    // In a real app, this would navigate to a detailed patient view
    console.log("View patient:", patientId);
    alert(`Viewing details for patient ${patientId}`);
  };

  const getRiskLevelColor = (riskLevel) => {
    switch (riskLevel.toLowerCase()) {
      case 'high risk':
        return '#e74c3c';
      case 'medium risk':
        return '#f39c12';
      case 'low risk':
        return '#27ae60';
      default:
        return '#6c757d';
    }
  };

  const getStatusColor = (status) => {
    if (status.includes('Seizure')) return '#e74c3c';
    if (status.includes('Neurodegeneration')) return '#f39c12';
    return '#27ae60';
  };

  const exportPatientData = () => {
    const dataStr = JSON.stringify(filteredPatients, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `Patients_Export_${new Date().toLocaleDateString()}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  return (
    <div className="patients-page">
      <div className="patients-header">
        <div className="header-title">
          <h1>Patient Management</h1>
          <p>Monitor and manage patient EEG analysis records</p>
        </div>
        
        <div className="header-actions">
          <button 
            className="btn-primary"
            onClick={() => navigate('/upload')}
          >
            <FiPlus /> Add New Patient
          </button>
          <button 
            className="btn-secondary"
            onClick={exportPatientData}
          >
            <FiDownload /> Export Data
          </button>
        </div>
      </div>

      {/* Search and Filter Controls */}
      <div className="controls-section">
        <div className="search-box">
          <FiSearch className="search-icon" />
          <input
            type="text"
            placeholder="Search patients..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="search-input"
          />
        </div>

        <div className="filter-controls">
          <FiFilter className="filter-icon" />
          <select
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
            className="filter-select"
          >
            <option value="all">All Status</option>
            <option value="seizure">Seizure Detected</option>
            <option value="normal">Normal</option>
            <option value="neurodegeneration">Neurodegeneration Risk</option>
          </select>
        </div>
      </div>

      {/* Patients Table */}
      <div className="patients-table-container">
        <table className="patients-table">
          <thead>
            <tr>
              <th>Patient ID</th>
              <th>Name</th>
              <th>Age</th>
              <th>Last Test</th>
              <th>Status</th>
              <th>Risk Level</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {filteredPatients.length === 0 ? (
              <tr>
                <td colSpan="7" className="no-data">
                  {searchTerm || filterStatus !== "all" 
                    ? "No patients match your search criteria" 
                    : "No patients found. Upload an EEG file to add a patient."
                  }
                </td>
              </tr>
            ) : (
              filteredPatients.map((patient) => (
                <tr key={patient.id}>
                  <td className="patient-id">{patient.id}</td>
                  <td className="patient-name">{patient.name}</td>
                  <td>{patient.age || 'N/A'}</td>
                  <td>{patient.lastTest}</td>
                  <td>
                    <span 
                      className="status-badge"
                      style={{ backgroundColor: getStatusColor(patient.status) }}
                    >
                      {patient.status}
                    </span>
                  </td>
                  <td>
                    <span 
                      className="risk-badge"
                      style={{ color: getRiskLevelColor(patient.riskLevel) }}
                    >
                      {patient.riskLevel}
                    </span>
                  </td>
                  <td>
                    <button
                      className="btn-view"
                      onClick={() => handleViewPatient(patient.id)}
                      title="View Patient Details"
                    >
                      <FiEye /> View
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Statistics Summary */}
      
    </div>
  );
}
