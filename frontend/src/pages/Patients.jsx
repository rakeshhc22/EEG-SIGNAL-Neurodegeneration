// src/pages/Patients.jsx
import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "../css/patients.css";
import { FiSearch, FiPlus, FiFilter, FiDownload, FiTrash2, FiUsers, FiAlertTriangle, FiCheckCircle, FiCpu, FiBarChart } from "react-icons/fi";

export default function Patients() {
  const navigate = useNavigate();
  const [patients, setPatients] = useState([]);
  const [filteredPatients, setFilteredPatients] = useState([]);
  const [searchTerm, setSearchTerm] = useState("");
  const [filterStatus, setFilterStatus] = useState("all");

  useEffect(() => {
    // Load patients from localStorage
    const storedPatients = JSON.parse(localStorage.getItem('patients') || '[]');
    
    // Add enhanced demo patients with neurodegeneration support
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
          timestamp: "2024-08-15T10:30:00.000Z",
          confidence: "89.7%"
        },
        {
          id: "MED-2024-002",
          name: "Sarah Johnson",
          age: "34",
          lastTest: "Aug 20, 2024",
          status: "Normal",
          riskLevel: "Low Risk",
          medicalId: "MED-2024-002",
          timestamp: "2024-08-20T14:15:00.000Z",
          confidence: "97.2%"
        },
        {
          id: "MED-2024-003",
          name: "Robert Wilson",
          age: "67",
          lastTest: "Aug 18, 2024",
          status: "Neurodegeneration Detected", // UPDATED: Full neurodegeneration status
          riskLevel: "Medium Risk",
          medicalId: "MED-2024-003",
          timestamp: "2024-08-18T09:45:00.000Z",
          confidence: "78.4%"
        },
        {
          id: "MED-2024-004",
          name: "Maria Garcia",
          age: "72",
          lastTest: "Aug 22, 2024",
          status: "Neurodegeneration Detected",
          riskLevel: "High Risk",
          medicalId: "MED-2024-004",
          timestamp: "2024-08-22T16:20:00.000Z",
          confidence: "85.1%"
        },
        {
          id: "MED-2024-005",
          name: "David Chen",
          age: "28",
          lastTest: "Aug 23, 2024",
          status: "Normal",
          riskLevel: "Low Risk",
          medicalId: "MED-2024-005",
          timestamp: "2024-08-23T11:45:00.000Z",
          confidence: "95.8%"
        }
      ];
      setPatients(demoPatients);
      setFilteredPatients(demoPatients);
      localStorage.setItem('patients', JSON.stringify(demoPatients));
    } else {
      setPatients(storedPatients);
      setFilteredPatients(storedPatients);
    }
  }, []);

  useEffect(() => {
    // Enhanced filter logic for 3-class support
    let filtered = patients;
    
    if (searchTerm) {
      filtered = filtered.filter(patient => 
        patient.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        patient.id.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }
    
    if (filterStatus !== "all") {
      filtered = filtered.filter(patient => {
        const status = patient.status.toLowerCase();
        switch (filterStatus) {
          case "normal": return status === "normal";
          case "seizure": return status.includes("seizure");
          case "neurodegeneration": return status.includes("neurodegeneration");
          default: return true;
        }
      });
    }
    
    setFilteredPatients(filtered);
  }, [patients, searchTerm, filterStatus]);

  // ADDED: Delete patient function
  const handleDeletePatient = (patientId) => {
    if (window.confirm(`Are you sure you want to delete patient ${patientId}?\n\nThis action cannot be undone and will permanently remove all patient data.`)) {
      const updatedPatients = patients.filter(patient => patient.id !== patientId);
      setPatients(updatedPatients);
      localStorage.setItem('patients', JSON.stringify(updatedPatients));
      
      // Show success message
      alert(`Patient ${patientId} has been successfully deleted.`);
    }
  };

  const getRiskLevelColor = (riskLevel) => {
    switch (riskLevel.toLowerCase()) {
      case 'high risk': return '#e74c3c';
      case 'medium risk': return '#f39c12';
      case 'low risk': return '#27ae60';
      default: return '#6c757d';
    }
  };

  const getStatusColor = (status) => {
    if (status.includes('Seizure')) return '#e74c3c';
    if (status.includes('Neurodegeneration')) return '#f39c12'; // ADDED: Neurodegeneration color
    return '#27ae60';
  };

  const getStatusGradient = (status) => {
    if (status.includes('Seizure')) return 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)';
    if (status.includes('Neurodegeneration')) return 'linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)';
    return 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)';
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

  // Calculate statistics for all 3 classes
  const totalPatients = patients.length;
  const normalCount = patients.filter(p => p.status === 'Normal').length;
  const seizureCount = patients.filter(p => p.status.includes('Seizure')).length;
  const neurodegenerationCount = patients.filter(p => p.status.includes('Neurodegeneration')).length;

  return (
    <div className="patients-page fade-in">
      {/* Enhanced Header Section */}
      <div className="patients-header">
        <div className="header-title">
          <h1>Patient Management</h1>
          <p>Monitor and manage patient EEG analysis records</p>
        </div>
        <div className="header-actions">
          <button className="btn-primary" onClick={() => navigate('/signal-analysis')}>
            <FiPlus />
            Add Patient
          </button>
          <button className="btn-secondary" onClick={exportPatientData}>
            <FiDownload />
            Export Data
          </button>
        </div>
      </div>

      {/* Enhanced Statistics Cards */}
      <div className="stats-summary">
        <div className="stat-card total-card">
          <div className="stat-icon">
            <FiUsers />
          </div>
          <div className="stat-content">
            <h3>{totalPatients}</h3>
            <p>Total Patients</p>
          </div>
        </div>
        
        <div className="stat-card normal-card">
          <div className="stat-icon normal-bg">
            <FiCheckCircle />
          </div>
          <div className="stat-content">
            <h3>{normalCount}</h3>
            <p>Normal Cases</p>
          </div>
        </div>
        
        <div className="stat-card seizure-card">
          <div className="stat-icon seizure-bg">
            <FiAlertTriangle />
          </div>
          <div className="stat-content">
            <h3>{seizureCount}</h3>
            <p>Seizure Cases</p>
          </div>
        </div>
        
        <div className="stat-card neuro-card">
          <div className="stat-icon neuro-bg">
            <FiCpu />
          </div>
          <div className="stat-content">
            <h3>{neurodegenerationCount}</h3>
            <p>Neurodegeneration</p>
          </div>
        </div>
      </div>

      {/* Enhanced Controls Section */}
      <div className="controls-section">
        <div className="search-box">
          <FiSearch className="search-icon" />
          <input
            type="text"
            className="search-input"
            placeholder="Search by name or patient ID..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        
        <div className="filter-controls">
          <FiFilter className="filter-icon" />
          <select 
            className="filter-select"
            value={filterStatus} 
            onChange={(e) => setFilterStatus(e.target.value)}
          >
            <option value="all">All Patients</option>
            <option value="normal">Normal</option>
            <option value="seizure">Seizure Detected</option>
            <option value="neurodegeneration">Neurodegeneration</option>
          </select>
        </div>

        <div className="results-count">
          <FiBarChart />
          <span>Showing {filteredPatients.length} of {totalPatients} patients</span>
        </div>
      </div>

      {/* Enhanced Patient Table */}
      <div className="patients-table-container">
        <table className="patients-table">
          <thead>
            <tr>
              <th>Patient ID</th>
              <th>Name</th>
              <th>Age</th>
              <th>Last Test</th>
              <th>Status</th>
              {/* <th>Confidence</th> */}
              <th>Risk Level</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {filteredPatients.length === 0 ? (
              <tr>
                <td colSpan="8" className="no-data">
                  {searchTerm || filterStatus !== "all" 
                    ? "No patients match your search criteria" 
                    : "No patients found. Upload an EEG file to add a patient."
                  }
                </td>
              </tr>
            ) : (
              filteredPatients.map((patient) => (
                <tr key={patient.id} className="patient-row">
                  <td className="patient-id">{patient.id}</td>
                  <td className="patient-name">{patient.name}</td>
                  <td className="patient-age">{patient.age || 'N/A'}</td>
                  <td className="last-test">{patient.lastTest}</td>
                  <td className="patient-status">
                    <span 
                      className="status-badge"
                      style={{ background: getStatusGradient(patient.status) }}
                    >
                      {patient.status}
                    </span>
                  </td>
                  {/* <td className="confidence-score">
                    <span className="confidence-value">{patient.confidence || 'N/A'}</span>
                  </td> */}
                  <td className="risk-level">
                    <span 
                      className="risk-badge"
                      style={{ color: getRiskLevelColor(patient.riskLevel) }}
                    >
                      {patient.riskLevel}
                    </span>
                  </td>
                  <td className="actions">
                    <button 
                      className="btn-delete"
                      onClick={() => handleDeletePatient(patient.id)}
                      title="Delete Patient"
                    >
                      <FiTrash2 />
                      Delete
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Patient Distribution Chart */}
      <div className="distribution-section">
        <div className="chart-container">
          <h3>
            <FiBarChart />
            Patient Distribution by Classification
          </h3>
          <div className="distribution-chart">
            <div className="dist-bar">
              <div className="dist-label">
                <FiCheckCircle style={{ color: '#4facfe' }} />
                Normal Cases
              </div>
              <div className="dist-track">
                <div 
                  className="dist-fill normal-fill" 
                  style={{ width: `${totalPatients > 0 ? (normalCount / totalPatients) * 100 : 0}%` }}
                ></div>
              </div>
              <span className="dist-value">{normalCount} ({totalPatients > 0 ? ((normalCount / totalPatients) * 100).toFixed(1) : 0}%)</span>
            </div>
            
            <div className="dist-bar">
              <div className="dist-label">
                <FiAlertTriangle style={{ color: '#fa709a' }} />
                Seizure Cases
              </div>
              <div className="dist-track">
                <div 
                  className="dist-fill seizure-fill" 
                  style={{ width: `${totalPatients > 0 ? (seizureCount / totalPatients) * 100 : 0}%` }}
                ></div>
              </div>
              <span className="dist-value">{seizureCount} ({totalPatients > 0 ? ((seizureCount / totalPatients) * 100).toFixed(1) : 0}%)</span>
            </div>
            
            <div className="dist-bar">
              <div className="dist-label">
                <FiCpu style={{ color: '#a8edea' }} />
                Neurodegeneration
              </div>
              <div className="dist-track">
                <div 
                  className="dist-fill neuro-fill" 
                  style={{ width: `${totalPatients > 0 ? (neurodegenerationCount / totalPatients) * 100 : 0}%` }}
                ></div>
              </div>
              <span className="dist-value">{neurodegenerationCount} ({totalPatients > 0 ? ((neurodegenerationCount / totalPatients) * 100).toFixed(1) : 0}%)</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
