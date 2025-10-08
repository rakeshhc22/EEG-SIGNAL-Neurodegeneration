// src/pages/Dashboard.jsx
import React, { useState, useEffect } from "react";
import { FiUsers, FiActivity, FiAlertTriangle, FiCheckCircle, FiTrendingUp, FiFileText, FiCpu, FiBarChart2, FiPieChart } from "react-icons/fi";
import "../css/dashboard.css";

export default function Dashboard() {
  const [dashboardData, setDashboardData] = useState({
    totalPatients: 156,
    processedSignals: 1247,
    seizuresDetected: 23,
    neurodegenerationDetected: 15, // ADDED: Neurodegeneration count
    normalCases: 1209, // ADDED: Normal cases count
    accuracyRate: 94.2,
    analysisHistory: [],
    recentActivity: []
  });

  const [chartData] = useState({
    monthlyData: [
      { month: 'Jan', processedSignals: 120, seizuresDetected: 8, neurodegenerationDetected: 5, normalCases: 107 },
      { month: 'Feb', processedSignals: 145, seizuresDetected: 12, neurodegenerationDetected: 7, normalCases: 126 },
      { month: 'Mar', processedSignals: 170, seizuresDetected: 15, neurodegenerationDetected: 9, normalCases: 146 },
      { month: 'Apr', processedSignals: 160, seizuresDetected: 11, neurodegenerationDetected: 6, normalCases: 143 },
      { month: 'May', processedSignals: 185, seizuresDetected: 18, neurodegenerationDetected: 8, normalCases: 159 },
      { month: 'Jun', processedSignals: 210, seizuresDetected: 14, neurodegenerationDetected: 12, normalCases: 184 },
      { month: 'Jul', processedSignals: 195, seizuresDetected: 16, neurodegenerationDetected: 10, normalCases: 169 },
      { month: 'Aug', processedSignals: 220, seizuresDetected: 19, neurodegenerationDetected: 13, normalCases: 188 }
    ],
    modelAccuracy: {
      tabnet: 68.5,
      qda: 31.5
    }
  });

  useEffect(() => {
    // Load recent activity from localStorage
    const patients = JSON.parse(localStorage.getItem('patients') || '[]');
    setDashboardData(prev => ({
      ...prev,
      totalPatients: patients.length > 0 ? patients.length : 156,
      seizuresDetected: patients.filter(p => p.status?.includes('Seizure')).length || 23,
      neurodegenerationDetected: patients.filter(p => p.status?.includes('Neurodegeneration')).length || 15, // ADDED
      normalCases: patients.filter(p => p.status === 'Normal').length || 1209 // ADDED
    }));
  }, []);

  const StatCard = ({ icon: Icon, title, value, color, trend, gradient }) => (
    <div className="stat-card" style={{ '--card-gradient': gradient }}>
      <div className="stat-icon" style={{ background: gradient }}>
        <Icon size={28} />
      </div>
      <div className="stat-content">
        <div className="stat-value">{value}</div>
        <div className="stat-title">{title}</div>
        {trend && (
          <div className={`stat-trend ${trend > 0 ? 'positive' : 'negative'}`}>
            <FiTrendingUp size={14} />
            {Math.abs(trend)}%
          </div>
        )}
      </div>
    </div>
  );

  
  

  return (
    <div className="dashboard-page fade-in">
      {/* Header Section */}
      <div className="dashboard-header">
        <h1>NeuroDetect Dashboard</h1>
        <p>Monitor EEG analysis performance and system activity</p>
      </div>

      {/* Statistics Grid */}
      <div className="stats-grid">
        <StatCard
          icon={FiUsers}
          title="Total Patients"
          value={dashboardData.totalPatients.toLocaleString()}
          gradient="linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
          trend={8.2}
        />
        <StatCard
          icon={FiActivity}
          title="Processed Signals"
          value={dashboardData.processedSignals.toLocaleString()}
          gradient="linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"
          trend={12.5}
        />
        <StatCard
          icon={FiCheckCircle}
          title="Normal Cases"
          value={dashboardData.normalCases.toLocaleString()}
          gradient="linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)"
          trend={5.3}
        />
        <StatCard
          icon={FiAlertTriangle}
          title="Seizures Detected"
          value={dashboardData.seizuresDetected}
          gradient="linear-gradient(135deg, #fa709a 0%, #fee140 100%)"
          trend={-2.1}
        />
        <StatCard
          icon={FiCpu}
          title="Neurodegeneration"
          value={dashboardData.neurodegenerationDetected}
          gradient="linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)"
          trend={3.7}
        />
        <StatCard
          icon={FiTrendingUp}
          title="Model Accuracy"
          value={`${dashboardData.accuracyRate}%`}
          gradient="linear-gradient(135deg, #d299c2 0%, #fef9d7 100%)"
          trend={1.8}
        />
      </div>

      {/* Charts Section */}
      <div className="charts-section">
        {/* Monthly Trends Chart */}
        <div className="chart-container">
          <h3>
            <FiBarChart2 />
            Monthly Analysis Trends
          </h3>
          <div className="trend-chart">
            <div className="monthly-bars">
              {chartData.monthlyData.map((data, index) => (
                <div key={index} className="month-column">
                  <div className="bar-stack">
                    <div 
                      className="bar normal-bar" 
                      style={{ height: `${(data.normalCases / data.processedSignals) * 100}%` }}
                      title={`Normal: ${data.normalCases}`}
                    ></div>
                    <div 
                      className="bar seizure-bar" 
                      style={{ height: `${(data.seizuresDetected / data.processedSignals) * 100}%` }}
                      title={`Seizures: ${data.seizuresDetected}`}
                    ></div>
                    <div 
                      className="bar neurodegeneration-bar" 
                      style={{ height: `${(data.neurodegenerationDetected / data.processedSignals) * 100}%` }}
                      title={`Neurodegeneration: ${data.neurodegenerationDetected}`}
                    ></div>
                  </div>
                  <div className="month-label">{data.month}</div>
                </div>
              ))}
            </div>
          </div>
          <div className="chart-legend">
            <div className="legend-item">
              <div className="legend-color normal-color"></div>
              <span>Normal Cases</span>
            </div>
            <div className="legend-item">
              <div className="legend-color seizure-color"></div>
              <span>Seizure Detected</span>
            </div>
            <div className="legend-item">
              <div className="legend-color neurodegeneration-color"></div>
              <span>Neurodegeneration</span>
            </div>
          </div>
        </div>

        {/* Model Performance */}
        <div className="chart-container model-performance">
          <h3>
            <FiPieChart />
            Model Performance
          </h3>
          <div className="model-performance-bar">
            <div className="performance-item">
              <div className="performance-header">
                <span className="model-name">TabNet Classifier</span>
                <span className="model-score">{chartData.modelAccuracy.tabnet}%</span>
              </div>
              <div className="progress-track">
                <div 
                  className="progress-fill tabnet-progress" 
                  style={{ width: `${chartData.modelAccuracy.tabnet}%` }}
                ></div>
              </div>
            </div>
            <div className="performance-item">
              <div className="performance-header">
                <span className="model-name">QDA Classifier</span>
                <span className="model-score">{chartData.modelAccuracy.qda}%</span>
              </div>
              <div className="progress-track">
                <div 
                  className="progress-fill qda-progress" 
                  style={{ width: `${chartData.modelAccuracy.qda}%` }}
                ></div>
              </div>
            </div>
          </div>

          {/* Classification Distribution */}
          <div className="distribution-chart">
            <h4>Classification Distribution</h4>
            <div className="distribution-bars">
              <div className="dist-bar">
                <div className="dist-label">Normal</div>
                <div className="dist-track">
                  <div className="dist-fill normal-dist" style={{ width: '85.2%' }}></div>
                </div>
                <span className="dist-percentage">85.2%</span>
              </div>
              <div className="dist-bar">
                <div className="dist-label">Seizure</div>
                <div className="dist-track">
                  <div className="dist-fill seizure-dist" style={{ width: '8.9%' }}></div>
                </div>
                <span className="dist-percentage">8.9%</span>
              </div>
              <div className="dist-bar">
                <div className="dist-label">Neurodegeneration</div>
                <div className="dist-track">
                  <div className="dist-fill neuro-dist" style={{ width: '5.9%' }}></div>
                </div>
                <span className="dist-percentage">5.9%</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Activity Section */}
      

      {/* Quick Stats Summary */}
      <div className="quick-stats-section">
        <div className="chart-container">
          <h3>
            <FiActivity />
            Today's Summary
          </h3>
          <div className="today-stats">
            <div className="today-stat-item">
              <div className="today-stat-icon normal-bg">
                <FiCheckCircle />
              </div>
              <div className="today-stat-content">
                <div className="today-stat-number">47</div>
                <div className="today-stat-label">Normal Cases</div>
              </div>
            </div>
            <div className="today-stat-item">
              <div className="today-stat-icon seizure-bg">
                <FiAlertTriangle />
              </div>
              <div className="today-stat-content">
                <div className="today-stat-number">3</div>
                <div className="today-stat-label">Seizure Cases</div>
              </div>
            </div>
            <div className="today-stat-item">
              <div className="today-stat-icon neuro-bg">
                <FiCpu />
              </div>
              <div className="today-stat-content">
                <div className="today-stat-number">2</div>
                <div className="today-stat-label">Neurodegeneration</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
