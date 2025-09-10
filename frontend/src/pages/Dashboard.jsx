// src/pages/Dashboard.jsx
import React, { useState, useEffect } from "react";
import { 
  FiUsers, 
  FiActivity, 
  FiAlertTriangle, 
  FiCheckCircle,
  FiTrendingUp,
  FiClock,
  FiFileText
} from "react-icons/fi";
import "../css/dashboard.css";

export default function Dashboard() {
  const [dashboardData, setDashboardData] = useState({
    totalPatients: 156,
    processedSignals: 1247,
    seizuresDetected: 23,
    accuracyRate: 94.2,
    analysisHistory: [],
    recentActivity: []
  });

  const [chartData] = useState({
    monthlyData: [
      { month: 'Jan', processedSignals: 120, seizuresDetected: 8 },
      { month: 'Feb', processedSignals: 145, seizuresDetected: 12 },
      { month: 'Mar', processedSignals: 170, seizuresDetected: 15 },
      { month: 'Apr', processedSignals: 160, seizuresDetected: 11 },
      { month: 'May', processedSignals: 185, seizuresDetected: 18 },
      { month: 'Jun', processedSignals: 210, seizuresDetected: 14 },
      { month: 'Jul', processedSignals: 195, seizuresDetected: 16 },
      { month: 'Aug', processedSignals: 220, seizuresDetected: 19 }
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
      seizuresDetected: patients.filter(p => p.status?.includes('Seizure')).length || 23
    }));
  }, []);

  const StatCard = ({ icon: Icon, title, value, color, trend }) => (
    <div className="stat-card">
      <div className="stat-icon" style={{ background: color }}>
        <Icon size={24} />
      </div>
      <div className="stat-content">
        <div className="stat-value">{value}</div>
        <div className="stat-title">{title}</div>
        {trend && (
          <div className="stat-trend">
            <FiTrendingUp size={14} />
            <span>{trend}</span>
          </div>
        )}
      </div>
    </div>
  );

  const TrendChart = () => (
    <div className="trend-chart">
      <svg width="100%" height="200" viewBox="0 0 800 200">
        {/* Grid lines */}
        {[0, 50, 100, 150, 200, 250].map(y => (
          <line
            key={y}
            x1="60"
            y1={160 - (y * 0.6)}
            x2="740"
            y2={160 - (y * 0.6)}
            stroke="#e9ecef"
            strokeWidth="1"
          />
        ))}
        
        {/* Y-axis labels */}
        {[0, 50, 100, 150, 200, 250].map(y => (
          <text
            key={y}
            x="50"
            y={165 - (y * 0.6)}
            fontSize="12"
            fill="#6c757d"
            textAnchor="end"
          >
            {y}
          </text>
        ))}

        {/* Processed Signals Line */}
        <polyline
          points={chartData.monthlyData.map((point, index) => 
            `${80 + index * 85},${160 - (point.processedSignals * 0.6)}`
          ).join(' ')}
          fill="none"
          stroke="#17a2b8"
          strokeWidth="3"
        />

        {/* Seizures Detected Line */}
        <polyline
          points={chartData.monthlyData.map((point, index) => 
            `${80 + index * 85},${160 - (point.seizuresDetected * 0.6)}`
          ).join(' ')}
          fill="none"
          stroke="#e74c3c"
          strokeWidth="3"
        />

        {/* Data points for Processed Signals */}
        {chartData.monthlyData.map((point, index) => (
          <circle
            key={`processed-${index}`}
            cx={80 + index * 85}
            cy={160 - (point.processedSignals * 0.6)}
            r="4"
            fill="#17a2b8"
          />
        ))}

        {/* Data points for Seizures */}
        {chartData.monthlyData.map((point, index) => (
          <circle
            key={`seizures-${index}`}
            cx={80 + index * 85}
            cy={160 - (point.seizuresDetected * 0.6)}
            r="4"
            fill="#e74c3c"
          />
        ))}

        {/* X-axis labels */}
        {chartData.monthlyData.map((point, index) => (
          <text
            key={point.month}
            x={80 + index * 85}
            y="185"
            fontSize="12"
            fill="#6c757d"
            textAnchor="middle"
          >
            {point.month}
          </text>
        ))}
      </svg>
      
      <div className="chart-legend">
        <div className="legend-item">
          <div className="legend-color" style={{ background: '#17a2b8' }}></div>
          <span>Processed Signals</span>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{ background: '#e74c3c' }}></div>
          <span>Seizures Detected</span>
        </div>
      </div>
    </div>
  );

  const ModelPerformanceBarChart = () => {
  // Data for the models
  const data = [
    { model: 'TabNet', accuracy: 94.2, color: '#17a2b8' },
    { model: 'QDA', accuracy: 87.6, color: '#fdb462' }
  ];

  const maxAccuracy = 100;
  const barHeightMax = 120;

  return (
    <div className="model-performance-bar">
      <svg width="280" height="180" viewBox="0 0 280 180">
        {/* Grid lines */}
        {[0, 25, 50, 75, 100].map((val) => (
          <line
            key={val}
            x1="40"
            y1={140 - (val / 100) * barHeightMax}
            x2="260"
            y2={140 - (val / 100) * barHeightMax}
            stroke="#e9ecef"
            strokeWidth="1"
          />
        ))}

        {/* Y-axis */}
        <line x1="40" y1="20" x2="40" y2="140" stroke="#ced4da" strokeWidth="2"/>
        
        {/* X-axis */}
        <line x1="40" y1="140" x2="260" y2="140" stroke="#ced4da" strokeWidth="2"/>

        {/* Bars */}
        {data.map((d, i) => {
          const barHeight = (d.accuracy / maxAccuracy) * barHeightMax;
          const barWidth = 50;
          const xPosition = 70 + i * 100;
          
          return (
            <g key={d.model}>
              {/* Bar */}
              <rect
                x={xPosition}
                y={140 - barHeight}
                width={barWidth}
                height={barHeight}
                fill={d.color}
                rx="6"
                ry="6"
              />
              
              {/* Percentage text on top of bar */}
              <text
                x={xPosition + barWidth/2}
                y={140 - barHeight - 8}
                textAnchor="middle"
                fontSize="14"
                fontWeight="700"
                fill="#2c3e50"
              >
                {d.accuracy}%
              </text>
              
              {/* Model name below X-axis */}
              <text
                x={xPosition + barWidth/2}
                y="158"
                textAnchor="middle"
                fontSize="13"
                fontWeight="600"
                fill="#495057"
              >
                {d.model}
              </text>
            </g>
          );
        })}

        {/* Y-axis labels */}
        {[0, 25, 50, 75, 100].map((val) => (
          <text
            key={val}
            x="35"
            y={145 - (val / 100) * barHeightMax}
            fontSize="11"
            fill="#6c757d"
            textAnchor="end"
          >
            {val}%
          </text>
        ))}
      </svg>
      
      {/* Legend */}
      <div className="bar-chart-legend">
        {data.map((d) => (
          <div key={d.model} className="legend-item">
            <div 
              className="legend-color" 
              style={{ background: d.color }}
            ></div>
            <span>{d.model} Accuracy</span>
          </div>
        ))}
      </div>
    </div>
  );
};


  const ActivityItem = ({ activity }) => {
    const getActivityIcon = () => {
      switch (activity.type) {
        case 'analysis':
          return <FiActivity className="activity-icon analysis" />;
        case 'system':
          return <FiCheckCircle className="activity-icon system" />;
        case 'patient':
          return <FiUsers className="activity-icon patient" />;
        case 'alert':
          return <FiAlertTriangle className="activity-icon alert" />;
        case 'report':
          return <FiFileText className="activity-icon report" />;
        default:
          return <FiClock className="activity-icon default" />;
      }
    };

    return (
      <div className="activity-item">
        {getActivityIcon()}
        <div className="activity-content">
          <div className="activity-text">
            <strong>{activity.user}</strong> {activity.action}
            {activity.patient && <span className="patient-name"> for {activity.patient}</span>}
            {activity.details && <span className="activity-details"> - {activity.details}</span>}
          </div>
          <div className="activity-time">{activity.timestamp}</div>
        </div>
        <div className={`activity-status ${activity.status}`}>
          {activity.status === 'critical' && <FiAlertTriangle size={16} />}
          {activity.status === 'completed' && <FiCheckCircle size={16} />}
          {activity.status === 'success' && <FiCheckCircle size={16} />}
        </div>
      </div>
    );
  };

  return (
    <div className="dashboard-page">
      <div className="dashboard-header">
        <h1>Dashboard</h1>
        <p>Monitor EEG analysis performance and system activity</p>
      </div>

      {/* Statistics Cards */}
      <div className="stats-grid">
        <StatCard
          icon={FiUsers}
          title="Total Patients"
          value={dashboardData.totalPatients}
          color="linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
          trend="+12% this month"
        />
        <StatCard
          icon={FiActivity}
          title="Processed Signals"
          value={dashboardData.processedSignals.toLocaleString()}
          color="linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"
          trend="+8% this week"
        />
        <StatCard
          icon={FiAlertTriangle}
          title="Seizures Detected"
          value={dashboardData.seizuresDetected}
          color="linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)"
          trend="+3 this week"
        />
        <StatCard
          icon={FiCheckCircle}
          title="Accuracy Rate"
          value={`${dashboardData.accuracyRate}%`}
          color="linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)"
          trend="+2.1% improved"
        />
      </div>

      {/* Charts Section */}
      <div className="charts-section">
        <div className="chart-container trend-container">
          <h3>Analysis Trends</h3>
          <TrendChart />
        </div>
        
        <div className="chart-container model-performance">
          <h3>Model Performance</h3>
          <ModelPerformanceBarChart />
        </div>
      </div>
      
    </div>
  );
}
