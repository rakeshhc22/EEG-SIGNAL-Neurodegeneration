// src/pages/Reports.jsx
import React, { useState, useEffect } from "react";
import {
  FiDownload,
  FiCalendar,
  FiFileText,
  FiBarChart2,
  FiUsers,
  FiActivity,
  FiAlertTriangle,
  FiTrendingUp,
  FiFilter,
  FiRefreshCw,
  FiEye,
  FiShare2
} from "react-icons/fi";
import "../css/reports.css";

export default function Reports() {
  const [dateRange, setDateRange] = useState("Last 7 days");
  const [reportType, setReportType] = useState("Performance Summary");
  const [format, setFormat] = useState("PDF");
  const [isGenerating, setIsGenerating] = useState(false);
  
  const [analyticsData] = useState({
    weekly: [
      { week: "Week 1", normal: 45, seizure: 8, neurodegeneration: 3, total: 56 },
      { week: "Week 2", normal: 52, seizure: 12, neurodegeneration: 4, total: 68 },
      { week: "Week 3", normal: 48, seizure: 7, neurodegeneration: 2, total: 57 },
      { week: "Week 4", normal: 61, seizure: 11, neurodegeneration: 5, total: 77 }
    ],
    summary: {
      totalPatients: 10,
      totalAnalyses: 258,
      seizuresDetected: 38,
      accuracy: 94.2,
      improvement: 2.1
    }
  });

  const [recentReports] = useState([
    {
      id: 1,
      name: "Weekly Performance Report",
      type: "Performance Summary",
      date: "Sep 1, 2025",
      size: "2.4 MB",
      format: "PDF",
      status: "Completed"
    },
    {
      id: 2,
      name: "Patient Analysis Report",
      type: "Patient Summary",
      date: "Aug 28, 2025",
      size: "1.8 MB",
      format: "Excel",
      status: "Completed"
    },
    {
      id: 3,
      name: "Model Performance Report",
      type: "Technical Report",
      date: "Aug 25, 2025",
      size: "3.2 MB",
      format: "PDF",
      status: "Completed"
    }
  ]);

  const handleGenerateReport = async () => {
    setIsGenerating(true);
    
    // Simulate report generation
    setTimeout(() => {
      setIsGenerating(false);
      alert(`${reportType} report generated successfully in ${format} format!`);
    }, 3000);
  };

  const StatCard = ({ icon: Icon, title, value, subtitle, color }) => (
    <div className="summary-stat-card">
      <div className="stat-icon" style={{ background: color }}>
        <Icon size={24} />
      </div>
      <div className="stat-info">
        <div className="stat-value">{value}</div>
        <div className="stat-title">{title}</div>
        {subtitle && <div className="stat-subtitle">{subtitle}</div>}
      </div>
    </div>
  );

  const StackedBarChart = () => (
    <div className="analytics-chart">
      <div className="chart-header">
        <h3>Analytics Overview</h3>
        <div className="chart-legend">
          <div className="legend-item">
            <div className="legend-color normal"></div>
            <span>Normal</span>
          </div>
          <div className="legend-item">
            <div className="legend-color seizure"></div>
            <span>Seizure</span>
          </div>
          <div className="legend-item">
            <div className="legend-color neurodegeneration"></div>
            <span>Neurodegeneration</span>
          </div>
        </div>
      </div>

      <div className="chart-container">
        <svg width="100%" height="400" viewBox="0 0 800 400">
          {/* Y-axis grid lines and labels */}
          {[0, 20, 40, 60, 80].map(y => (
            <g key={y}>
              <line
                x1="60"
                y1={340 - (y * 3.5)}
                x2="740"
                y2={340 - (y * 3.5)}
                stroke="#e9ecef"
                strokeWidth="1"
              />
              <text
                x="45"
                y={345 - (y * 3.5)}
                fontSize="12"
                fill="#6c757d"
                textAnchor="end"
              >
                {y}
              </text>
            </g>
          ))}

          {/* Bars */}
          {analyticsData.weekly.map((data, index) => {
            const barWidth = 120;
            const barX = 100 + index * 160;
            
            // Calculate heights
            const normalHeight = (data.normal / 80) * 280;
            const seizureHeight = (data.seizure / 80) * 280;
            const neuroHeight = (data.neurodegeneration / 80) * 280;
            
            return (
              <g key={data.week}>
                {/* Normal (bottom) */}
                <rect
                  x={barX}
                  y={340 - normalHeight}
                  width={barWidth}
                  height={normalHeight}
                  fill="#17a2b8"
                  rx="0"
                />
                
                {/* Seizure (middle) */}
                <rect
                  x={barX}
                  y={340 - normalHeight - seizureHeight}
                  width={barWidth}
                  height={seizureHeight}
                  fill="#dc3545"
                  rx="0"
                />
                
                {/* Neurodegeneration (top) */}
                <rect
                  x={barX}
                  y={340 - normalHeight - seizureHeight - neuroHeight}
                  width={barWidth}
                  height={neuroHeight}
                  fill="#fd7e14"
                  rx="0"
                />
                
                {/* Week label */}
                <text
                  x={barX + barWidth/2}
                  y={365}
                  fontSize="14"
                  fill="#495057"
                  textAnchor="middle"
                  fontWeight="600"
                >
                  {data.week}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );

  const ReportItem = ({ report }) => (
    <div className="report-item">
      <div className="report-icon">
        <FiFileText />
      </div>
      <div className="report-details">
        <div className="report-name">{report.name}</div>
        <div className="report-meta">
          <span className="report-type">{report.type}</span>
          <span className="report-date">{report.date}</span>
          <span className="report-size">{report.size}</span>
        </div>
      </div>
      <div className="report-actions">
        <div className={`report-status ${report.status.toLowerCase()}`}>
          {report.status}
        </div>
        <div className="action-buttons">
          <button className="action-btn view-btn" title="View Report">
            <FiEye />
          </button>
          <button className="action-btn download-btn" title="Download">
            <FiDownload />
          </button>
          <button className="action-btn share-btn" title="Share">
            <FiShare2 />
          </button>
        </div>
      </div>
    </div>
  );

  return (
    <div className="reports-page">
      <div className="reports-header">
        <h1>Reports</h1>
        <p>Generate comprehensive analysis reports and track performance metrics</p>
      </div>

      {/* Summary Statistics */}
      <div className="summary-stats">
        <StatCard
          icon={FiUsers}
          title="Total Patients"
          value={analyticsData.summary.totalPatients}
          color="linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
        />
        <StatCard
          icon={FiActivity}
          title="Total Analyses"
          value={analyticsData.summary.totalAnalyses}
          color="linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"
        />
        <StatCard
          icon={FiAlertTriangle}
          title="Seizures Detected"
          value={analyticsData.summary.seizuresDetected}
          color="linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)"
        />
        <StatCard
          icon={FiTrendingUp}
          title="Accuracy Rate"
          value={`${analyticsData.summary.accuracy}%`}
          subtitle={`+${analyticsData.summary.improvement}% improved`}
          color="linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)"
        />
      </div>

      {/* Report Generation Section */}
      <div className="report-generation-section">
        <div className="generation-card">
          <h3>
            <FiBarChart2 /> Generate New Report
          </h3>
          
          <div className="generation-controls">
            <div className="control-group">
              <label>
                <FiCalendar /> Date Range
              </label>
              <select 
                value={dateRange} 
                onChange={(e) => setDateRange(e.target.value)}
                className="control-select"
              >
                <option>Last 7 days</option>
                <option>Last 30 days</option>
                <option>Last 3 months</option>
                <option>Last 6 months</option>
                <option>Last year</option>
                <option>Custom range</option>
              </select>
            </div>

            <div className="control-group">
              <label>
                <FiFileText /> Report Type
              </label>
              <select 
                value={reportType} 
                onChange={(e) => setReportType(e.target.value)}
                className="control-select"
              >
                <option>Performance Summary</option>
                <option>Patient Summary</option>
                <option>Technical Report</option>
                <option>Compliance Report</option>
                <option>Custom Report</option>
              </select>
            </div>

            <div className="control-group">
              <label>
                <FiDownload /> Format
              </label>
              <select 
                value={format} 
                onChange={(e) => setFormat(e.target.value)}
                className="control-select"
              >
                <option>PDF</option>
                <option>Excel</option>
                <option>Word</option>
                <option>CSV</option>
              </select>
            </div>
          </div>

          <button 
            className={`generate-btn ${isGenerating ? 'generating' : ''}`}
            onClick={handleGenerateReport}
            disabled={isGenerating}
          >
            {isGenerating ? (
              <>
                <FiRefreshCw className="spinning" />
                Generating Report...
              </>
            ) : (
              <>
                <FiDownload />
                Generate Report
              </>
            )}
          </button>
        </div>
      </div>

      {/* Analytics Chart */}
      <div className="analytics-section">
        <StackedBarChart />
      </div>

      {/* Recent Reports */}
      <div className="recent-reports-section">
        <div className="section-header">
          <h3>
            <FiFileText /> Recent Reports
          </h3>
          <div className="header-actions">
            <button className="filter-btn">
              <FiFilter /> Filter
            </button>
            <button className="refresh-btn">
              <FiRefreshCw /> Refresh
            </button>
          </div>
        </div>

        <div className="reports-list">
          {recentReports.map(report => (
            <ReportItem key={report.id} report={report} />
          ))}
        </div>
      </div>
    </div>
  );
}
