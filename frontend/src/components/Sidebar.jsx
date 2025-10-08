// src/components/Sidebar.jsx
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  FaTachometerAlt,
  FaUpload,
  FaChartLine,
  FaClipboardList,
  FaUserInjured,
  FaFileAlt,
  FaSignOutAlt,
  FaBrain,
  FaBell,
  FaCog
} from "react-icons/fa";
import "../css/sidebar.css";

export default function Sidebar() {
  const [active, setActive] = useState("Dashboard");
  const navigate = useNavigate();

  const menuItems = [
    { name: "Dashboard", icon: FaTachometerAlt, route: "/dashboard" },
    { name: "Upload EEG", icon: FaUpload, route: "/upload" },
    { name: "Signal Analysis", icon: FaChartLine, route: "/analysis" },
    { name: "Results", icon: FaClipboardList, route: "/results" },
    { name: "Patients", icon: FaUserInjured, route: "/patients" },
    
  ];

  const handleMenuClick = (item) => {
    setActive(item.name);
    navigate(item.route);
  };

  const handleLogout = () => {
    // Add logout logic here
    if (window.confirm("Are you sure you want to logout?")) {
      // Perform logout
      window.location.href = "/login";
    }
  };

  return (
    <div className="sidebar">
      {/* Enhanced Header with Profile */}
      <div className="sidebar-header">
        <div className="brand-section">
          <div className="brand-icon">
            <FaBrain />
          </div>
          <h2 className="brand-name">NeuroDetect</h2>
        </div>
        
        
      </div>

      {/* Navigation Menu */}
      <nav className="navigation">
        <ul className="menu">
          {menuItems.map((item) => (
            <li
              key={item.name}
              className={`menu-item ${active === item.name ? "active" : ""}`}
              onClick={() => handleMenuClick(item)}
            >
              <item.icon className="menu-icon" />
              <span className="menu-text">{item.name}</span>
            </li>
          ))}
        </ul>
      </nav>

      
    </div>
  );
}
