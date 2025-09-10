// src/components/Navbar.jsx
import React, { useState } from "react";
import { 
  FaBell, 
  FaChevronDown, 
  FaCog, 
  FaUser, 
  FaSignOutAlt,
  FaBrain,
  FaSearch
} from "react-icons/fa";
import "../css/navbar.css";

export default function Navbar() {
  const [showProfileMenu, setShowProfileMenu] = useState(false);
  const [notificationCount, setNotificationCount] = useState(3);

  const handleProfileClick = () => {
    setShowProfileMenu(!showProfileMenu);
  };
  const handleLogout = () => {
    // Add logout logic here
    if (window.confirm("Are you sure you want to logout?")) {
      // Perform logout
      window.location.href = "/login";
    }
  };

  const handleNotificationClick = () => {
    // Handle notification click
    setNotificationCount(0);
  };

  return (
    <nav className="navbar">
      {/* Left Section - Brand */}
      <div className="navbar-left">
        <div className="brand-section">
          <div className="brand-icon">
            <FaBrain />
          </div>
          <div className="brand-info">
            <h1 className="brand-title">NeuroDetect</h1>
            <p className="brand-subtitle">Advanced EEG Analysis Platform</p>
          </div>
        </div>
      </div>

      {/* Center Section - Search */}
      <div className="navbar-center">
        <div className="search-container">
          <FaSearch className="search-icon" />
          <input 
            type="text" 
            placeholder="Search patients, reports, or analyses..."
            className="search-input"
          />
        </div>
      </div>

      {/* Right Section - Actions & Profile */}
      <div className="navbar-right">
        {/* System Status */}
        <div className="system-status">
          <div className="status-indicator online">
            <div className="status-dot"></div>
            <span className="status-text">Online</span>
          </div>
        </div>

        {/* Notifications */}
        <div className="notifications" onClick={handleNotificationClick}>
          <FaBell className="notif-icon" />
          {notificationCount > 0 && (
            <span className="notif-badge">{notificationCount}</span>
          )}
        </div>

        {/* Profile Section */}
        <div className="profile-section">
          <div className="profile" onClick={handleProfileClick}>
            <img 
              src="https://images.unsplash.com/photo-1559839734-2b71ea197ec2?w=150&h=150&fit=crop&crop=face&auto=format"
              alt="Dr. Sarah Johnson" 
              className="profile-pic"
            />
            <div className="profile-info">
              <span className="profile-name">Dr. Sarah Johnson</span>
              <span className="profile-role">Neurologist</span>
            </div>
            <FaChevronDown className={`profile-chevron ${showProfileMenu ? 'rotated' : ''}`} />
          </div>

          {/* Profile Dropdown */}
          {showProfileMenu && (
            <div className="profile-dropdown">
              <div className="dropdown-item">
                <FaUser className="dropdown-icon" />
                <span>My Profile</span>
              </div>
              <div className="dropdown-item">
                <FaCog className="dropdown-icon" />
                <span>Settings</span>
              </div>
              <div className="dropdown-divider"></div>
              <div className="dropdown-item logout">
                <FaSignOutAlt className="dropdown-icon" />
                <button className="logout-btn" onClick={handleLogout}>
                <span>Logout</span>
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </nav>
  );
}
