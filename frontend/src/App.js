// src/App.js
import React, { useState } from "react";
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";

import Navbar from "./components/Navbar";
import Sidebar from "./components/Sidebar";

import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";
import UploadEEG from "./pages/UploadEEG";
import SignalAnalysis from "./pages/SignalAnalysis";
import Results from "./pages/Results";
import Patients from "./pages/Patients";
import Reports from "./pages/Reports";

function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  return (
    <Router>
      <div className="app-shell">
        <Routes>
          {/* Login Page */}
          <Route
            path="/login"
            element={
              isLoggedIn ? (
                <Navigate to="/dashboard" replace />
              ) : (
                <Login onLogin={() => setIsLoggedIn(true)} />
              )
            }
          />

          {/* Protected Routes */}
          <Route
            path="/*"
            element={
              isLoggedIn ? (
                <>
                  <Navbar />
                  <Sidebar />
                  <Routes>
                    <Route path="/" element={<Navigate to="/dashboard" replace />} />
                    <Route path="/dashboard" element={<Dashboard />} />
                    <Route path="/upload" element={<UploadEEG />} />
                    <Route path="/analysis" element={<SignalAnalysis />} />
                    <Route path="/results" element={<Results />} />
                    <Route path="/patients" element={<Patients />} />
                    <Route path="/reports" element={<Reports />} />
                    <Route path="*" element={<Navigate to="/dashboard" replace />} />
                  </Routes>
                </>
              ) : (
                <Navigate to="/login" replace />
              )
            }
          />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
