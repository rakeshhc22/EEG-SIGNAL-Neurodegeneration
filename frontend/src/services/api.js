// src/services/api.js

// Backend API base URL
const API_BASE = "http://localhost:8000/api";

// Helper for GET requests
export async function apiGet(endpoint) {
  try {
    const res = await fetch(`${API_BASE}${endpoint}`, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    });

    if (!res.ok) {
      const errorText = await res.text();
      throw new Error(`GET ${endpoint} failed: ${res.status} ${errorText}`);
    }

    return res.json();
  } catch (error) {
    console.error(`API GET Error for ${endpoint}:`, error);
    throw error;
  }
}

// Helper for POST requests (JSON body)
export async function apiPost(endpoint, body) {
  try {
    const res = await fetch(`${API_BASE}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const errorText = await res.text();
      throw new Error(`POST ${endpoint} failed: ${res.status} ${errorText}`);
    }

    return res.json();
  } catch (error) {
    console.error(`API POST Error for ${endpoint}:`, error);
    throw error;
  }
}

// Helper for file uploads
export async function apiUpload(endpoint, file) {
  try {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(`${API_BASE}${endpoint}`, {
      method: "POST",
      body: formData,
      // Don't set Content-Type header for FormData - browser will set it automatically
    });

    if (!res.ok) {
      let errorMessage;
      try {
        const errorData = await res.json();
        errorMessage = errorData.detail || errorData.error || `Upload failed with status ${res.status}`;
      } catch {
        const errorText = await res.text();
        errorMessage = errorText || `Upload failed with status ${res.status}`;
      }
      throw new Error(errorMessage);
    }

    return res.json();
  } catch (error) {
    console.error(`API Upload Error for ${endpoint}:`, error);
    throw error;
  }
}

// EEG Analysis function
export async function analyzeEEG(file) {
  return apiUpload("/analysis", file);
}

// Simple file upload function  
export async function uploadEEG(file) {
  return apiUpload("/upload", file);
}
