import React from "react";
import "../css/card.css";

export default function Card({ title, value, icon, color }) {
  return (
    <div className="card">
      <div className="card-left">
        <div className="card-icon" style={{ backgroundColor: color || "#edf2f7" }}>
          {icon}
        </div>
        <div className="card-info">
          <h3 className="card-value">{value}</h3>
          <p className="card-title">{title}</p>
        </div>
      </div>
    </div>
  );
}
