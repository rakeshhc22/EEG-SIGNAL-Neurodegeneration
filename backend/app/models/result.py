# backend/app/models/result.py
from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, JSON
from sqlalchemy.orm import relationship
from app.db import Base


class Result(Base):
    __tablename__ = "results"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    file_name = Column(String(255), nullable=False)
    model_used = Column(String(50), nullable=False)   # e.g. QDA, TabNet
    predicted_class = Column(String(50), nullable=False)  # Normal / Seizure Detected
    confidence = Column(Float, nullable=False)
    probabilities = Column(JSON, nullable=True)  # store probability distribution
    analyzed_at = Column(DateTime, nullable=False)

    # relationship to Patient
    patient = relationship("Patient", backref="results")

    def __repr__(self):
        return f"<Result(id={self.id}, patient_id={self.patient_id}, class={self.predicted_class})>"
