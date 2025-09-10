# backend/app/models/patient.py
from sqlalchemy import Column, Integer, String, Date, Text
from app.db import Base


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String(10), nullable=False)
    medical_history = Column(Text, nullable=True)
    created_at = Column(Date, nullable=False)

    def __repr__(self):
        return f"<Patient(id={self.id}, name={self.name})>"
