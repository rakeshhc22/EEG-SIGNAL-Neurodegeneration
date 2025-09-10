# backend/app/routes/patients_routes.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from datetime import date
from app.models.patient import Patient
from app.dependencies import get_db

router = APIRouter(prefix="/api/patients", tags=["Patients"])

@router.post("/")
async def create_patient(name: str, age: int, gender: str, medical_history: str = "", db: Session = Depends(get_db)):
    patient = Patient(
        name=name,
        age=age,
        gender=gender,
        medical_history=medical_history,
        created_at=date.today()
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient

@router.get("/")
async def get_all_patients(db: Session = Depends(get_db)):
    return db.query(Patient).all()

@router.get("/{patient_id}")
async def get_patient(patient_id: int, db: Session = Depends(get_db)):
    return db.query(Patient).filter(Patient.id == patient_id).first()
