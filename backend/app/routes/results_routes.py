# backend/app/routes/results_routes.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from datetime import datetime
from app.models.result import Result
from app.dependencies import get_db

router = APIRouter(prefix="/api/results", tags=["Results"])

@router.get("/")
async def get_all_results(db: Session = Depends(get_db)):
    return db.query(Result).all()

@router.get("/{result_id}")
async def get_result(result_id: int, db: Session = Depends(get_db)):
    return db.query(Result).filter(Result.id == result_id).first()

@router.post("/")
async def save_result(patient_id: int, file_name: str, model_used: str,
                      predicted_class: str, confidence: float, probabilities: dict,
                      db: Session = Depends(get_db)):
    new_result = Result(
        patient_id=patient_id,
        file_name=file_name,
        model_used=model_used,
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities=probabilities,
        analyzed_at=datetime.utcnow()
    )
    db.add(new_result)
    db.commit()
    db.refresh(new_result)
    return new_result
