# backend/app/routes/reports_routes.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from app.dependencies import get_db
from app.models.patient import Patient
from app.models.result import Result

router = APIRouter(prefix="/api/reports", tags=["Reports"])


@router.get("/")
async def generate_report(db: Session = Depends(get_db)):
    """
    Generate a summary report:
    - Total patients
    - Total EEG analyses
    - Seizure vs Normal counts
    - Recent analyses (last 7 days)
    """

    total_patients = db.query(Patient).count()
    total_results = db.query(Result).count()

    seizure_count = db.query(Result).filter(Result.predicted_class == "Seizure Detected").count()
    normal_count = db.query(Result).filter(Result.predicted_class == "Normal").count()

    # Last 7 days activity
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    recent_results = db.query(Result).filter(Result.analyzed_at >= seven_days_ago).count()

    return {
        "total_patients": total_patients,
        "total_results": total_results,
        "seizure_cases": seizure_count,
        "normal_cases": normal_count,
        "recent_activity": recent_results,
    }
