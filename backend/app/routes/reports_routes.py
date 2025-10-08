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
    - Normal vs Seizure vs Neurodegeneration counts
    - Recent analyses (last 7 days)
    """
    total_patients = db.query(Patient).count()
    total_results = db.query(Result).count()
    
    # Count all 3 classes
    normal_count = db.query(Result).filter(Result.predicted_class == "Normal").count()
    seizure_count = db.query(Result).filter(Result.predicted_class == "Seizure Detected").count()
    neurodegeneration_count = db.query(Result).filter(Result.predicted_class == "Neurodegeneration Detected").count()
    
    # Last 7 days activity
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    recent_results = db.query(Result).filter(Result.analyzed_at >= seven_days_ago).count()
    
    # Recent activity by class (last 7 days)
    recent_normal = db.query(Result).filter(
        Result.analyzed_at >= seven_days_ago,
        Result.predicted_class == "Normal"
    ).count()
    
    recent_seizure = db.query(Result).filter(
        Result.analyzed_at >= seven_days_ago,
        Result.predicted_class == "Seizure Detected"
    ).count()
    
    recent_neurodegeneration = db.query(Result).filter(
        Result.analyzed_at >= seven_days_ago,
        Result.predicted_class == "Neurodegeneration Detected"
    ).count()
    
    return {
        "total_patients": total_patients,
        "total_results": total_results,
        "classification_summary": {
            "normal_cases": normal_count,
            "seizure_cases": seizure_count,
            "neurodegeneration_cases": neurodegeneration_count
        },
        "recent_activity": {
            "total_recent": recent_results,
            "recent_normal": recent_normal,
            "recent_seizure": recent_seizure,
            "recent_neurodegeneration": recent_neurodegeneration
        },
        "class_distribution": {
            "normal_percentage": round((normal_count / total_results) * 100, 2) if total_results > 0 else 0,
            "seizure_percentage": round((seizure_count / total_results) * 100, 2) if total_results > 0 else 0,
            "neurodegeneration_percentage": round((neurodegeneration_count / total_results) * 100, 2) if total_results > 0 else 0
        }
    }

@router.get("/class-statistics")
async def get_class_statistics(db: Session = Depends(get_db)):
    """
    Get detailed statistics for each classification class
    """
    normal_results = db.query(Result).filter(Result.predicted_class == "Normal").all()
    seizure_results = db.query(Result).filter(Result.predicted_class == "Seizure Detected").all()
    neurodegeneration_results = db.query(Result).filter(Result.predicted_class == "Neurodegeneration Detected").all()
    
    def calculate_stats(results):
        if not results:
            return {"count": 0, "avg_confidence": 0, "min_confidence": 0, "max_confidence": 0}
        
        confidences = [r.confidence for r in results]
        return {
            "count": len(results),
            "avg_confidence": round(sum(confidences) / len(confidences), 2),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences)
        }
    
    return {
        "normal_stats": calculate_stats(normal_results),
        "seizure_stats": calculate_stats(seizure_results),
        "neurodegeneration_stats": calculate_stats(neurodegeneration_results)
    }
