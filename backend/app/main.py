# backend/app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.db import Base, engine
from app.routes import (
    analysis_routes,
    patients_routes,
    results_routes,
    reports_routes,
    auth_routes,
    eeg_routes
)

# ✅ Create DB tables at startup
Base.metadata.create_all(bind=engine)

# ✅ Initialize FastAPI app
app = FastAPI(title="NeuroDetect API", version="1.0.0")

# ✅ CORS setup (allow React frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production restrict to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Register API routers
app.include_router(auth_routes.router)
app.include_router(patients_routes.router)
app.include_router(results_routes.router)
app.include_router(reports_routes.router)
app.include_router(analysis_routes.router)
app.include_router(eeg_routes.router)

# ✅ Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to NeuroDetect API"}
