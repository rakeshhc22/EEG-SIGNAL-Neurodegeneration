# backend/app/routes/auth_routes.py
from fastapi import APIRouter, Form

router = APIRouter(prefix="/api", tags=["Auth"])

@router.post("/login")
def dummy_login(username: str = Form(...), password: str = Form(...)):
    """
    Dummy login endpoint — no authentication.
    Always succeeds and returns success message.
    """
    return {"message": "Login successful", "user": username}
