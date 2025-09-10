# backend/run.py
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",   # accessible from other devices on same network
        port=8000,
        reload=True       # auto-reload on code changes (for dev mode)
    )
