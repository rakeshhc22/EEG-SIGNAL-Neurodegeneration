# backend/run.py

"""
FastAPI Server Startup Script
Fixed for Windows multiprocessing and model loading
"""

import uvicorn
import sys
import os

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

if __name__ == "__main__":
    # Fix multiprocessing issues on Windows
    import multiprocessing
    multiprocessing.freeze_support()
    
    try:
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",  # Accessible from other devices
            port=8000,
            reload=True,     # Auto-reload on code changes
            workers=1,       # Single worker to avoid multiprocessing issues
            log_level="info"
        )
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)
