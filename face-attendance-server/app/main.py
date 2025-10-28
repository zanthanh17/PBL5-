from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from .routers import health, enroll, recognize, attendance, ui_dashboard, employees, enroll_jobs
from app.utils.centroid_cache import cache
from app.db import SessionLocal

app = FastAPI(title="Face Attendance Server")

@app.on_event("startup")
def _startup():
    db = SessionLocal()
    try:
        cache.load_now(db)
    finally:
        db.close()

app.include_router(health.router)
app.include_router(enroll.router)
app.include_router(recognize.router)
app.include_router(attendance.router)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/uploads", StaticFiles(directory="app/uploads"), name="uploads")
app.include_router(ui_dashboard.router)
app.include_router(employees.router)
app.include_router(enroll_jobs.router)
