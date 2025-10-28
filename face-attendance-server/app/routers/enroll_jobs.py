# app/routers/enroll_jobs.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime, timezone
from ..db import SessionLocal

router = APIRouter(prefix="/v1/enroll_jobs", tags=["enroll_jobs"])

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

@router.post("")
def create_job(emp_id: str, device_id: str, db: Session = Depends(get_db)):
    e = db.execute(text("SELECT 1 FROM employees WHERE emp_id=:id"), {"id": emp_id}).first()
    if not e:
        raise HTTPException(404, "emp_id not found")
    row = db.execute(text("""
      INSERT INTO enroll_jobs(emp_id, device_id) VALUES (:emp_id, :device_id)
      RETURNING id, emp_id, device_id, status, created_at
    """), {"emp_id": emp_id, "device_id": device_id}).mappings().first()
    db.commit()
    return dict(row)

@router.get("/next")
def next_job(device_id: str, db: Session = Depends(get_db)):
    r = db.execute(text("""
      SELECT id, emp_id, device_id, status, created_at
      FROM enroll_jobs
      WHERE status='pending' AND device_id=:d
      ORDER BY id
      LIMIT 1
    """), {"d": device_id}).mappings().first()
    return dict(r) if r else None

@router.post("/{job_id}/start")
def start_job(job_id: int, db: Session = Depends(get_db)):
    r = db.execute(text("""
      UPDATE enroll_jobs
      SET status='running', started_at=:ts
      WHERE id=:id AND status='pending'
      RETURNING id
    """), {"id": job_id, "ts": datetime.now(timezone.utc)}).first()
    db.commit()
    if not r: raise HTTPException(409, "job not pending or not found")
    return {"status":"ok"}

@router.post("/{job_id}/done")
def done_job(job_id: int, ok: bool = True, notes: str = "", db: Session = Depends(get_db)):
    status = "done" if ok else "failed"
    r = db.execute(text("""
      UPDATE enroll_jobs
      SET status=:s, finished_at=:ts, notes=:n
      WHERE id=:id AND status IN ('running','pending')
      RETURNING id
    """), {"id": job_id, "s": status, "ts": datetime.now(timezone.utc), "n": notes}).first()
    db.commit()
    if not r: raise HTTPException(409, "job not running/pending or not found")
    return {"status":"ok"}
