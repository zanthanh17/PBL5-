# app/routers/attendance.py
from fastapi import APIRouter, Query, Depends, Response
from sqlalchemy.orm import Session
from sqlalchemy import text  # <-- quan trọng
from ..db import SessionLocal
import csv, io

router = APIRouter(prefix="/v1/attendance", tags=["attendance"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("")
def list_attendance(
    db: Session = Depends(get_db),
    emp_id: str | None = None,
    date_from: str | None = None,   # ISO 8601, ví dụ: 2025-10-21T00:00:00Z hoặc 2025-10-21
    date_to: str | None = None,
    limit: int = 200
):
    cond = "WHERE 1=1"
    params: dict = {}
    if emp_id:
        cond += " AND emp_id = :emp_id"; params["emp_id"] = emp_id
    if date_from:
        cond += " AND ts >= :df"; params["df"] = date_from
    if date_to:
        cond += " AND ts <= :dt"; params["dt"] = date_to

    sql = text(f"""
        SELECT id, emp_id, device_id, decision, score, liveness, ts, type
        FROM attendance
        {cond}
        ORDER BY ts DESC
        LIMIT :limit
    """)
    params["limit"] = limit

    rows = db.execute(sql, params).mappings().all()
    return [dict(r) for r in rows]

@router.get("/export.csv")
def export_csv(
    db: Session = Depends(get_db),
    emp_id: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
):
    cond = "WHERE 1=1"
    params: dict = {}
    if emp_id:
        cond += " AND emp_id = :emp_id"; params["emp_id"] = emp_id
    if date_from:
        cond += " AND ts >= :df"; params["df"] = date_from
    if date_to:
        cond += " AND ts <= :dt"; params["dt"] = date_to

    sql = text(f"""
        SELECT ts, emp_id, device_id, decision, score, liveness, type
        FROM attendance
        {cond}
        ORDER BY ts DESC
    """)

    rows = db.execute(sql, params).all()

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["timestamp","emp_id","device_id","decision","score","liveness","type"])
    for r in rows:
        w.writerow(list(r))
    return Response(content=buf.getvalue(), media_type="text/csv")
