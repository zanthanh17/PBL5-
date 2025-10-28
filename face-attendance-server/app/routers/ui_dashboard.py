from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import text
from ..db import SessionLocal
import urllib.parse
import csv, io
import pytz, os
from datetime import datetime

WORKDAY_TZ = os.getenv("WORKDAY_TZ", "Asia/Ho_Chi_Minh")

router = APIRouter()

templates = Jinja2Templates(directory="app/templates")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



@router.get("/dashboard", response_class=HTMLResponse)
def dashboard(
    request: Request,
    emp_id: str | None = None,
    emp_name: str | None = None,
    date_from: str | None = None,  # "2025-10-21"
    date_to: str | None = None,
    decision: str | None = None,   # accepted/reject
    page: int = 1,
    page_size: int = 20,
    db: Session = Depends(get_db),
):
    tz = pytz.timezone(WORKDAY_TZ)
    today = datetime.now(tz).date()

    def parse_date(value: str | None, default):
        if not value:
            return default
        try:
            return datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError:
            return default

    start_date = parse_date(date_from, today)
    end_date = parse_date(date_to, start_date)
    if end_date < start_date:
        start_date, end_date = end_date, start_date

    params: dict[str, object] = {
        "tz": WORKDAY_TZ,
        "date_from": start_date,
        "date_to": end_date,
    }
    cond_clauses = [
        "DATE(timezone(:tz, att.ts)) BETWEEN :date_from AND :date_to",
        "(att.emp_id IS NULL OR em.emp_id IS NOT NULL)",
    ]

    if emp_id:
        cond_clauses.append("att.emp_id = :emp_id"); params["emp_id"] = emp_id
    if emp_name:
        cond_clauses.append("em.full_name ILIKE :emp_name"); params["emp_name"] = f"%{emp_name}%"
    if decision in ("accepted","reject"):
        cond_clauses.append("att.decision = :dc"); params["dc"] = decision

    cond = "WHERE " + " AND ".join(cond_clauses)

    offset = (page - 1) * page_size
    sql = text(f"""
      SELECT att.id,
             att.emp_id,
             att.device_id,
             att.decision,
             att.score,
             att.liveness,
             att.ts,
             em.full_name,
             em.dept,
             em.photo_path
      FROM attendance att
      LEFT JOIN employees em ON em.emp_id = att.emp_id
      {cond}
      ORDER BY att.ts DESC
      LIMIT :limit OFFSET :offset
    """)
    params["limit"] = page_size + 1
    params["offset"] = offset

    rows = db.execute(sql, params).mappings().all()
    has_more = len(rows) > page_size
    rows = rows[:page_size]

    view_rows = []
    for r in rows:
        d = dict(r)
        d.setdefault("full_name", "")
        d.setdefault("dept", "")
        ts = d.get("ts")
        d["ts_local"] = ts.astimezone(tz).strftime("%Y-%m-%d %H:%M:%S") if ts else ""
        full_name = d.get("full_name") or ""
        if full_name:
            initials = "".join(part[0].upper() for part in full_name.split()[:2] if part)
        else:
            emp_code = (d.get("emp_id") or "")[:2]
            initials = emp_code.upper() or "?"
        d["initials"] = initials
        view_rows.append(d)

    date_from_str = start_date.strftime("%Y-%m-%d")
    date_to_str = end_date.strftime("%Y-%m-%d")

    q = {
        "emp_id": emp_id or "",
        "emp_name": emp_name or "",
        "date_from": date_from_str,
        "date_to": date_to_str,
        "decision": decision or "",
    }
    qstr = urllib.parse.urlencode(q)

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "title": "Dashboard - Face Attendance",
        "rows": view_rows,
        "page": page,
        "has_more": has_more,
        "qstr": qstr,
        "emp_id": emp_id,
        "emp_name": emp_name,
        "date_from": date_from_str,
        "date_to": date_to_str,
        "decision": decision,
    })


@router.get("/reports", response_class=HTMLResponse)
def reports(
    request: Request,
    q: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    export: bool = False,
    db: Session = Depends(get_db),
):
    tz = pytz.timezone(WORKDAY_TZ)
    today = datetime.now(tz).date()

    def parse_date(value: str | None, default):
        if not value:
            return default
        try:
            return datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError:
            return default

    start_date = parse_date(date_from, today)
    end_date = parse_date(date_to, start_date)
    if end_date < start_date:
        start_date, end_date = end_date, start_date

    params = {"tz": WORKDAY_TZ, "date_from": start_date, "date_to": end_date}
    cond = """
      WHERE att.decision = 'accepted'
        AND att.emp_id IS NOT NULL
        AND em.emp_id IS NOT NULL
        AND DATE(timezone(:tz, att.ts)) BETWEEN :date_from AND :date_to
    """
    if q:
        cond += " AND (att.emp_id ILIKE :q OR em.full_name ILIKE :q)"
        params["q"] = f"%{q}%"

    sql = text(f"""
      SELECT em.emp_id,
             em.full_name,
             em.dept,
             em.photo_path,
             DATE(timezone(:tz, att.ts)) AS work_date,
             MIN(att.ts) AS first_ts,
             MAX(att.ts) AS last_ts,
             COUNT(*) AS total_scans
      FROM attendance att
      JOIN employees em ON em.emp_id = att.emp_id
      {cond}
      GROUP BY em.emp_id, em.full_name, em.dept, em.photo_path, work_date
      ORDER BY work_date DESC, em.emp_id
    """)

    rows = db.execute(sql, params).mappings().all()

    view_rows = []
    for r in rows:
        first_ts = r["first_ts"]
        last_ts = r["last_ts"]
        if not first_ts or not last_ts:
            continue
        first_local = first_ts.astimezone(tz)
        last_local = last_ts.astimezone(tz)
        duration_seconds = max((last_local - first_local).total_seconds(), 0)
        full_name = r["full_name"] or ""
        initials = "".join(part[0].upper() for part in full_name.split()[:2] if part) or (r["emp_id"] or "?")[:2].upper()

        view_rows.append({
            "emp_id": r["emp_id"],
            "full_name": full_name or r["emp_id"],
            "dept": r["dept"],
            "photo_path": r["photo_path"],
            "initials": initials,
            "checkin": first_local.strftime("%H:%M:%S"),
            "checkout": last_local.strftime("%H:%M:%S"),
            "duration_hours": round(duration_seconds / 3600.0, 2),
            "duration_human": f"{int(duration_seconds // 3600)}h {int((duration_seconds % 3600) // 60)}m",
            "total_scans": r["total_scans"],
            "work_date": r["work_date"].strftime("%Y-%m-%d") if r["work_date"] else "",
        })

    if export:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["work_date","emp_id","full_name","dept","checkin","checkout","hours","scan_count"])
        for r in view_rows:
            writer.writerow([
                r["work_date"],
                r["emp_id"],
                r["full_name"],
                r["dept"] or "",
                r["checkin"],
                r["checkout"],
                f"{r['duration_hours']:.2f}",
                r["total_scans"],
            ])
        filename = f"attendance-report-{start_date}-to-{end_date}.csv"
        return Response(content=buf.getvalue(), media_type="text/csv", headers={
            "Content-Disposition": f"attachment; filename={filename}"
        })

    return templates.TemplateResponse("reports.html", {
        "request": request,
        "title": "Reports - Face Attendance",
        "rows": view_rows,
        "q": q or "",
        "date_from": start_date.strftime("%Y-%m-%d"),
        "date_to": end_date.strftime("%Y-%m-%d"),
        "today": today.strftime("%Y-%m-%d"),
    })


