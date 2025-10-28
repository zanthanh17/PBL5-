from fastapi import APIRouter, Depends, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, Response, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Optional
from ..db import SessionLocal
import os, shutil, csv, io, uuid
import pytz
from .enroll_jobs import create_job as api_create_job

WORKDAY_TZ = os.getenv("WORKDAY_TZ", "Asia/Ho_Chi_Minh")
templates = Jinja2Templates(directory="app/templates")

router = APIRouter(tags=["employees"])

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

# ---------- HTML PAGES ----------

@router.get("/employees", response_class=HTMLResponse)
def employees_list(
    request: Request,
    q: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    db: Session = Depends(get_db),
):
    cond = "WHERE 1=1"
    params = {}
    if q:
        cond += " AND (emp_id ILIKE :q OR full_name ILIKE :q OR COALESCE(dept,'') ILIKE :q)"
        params["q"] = f"%{q}%"

    offset = (page-1)*page_size
    sql = text(f"""
      SELECT emp_id, full_name, dept, photo_path
      FROM employees
      {cond}
      ORDER BY emp_id
      LIMIT :limit OFFSET :offset
    """)
    params["limit"] = page_size + 1
    params["offset"] = offset
    rows = db.execute(sql, params).mappings().all()
    has_more = len(rows) > page_size
    rows = rows[:page_size]

    return templates.TemplateResponse("employees_list.html", {
        "request": request, "rows": rows, "q": q or "", "page": page, "has_more": has_more
    })

@router.get("/employees/new", response_class=HTMLResponse)
def employees_new(request: Request):
    return templates.TemplateResponse("employees_new.html", {"request": request})

@router.post("/employees/new")
def employees_create(
    emp_id: str = Form(...),
    full_name: str = Form(...),
    dept: Optional[str] = Form(None),
    photo: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
):
    # save avatar if any
    photo_path = None
    if photo and photo.filename:
        os.makedirs("app/uploads/avatars", exist_ok=True)
        ext = os.path.splitext(photo.filename)[1].lower() or ".jpg"
        fname = f"{emp_id}_{uuid.uuid4().hex}{ext}"
        out = os.path.join("app", "uploads", "avatars", fname)
        with open(out, "wb") as f:
            shutil.copyfileobj(photo.file, f)
        photo_path = f"/uploads/avatars/{fname}"

    sql = text("""
      INSERT INTO employees(emp_id, full_name, dept, photo_path)
      VALUES (:emp_id, :full_name, :dept, :photo_path)
      ON CONFLICT (emp_id) DO UPDATE
      SET full_name=EXCLUDED.full_name,
          dept=EXCLUDED.dept,
          photo_path=COALESCE(EXCLUDED.photo_path, employees.photo_path)
    """)
    db.execute(sql, {"emp_id": emp_id, "full_name": full_name, "dept": dept, "photo_path": photo_path})
    db.commit()
    return RedirectResponse(url=f"/employees/{emp_id}", status_code=303)

@router.get("/employees/{emp_id}", response_class=HTMLResponse)
def employees_detail(
    request: Request, emp_id: str, db: Session = Depends(get_db),
    page: int = 1, page_size: int = 20
):
    e = db.execute(text("SELECT emp_id, full_name, dept, photo_path FROM employees WHERE emp_id=:emp_id"),
                   {"emp_id": emp_id}).mappings().first()
    if not e:
        return templates.TemplateResponse("employees_notfound.html", {"request": request, "emp_id": emp_id}, status_code=404)

    offset = (page-1)*page_size
    rows = db.execute(text("""
      SELECT id, ts, device_id, type, decision, score, liveness
      FROM attendance
      WHERE emp_id=:emp_id
      ORDER BY ts DESC
      LIMIT :limit OFFSET :offset
    """), {"emp_id": emp_id, "limit": page_size+1, "offset": offset}).mappings().all()
    has_more = len(rows) > page_size
    rows = rows[:page_size]

    tz = pytz.timezone(WORKDAY_TZ)
    view_rows = []
    for r in rows:
        d = dict(r)
        d["ts_local"] = d["ts"].astimezone(tz).strftime("%Y-%m-%d %H:%M:%S")
        view_rows.append(d)

    return templates.TemplateResponse("employees_detail.html", {
        "request": request, "e": e, "rows": view_rows, "page": page, "has_more": has_more
    })

@router.post("/employees/{emp_id}/edit")
def employees_update(
    emp_id: str,
    full_name: str = Form(...),
    dept: Optional[str] = Form(None),
    photo: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
):
    photo_path = None
    if photo and photo.filename:
        os.makedirs("app/uploads/avatars", exist_ok=True)
        ext = os.path.splitext(photo.filename)[1].lower() or ".jpg"
        fname = f"{emp_id}_{uuid.uuid4().hex}{ext}"
        out = os.path.join("app", "uploads", "avatars", fname)
        with open(out, "wb") as f:
            shutil.copyfileobj(photo.file, f)
        photo_path = f"/uploads/avatars/{fname}"

    if photo_path:
        sql = text("UPDATE employees SET full_name=:full_name, dept=:dept, photo_path=:photo WHERE emp_id=:emp_id")
        db.execute(sql, {"full_name": full_name, "dept": dept, "photo": photo_path, "emp_id": emp_id})
    else:
        sql = text("UPDATE employees SET full_name=:full_name, dept=:dept WHERE emp_id=:emp_id")
        db.execute(sql, {"full_name": full_name, "dept": dept, "emp_id": emp_id})
    db.commit()
    return RedirectResponse(url=f"/employees/{emp_id}", status_code=303)

@router.post("/employees/{emp_id}/delete")
def employees_delete(emp_id: str, db: Session = Depends(get_db)):
    db.execute(text("DELETE FROM employees WHERE emp_id=:emp_id"), {"emp_id": emp_id})
    db.commit()
    return RedirectResponse(url="/employees", status_code=303)

@router.get("/employees/{emp_id}/export.csv")
def employees_export(emp_id: str, db: Session = Depends(get_db)):
    rows = db.execute(text("""
      SELECT ts, device_id, type, decision, score, liveness
      FROM attendance
      WHERE emp_id=:emp_id
      ORDER BY ts DESC
    """), {"emp_id": emp_id}).all()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["timestamp_utc","device_id","type","decision","score","liveness"])
    for r in rows:
        w.writerow(list(r))
    return Response(content=buf.getvalue(), media_type="text/csv")

# ---------- JSON API (debug/test nhanh) ----------

@router.post("/v1/employees")
def api_create_employee(emp_id: str = Form(...), full_name: str = Form(...), dept: Optional[str] = Form(None), db: Session = Depends(get_db)):
    db.execute(text("""
      INSERT INTO employees(emp_id, full_name, dept)
      VALUES (:emp_id, :full_name, :dept)
      ON CONFLICT (emp_id) DO UPDATE SET full_name=EXCLUDED.full_name, dept=EXCLUDED.dept
    """), {"emp_id": emp_id, "full_name": full_name, "dept": dept})
    db.commit()
    return JSONResponse({"status":"ok"})

@router.delete("/v1/employees/{emp_id}")
def api_delete_employee(emp_id: str, db: Session = Depends(get_db)):
    db.execute(text("DELETE FROM employees WHERE emp_id=:emp_id"), {"emp_id": emp_id})
    db.commit()
    return JSONResponse({"status":"ok"})


# Trong router employees:
@router.post("/employees/{emp_id}/enqueue-enroll")
def employees_enqueue_enroll(emp_id: str, device_id: str = Form(...), db: Session = Depends(get_db)):
    e = db.execute(text("SELECT 1 FROM employees WHERE emp_id=:x"), {"x": emp_id}).first()
    if not e: raise HTTPException(404, "emp_id not found")
    db.execute(text("INSERT INTO enroll_jobs(emp_id, device_id) VALUES (:e,:d)"),
               {"e": emp_id, "d": device_id})
    db.commit()
    return RedirectResponse(url=f"/employees/{emp_id}", status_code=303)
