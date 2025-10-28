# app/routers/enroll.py
import os
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.db import get_db

router = APIRouter()

MODEL_DEFAULT = os.getenv("MODEL_NAME", "mobilefacenet-192d")
VEC_DIM = 192

def _vec_literal(v):
    # pgvector (bản của bạn) yêu cầu dấu ngoặc vuông
    return "[" + ",".join(str(float(x)) for x in v) + "]"

def _l2_norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-12
    return v / n

def _coerce_vec_from_db(v) -> np.ndarray:
    """
    Trả về numpy array float64, chiều VEC_DIM.
    - Nếu driver trả list/tuple -> convert trực tiếp
    - Nếu driver trả str dạng "[...,...]" -> parse thủ công
    """
    if isinstance(v, np.ndarray):
        arr = v.astype(np.float64, copy=False)
    elif isinstance(v, (list, tuple)):
        arr = np.asarray(v, dtype=np.float64)
    elif isinstance(v, str):
        s = v.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        parts = [p.strip() for p in s.split(",") if p.strip() != ""]
        arr = np.asarray([float(p) for p in parts], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported vector type from DB: {type(v)}")

    # Chuẩn hoá chiều 192 cho chắc chắn
    if arr.shape[0] != VEC_DIM:
        fixed = np.zeros(VEC_DIM, dtype=np.float64)
        n = min(VEC_DIM, arr.shape[0])
        fixed[:n] = arr[:n]
        arr = fixed
    return arr

def _recompute_centroid(db: Session, emp_id: str, model_name: str):
    rows = db.execute(text("""
        SELECT embedding
        FROM embeddings
        WHERE emp_id = :e AND model = :m
    """), {"e": emp_id, "m": model_name}).fetchall()

    if not rows:
        # Không còn sample -> xoá centroid nếu có
        db.execute(text("""
            DELETE FROM employee_centroids
            WHERE emp_id = :e AND model = :m
        """), {"e": emp_id, "m": model_name})
        return

    vecs = []
    for r in rows:
        vecs.append(_coerce_vec_from_db(r.embedding))
    X = np.vstack(vecs)  # (n, 192)
    mu = _l2_norm(X.mean(axis=0))

    db.execute(text("""
        INSERT INTO employee_centroids(emp_id, model, dim, centroid, n_samples, updated_at)
        VALUES (:e, :m, :d, CAST(:c AS vector(192)), :n, now())
        ON CONFLICT (emp_id, model) DO UPDATE
          SET centroid = EXCLUDED.centroid,
              n_samples = EXCLUDED.n_samples,
              updated_at = now()
    """), {
        "e": emp_id,
        "m": model_name,
        "d": VEC_DIM,
        "c": _vec_literal(mu.tolist()),
        "n": X.shape[0],
    })

@router.post("/v1/enroll/{emp_id}")
def enroll(emp_id: str, req: dict, db: Session = Depends(get_db)):
    """
    Body:
    {
      "model": "mobilefacenet-192d",
      "embeddings": [[...192 floats...], ...]
    }
    """
    # 1) Kiểm tra nhân viên tồn tại (FK)
    emp = db.execute(text("SELECT emp_id FROM employees WHERE emp_id = :e"), {"e": emp_id}).fetchone()
    if not emp:
        raise HTTPException(status_code=400, detail=f"employee {emp_id} not found")

    # 2) Lấy model và embeddings
    model_name = (req.get("model") or MODEL_DEFAULT).strip()
    embs = req.get("embeddings") or []
    if not isinstance(embs, list) or len(embs) == 0:
        raise HTTPException(status_code=400, detail="empty embeddings")

    # 3) INSERT từng vector (ép chiều 192 và CAST literal)
    ins = text("""
        INSERT INTO embeddings(emp_id, model, embedding)
        VALUES (:e, :m, CAST(:v AS vector(192)))
    """)

    inserted = 0
    for v in embs:
        if not isinstance(v, (list, tuple, np.ndarray)):
            continue
        v = np.asarray(v, dtype=np.float64)
        if v.shape[0] != VEC_DIM:
            fixed = np.zeros(VEC_DIM, dtype=np.float64)
            n = min(VEC_DIM, v.shape[0])
            fixed[:n] = v[:n]
            v = fixed
        db.execute(ins, {"e": emp_id, "m": model_name, "v": _vec_literal(v.tolist())})
        inserted += 1

    if inserted == 0:
        raise HTTPException(status_code=400, detail="no valid embedding vectors")

    # 4) Recompute centroid (sau khi insert)
    _recompute_centroid(db, emp_id, model_name)

    # 5) Commit
    db.commit()
    return {"status": "ok", "inserted": inserted}
