# app/routers/recognize.py
import os, numpy as np
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.db import get_db
from app.utils.match import knn_cosine

router = APIRouter()

MODEL        = os.getenv("MODEL_NAME", "mobilefacenet-192d")
TH_CENTROID  = float(os.getenv("CENTROID_THRESHOLD", "0.65"))
TH_KNN       = float(os.getenv("KNN_THRESHOLD", "0.60"))
K_FALLBACK   = int(os.getenv("K_FALLBACK", "5"))
KNN_VOTE_MIN = int(os.getenv("KNN_VOTE_MIN", "3"))

def _l2norm(a: np.ndarray) -> np.ndarray:
    return a / (np.linalg.norm(a) + 1e-12)

@router.post("/v1/recognize")
def recognize(req: dict, db: Session = Depends(get_db)):
    # quick gates
    if req.get("liveness") is not None and float(req["liveness"]) < 0.5:
        return {"status":"reject", "result":None, "reason":"liveness_low"}
    if req.get("quality") is not None and float(req["quality"]) < 0.4:
        return {"status":"reject", "result":None, "reason":"quality_low"}

    q = np.asarray(req["embedding"], dtype=np.float64)
    q = _l2norm(q)

    # 1) Centroid-first (SQL thuần với pgvector)
    qv = "[" + ",".join(str(float(x)) for x in q) + "]"

    c_row = db.execute(text("""
    SELECT emp_id, (1 - (centroid <#> CAST(:qv AS vector(192)))) AS score
    FROM employee_centroids
    WHERE model = :m
    ORDER BY score DESC
    LIMIT 1
"""), {"qv": qv, "m": MODEL}).fetchone()


    if c_row and c_row.score is not None and float(c_row.score) >= TH_CENTROID:
        emp = db.execute(text("SELECT emp_id, full_name FROM employees WHERE emp_id=:e"),
                         {"e": c_row.emp_id}).fetchone()
        return {
            "status": "ok",
            "result": {
                "emp_id": c_row.emp_id,
                "full_name": emp.full_name if emp else c_row.emp_id,
                "score": float(c_row.score),
                "threshold": TH_CENTROID,
                "decision": "accepted",
                "via": "centroid"
            },
            "reason": None
        }

    # 2) Fallback: kNN + vote
    rows = knn_cosine(db, q.tolist(), k=K_FALLBACK, model=MODEL)
    if not rows:
        return {"status":"reject", "result":None, "reason":"empty_db"}

    from collections import defaultdict
    emp_scores = defaultdict(list)
    for r in rows:
        emp_scores[r.emp_id].append(float(r.score))

    voted = [(emp, len(sc), sum(sc)/len(sc)) for emp, sc in emp_scores.items()]
    voted.sort(key=lambda x: (x[1], x[2]), reverse=True)
    top_emp, votes, mean_cos = voted[0]

    if votes >= KNN_VOTE_MIN and mean_cos >= TH_KNN:
        emp = db.execute(text("SELECT emp_id, full_name FROM employees WHERE emp_id=:e"),
                         {"e": top_emp}).fetchone()
        return {
            "status": "ok",
            "result": {
                "emp_id": top_emp,
                "full_name": emp.full_name if emp else top_emp,
                "score": float(mean_cos),
                "threshold": TH_KNN,
                "decision": "accepted",
                "via": "knn_vote"
            },
            "reason": None
        }

    return {"status":"reject", "result":None, "reason":"below_threshold"}
