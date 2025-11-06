# app/routers/recognize.py
import os, numpy as np
import struct
import zlib
import base64
import logging
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.db import get_db
from app.utils.match import knn_cosine
from app.utils.centroid_cache import cache
from app.utils.att_logic import decide_type, should_debounce, vn_now, last_event_today

logger = logging.getLogger(__name__)

router = APIRouter()

MODEL        = os.getenv("MODEL_NAME", "mobilefacenet-192d")
TH_CENTROID  = float(os.getenv("CENTROID_THRESHOLD", "0.65"))
TH_KNN       = float(os.getenv("KNN_THRESHOLD", "0.60"))
K_FALLBACK   = int(os.getenv("K_FALLBACK", "5"))
KNN_VOTE_MIN = int(os.getenv("KNN_VOTE_MIN", "3"))

def _l2norm(a: np.ndarray) -> np.ndarray:
    return a / (np.linalg.norm(a) + 1e-12)

def _save_attendance(
    db: Session,
    emp_id: str,
    device_id: str,
    score: float,
    liveness: float,
    decision: str,
    att_type: str = None,
    image_path: str = None
):
    """
    Lưu attendance record vào database
    
    Args:
        db: Database session
        emp_id: Employee ID (None nếu rejected)
        device_id: Device ID
        score: Recognition score
        liveness: Liveness score
        decision: "accepted" or "reject"
        att_type: "checkin", "checkout", hoặc None (auto-detect)
        image_path: Path to saved image (optional)
    """
    try:
        now_dt = vn_now()
        
        # Auto-detect type nếu chưa có (chỉ khi accepted và có emp_id)
        if att_type is None and decision == "accepted" and emp_id:
            att_type = decide_type(db, emp_id, now_dt)
            
            # Check debounce: chỉ debounce khi cùng type (checkin-checkin hoặc checkout-checkout)
            # và cách nhau < DEBOUNCE_SEC giây
            last_event = last_event_today(db, emp_id)
            if last_event and should_debounce(last_event["ts"], now_dt):
                # Chỉ debounce nếu cùng type
                last_type = last_event.get("type")
                if last_type == att_type:
                    delta_sec = (now_dt - last_event["ts"]).total_seconds()
                    logger.info(f"Attendance skipped (debounce): emp_id={emp_id}, type={att_type}, last_event={delta_sec:.1f}s ago")
                    return  # Skip save due to debounce
                else:
                    # Khác type (checkin -> checkout hoặc ngược lại) → luôn lưu
                    logger.info(f"Attendance allowed (different type): emp_id={emp_id}, last_type={last_type}, new_type={att_type}")
        
        # Insert attendance record
        db.execute(text("""
            INSERT INTO attendance(emp_id, device_id, score, liveness, decision, type, image_path)
            VALUES (:emp_id, :device_id, :score, :liveness, :decision, :type, :image_path)
        """), {
            "emp_id": emp_id if decision == "accepted" else None,
            "device_id": device_id,
            "score": score,
            "liveness": liveness,
            "decision": decision,
            "type": att_type if decision == "accepted" else None,
            "image_path": image_path
        })
        db.commit()
        logger.info(f"Attendance saved: emp_id={emp_id}, device={device_id}, decision={decision}, type={att_type}")
    except Exception as e:
        db.rollback()
        # Log error nhưng không fail request
        logger.error(f"Error saving attendance: {e}", exc_info=True)

# PHASE 2 OPTIMIZATION: Network decompression
def decompress_embedding(data: bytes) -> np.ndarray:
    """
    Decompress embedding từ client
    
    Args:
        data: Compressed bytes
        
    Returns:
        Embedding vector (192d, float32)
    """
    # Decompress
    unpacked = zlib.decompress(data)
    
    # Unpack binary
    emb_int16 = struct.unpack(f'{len(unpacked)//2}h', unpacked)
    
    # Convert back to float32
    emb = np.array(emb_int16, dtype=np.float32) / 32767.0
    
    return emb

@router.post("/v1/recognize")
def recognize(req: dict, db: Session = Depends(get_db)):
    device_id = req.get("device_id", "unknown")
    liveness = float(req.get("liveness", 1.0)) if req.get("liveness") is not None else 1.0
    quality = float(req.get("quality", 0.5)) if req.get("quality") is not None else 0.5
    
    # quick gates
    if liveness < 0.5:
        _save_attendance(db, None, device_id, 0.0, liveness, "reject", image_path=None)
        return {"status":"reject", "result":None, "reason":"liveness_low"}
    if quality < 0.4:
        _save_attendance(db, None, device_id, 0.0, liveness, "reject", image_path=None)
        return {"status":"reject", "result":None, "reason":"quality_low"}

    # PHASE 2 OPTIMIZATION: Decompress embedding nếu có
    embedding_compressed = req.get("embedding_compressed")
    if embedding_compressed:
        try:
            # Decode base64
            compressed_bytes = base64.b64decode(embedding_compressed)
            # Decompress
            q = decompress_embedding(compressed_bytes)
            q = _l2norm(q)
        except Exception as e:
            # Fallback to uncompressed nếu decompress fail
            q = np.asarray(req.get("embedding", []), dtype=np.float64)
            if len(q) == 0:
                return {"status":"reject", "result":None, "reason":"invalid_embedding"}
            q = _l2norm(q)
    else:
        # Uncompressed (backward compatible)
        q = np.asarray(req.get("embedding", []), dtype=np.float64)
        if len(q) == 0:
            return {"status":"reject", "result":None, "reason":"invalid_embedding"}
        q = _l2norm(q)

    # PHASE 2 OPTIMIZATION: In-memory centroid matching (nhanh hơn 80-90%)
    cache.ensure_fresh(db)
    result = cache.match(q, TH_CENTROID)
    
    if result:
        emp_id, full_name, score = result
        # Lưu attendance record
        _save_attendance(db, emp_id, device_id, float(score), liveness, "accepted", image_path=None)
        return {
            "status": "ok",
            "result": {
                "emp_id": emp_id,
                "full_name": full_name,
                "score": float(score),
                "threshold": TH_CENTROID,
                "decision": "accepted",
                "via": "centroid_memory"
            },
            "reason": None
        }
    
    # Fallback: Old SQL method (nếu in-memory không match)
    # Chỉ dùng khi cần debug hoặc cache chưa load
    use_sql_fallback = os.getenv("USE_SQL_CENTROID", "false").lower() == "true"
    if use_sql_fallback:
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
            # Lưu attendance record
            _save_attendance(db, c_row.emp_id, device_id, float(c_row.score), liveness, "accepted", image_path=None)
            return {
                "status": "ok",
                "result": {
                    "emp_id": c_row.emp_id,
                    "full_name": emp.full_name if emp else c_row.emp_id,
                    "score": float(c_row.score),
                    "threshold": TH_CENTROID,
                    "decision": "accepted",
                    "via": "centroid_sql"
                },
                "reason": None
            }

    # 2) Fallback: kNN + vote
    rows = knn_cosine(db, q.tolist(), k=K_FALLBACK, model=MODEL)
    if not rows:
        _save_attendance(db, None, device_id, 0.0, liveness, "reject", image_path=None)
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
        # Lưu attendance record
        _save_attendance(db, top_emp, device_id, float(mean_cos), liveness, "accepted", image_path=None)
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

    # Rejected - below threshold (mean_cos từ voted[0])
    _save_attendance(db, None, device_id, float(mean_cos), liveness, "reject", image_path=None)
    return {"status":"reject", "result":None, "reason":"below_threshold"}
