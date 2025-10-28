# app/routers/recognize_optimized.py
"""
Recognition với thuật toán cải tiến để tăng accuracy:
1. Adaptive thresholds dựa trên quality
2. Multi-stage matching (centroid -> top-K -> verification)
3. Outlier detection
4. Score normalization
"""
import os, numpy as np
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.db import get_db
from app.utils.match import knn_cosine
from typing import Dict, Optional, List, Tuple

router = APIRouter()

# Thresholds
MODEL = os.getenv("MODEL_NAME", "mobilefacenet-192d")
BASE_TH_CENTROID = float(os.getenv("CENTROID_THRESHOLD", "0.65"))
BASE_TH_KNN = float(os.getenv("KNN_THRESHOLD", "0.60"))
K_CANDIDATES = int(os.getenv("K_CANDIDATES", "10"))  # Tăng từ 5
K_VERIFY = int(os.getenv("K_VERIFY", "3"))  # Top K để verify
MIN_SAMPLES_VOTE = int(os.getenv("MIN_SAMPLES_VOTE", "3"))

# Quality-based threshold adjustment
QUALITY_THRESHOLD_LOW = 0.4
QUALITY_THRESHOLD_HIGH = 0.7


def _l2norm(a: np.ndarray) -> np.ndarray:
    """L2 normalization"""
    return a / (np.linalg.norm(a) + 1e-12)


def _adjust_threshold_by_quality(base_threshold: float, quality: float) -> float:
    """
    Điều chỉnh threshold dựa trên quality của ảnh
    - Quality cao -> threshold cao hơn (strict hơn)
    - Quality thấp -> threshold thấp hơn (relaxed hơn)
    """
    if quality >= QUALITY_THRESHOLD_HIGH:
        # Quality tốt -> có thể strict hơn
        return base_threshold + 0.02
    elif quality <= QUALITY_THRESHOLD_LOW:
        # Quality kém -> phải relaxed
        return base_threshold - 0.03
    else:
        # Quality trung bình
        return base_threshold


def _compute_intra_variance(embeddings: List[np.ndarray]) -> float:
    """
    Tính variance trong một tập embeddings
    Variance thấp = embeddings đồng nhất = người đó stable
    """
    if len(embeddings) < 2:
        return 0.0
    
    X = np.vstack(embeddings)
    centroid = X.mean(axis=0)
    distances = [np.linalg.norm(emb - centroid) for emb in embeddings]
    return float(np.std(distances))


def _detect_outliers(scores: List[float], threshold: float = 2.0) -> List[bool]:
    """
    Detect outliers trong scores bằng Z-score
    Returns: List of booleans (True = outlier)
    """
    if len(scores) < 3:
        return [False] * len(scores)
    
    scores_arr = np.array(scores)
    mean = scores_arr.mean()
    std = scores_arr.std()
    
    if std < 1e-6:
        return [False] * len(scores)
    
    z_scores = np.abs((scores_arr - mean) / std)
    return (z_scores > threshold).tolist()


def _weighted_score(scores: List[float], weights: Optional[List[float]] = None) -> float:
    """
    Tính weighted average score
    Mặc định: weight cao hơn cho scores cao
    """
    if not scores:
        return 0.0
    
    if weights is None:
        # Exponential weighting: scores cao có weight cao hơn
        weights = [np.exp(s) for s in scores]
    
    total_weight = sum(weights)
    if total_weight < 1e-9:
        return float(np.mean(scores))
    
    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    return float(weighted_sum / total_weight)


def _multi_stage_matching(
    query_emb: np.ndarray,
    db: Session,
    quality: float,
    model: str = MODEL
) -> Tuple[Optional[str], float, str]:
    """
    Multi-stage matching:
    1. Centroid matching (fast screening)
    2. Top-K candidate selection
    3. Detailed verification với voting
    4. Outlier detection
    
    Returns: (emp_id, score, method)
    """
    
    # Stage 1: Centroid matching
    qv = "[" + ",".join(str(float(x)) for x in query_emb) + "]"
    
    centroid_rows = db.execute(text("""
        SELECT emp_id, (1 - (centroid <#> CAST(:qv AS vector(192)))) AS score, n_samples
        FROM employee_centroids
        WHERE model = :m
        ORDER BY score DESC
        LIMIT :k
    """), {"qv": qv, "m": model, "k": K_CANDIDATES}).fetchall()
    
    if not centroid_rows:
        return None, 0.0, "empty_db"
    
    # Adjust threshold by quality
    th_centroid = _adjust_threshold_by_quality(BASE_TH_CENTROID, quality)
    th_knn = _adjust_threshold_by_quality(BASE_TH_KNN, quality)
    
    # Check top centroid
    top_centroid = centroid_rows[0]
    if float(top_centroid.score) >= th_centroid:
        # High confidence match
        return top_centroid.emp_id, float(top_centroid.score), "centroid_high_conf"
    
    # Stage 2: Get top K candidates
    candidate_emp_ids = [r.emp_id for r in centroid_rows[:K_VERIFY]]
    
    # Stage 3: Detailed KNN matching cho từng candidate
    best_emp = None
    best_score = 0.0
    best_method = "none"
    
    for emp_id in candidate_emp_ids:
        # Lấy tất cả embeddings của candidate này
        emb_rows = db.execute(text("""
            SELECT embedding, (1 - (embedding <#> CAST(:qv AS vector(192)))) AS score
            FROM embeddings
            WHERE emp_id = :emp_id AND model = :m
            ORDER BY score DESC
        """), {"qv": qv, "emp_id": emp_id, "m": model}).fetchall()
        
        if not emb_rows:
            continue
        
        scores = [float(r.score) for r in emb_rows]
        
        # Outlier detection
        is_outlier = _detect_outliers(scores)
        clean_scores = [s for s, is_out in zip(scores, is_outlier) if not is_out]
        
        if len(clean_scores) < MIN_SAMPLES_VOTE:
            # Không đủ samples sau khi loại outliers
            clean_scores = scores  # Fallback to all scores
        
        # Tính weighted score (ưu tiên scores cao)
        avg_score = _weighted_score(clean_scores[:10])  # Top 10 scores
        
        # Tính consistency (variance thấp = tốt)
        if len(clean_scores) >= 3:
            variance = float(np.std(clean_scores))
            # Bonus cho consistency cao
            consistency_bonus = max(0, 0.05 * (1 - variance))
            avg_score += consistency_bonus
        
        if avg_score > best_score:
            best_score = avg_score
            best_emp = emp_id
            best_method = f"knn_weighted_n{len(clean_scores)}"
    
    # Final check
    if best_score >= th_knn:
        return best_emp, best_score, best_method
    
    return None, best_score, "below_threshold"


@router.post("/v1/recognize")
def recognize(req: dict, db: Session = Depends(get_db)):
    """
    Improved recognition endpoint với multi-stage matching
    """
    # Quality gates
    liveness = float(req.get("liveness", 1.0))
    quality = float(req.get("quality", 0.5))
    
    if liveness < 0.5:
        return {"status": "reject", "result": None, "reason": "liveness_low"}
    
    if quality < 0.3:  # Giảm từ 0.4 để accept thêm
        return {"status": "reject", "result": None, "reason": "quality_too_low"}
    
    # Normalize query embedding
    q = np.asarray(req["embedding"], dtype=np.float64)
    q = _l2norm(q)
    
    # Multi-stage matching
    emp_id, score, method = _multi_stage_matching(q, db, quality, MODEL)
    
    if emp_id is None:
        return {
            "status": "reject",
            "result": None,
            "reason": method,
            "debug": {"best_score": score, "quality": quality}
        }
    
    # Get employee info
    emp = db.execute(
        text("SELECT emp_id, full_name FROM employees WHERE emp_id=:e"),
        {"e": emp_id}
    ).fetchone()
    
    return {
        "status": "ok",
        "result": {
            "emp_id": emp_id,
            "full_name": emp.full_name if emp else emp_id,
            "score": score,
            "threshold": BASE_TH_KNN,
            "decision": "accepted",
            "via": method,
            "quality": quality
        },
        "reason": None
    }


@router.post("/v1/recognize/debug")
def recognize_debug(req: dict, db: Session = Depends(get_db)):
    """
    Debug endpoint để xem chi tiết matching process
    """
    quality = float(req.get("quality", 0.5))
    q = np.asarray(req["embedding"], dtype=np.float64)
    q = _l2norm(q)
    
    qv = "[" + ",".join(str(float(x)) for x in q) + "]"
    
    # Get all centroid scores
    centroid_rows = db.execute(text("""
        SELECT emp_id, (1 - (centroid <#> CAST(:qv AS vector(192)))) AS score, n_samples
        FROM employee_centroids
        WHERE model = :m
        ORDER BY score DESC
        LIMIT 10
    """), {"qv": qv, "m": MODEL}).fetchall()
    
    debug_info = {
        "quality": quality,
        "adjusted_th_centroid": _adjust_threshold_by_quality(BASE_TH_CENTROID, quality),
        "adjusted_th_knn": _adjust_threshold_by_quality(BASE_TH_KNN, quality),
        "centroid_scores": [
            {
                "emp_id": r.emp_id,
                "score": float(r.score),
                "n_samples": r.n_samples
            }
            for r in centroid_rows
        ]
    }
    
    # Get detailed scores for top candidate
    if centroid_rows:
        top_emp = centroid_rows[0].emp_id
        emb_rows = db.execute(text("""
            SELECT (1 - (embedding <#> CAST(:qv AS vector(192)))) AS score
            FROM embeddings
            WHERE emp_id = :emp_id AND model = :m
            ORDER BY score DESC
        """), {"qv": qv, "emp_id": top_emp, "m": MODEL}).fetchall()
        
        scores = [float(r.score) for r in emb_rows]
        is_outlier = _detect_outliers(scores)
        clean_scores = [s for s, is_out in zip(scores, is_outlier) if not is_out]
        
        debug_info["top_candidate_detail"] = {
            "emp_id": top_emp,
            "all_scores": scores,
            "outliers": is_outlier,
            "clean_scores": clean_scores,
            "weighted_score": _weighted_score(clean_scores[:10]) if clean_scores else 0.0,
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores))
        }
    
    return debug_info

