# app/utils/centroid_cache.py
import os, time, threading
import numpy as np
from typing import Dict, Tuple, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text

MODEL = os.getenv("MODEL_NAME", "mobilefacenet-192d")
REFRESH_SEC = int(os.getenv("CENTROID_REFRESH_SEC", "60"))

class CentroidCache:
    def __init__(self):
        self._lock = threading.Lock()
        self._data = {}      # emp_id -> (dim, centroid_list) - old format
        self._centroids = {}  # emp_id -> np.array (192d) - PHASE 2: numpy arrays
        self._emp_names = {}  # emp_id -> full_name - PHASE 2: cache names
        self._last_load = 0.0

    def _coerce_centroid_from_db(self, v) -> np.ndarray:
        """
        Convert centroid từ DB (có thể là string, list, hoặc numpy array) -> numpy array
        
        Args:
            v: Centroid từ database (pgvector)
            
        Returns:
            numpy array (192d, float32)
        """
        if isinstance(v, np.ndarray):
            arr = v.astype(np.float32, copy=False)
        elif isinstance(v, (list, tuple)):
            arr = np.asarray(v, dtype=np.float32)
        elif isinstance(v, str):
            # Parse string dạng "[1.0, 2.0, ...]"
            s = v.strip()
            if s.startswith("[") and s.endswith("]"):
                s = s[1:-1]
            parts = [p.strip() for p in s.split(",") if p.strip() != ""]
            arr = np.asarray([float(p) for p in parts], dtype=np.float32)
        else:
            # Try to convert directly
            try:
                arr = np.asarray(v, dtype=np.float32)
            except Exception:
                raise ValueError(f"Unsupported centroid type from DB: {type(v)}")
        
        # Ensure 192 dimensions
        if arr.shape[0] != 192:
            fixed = np.zeros(192, dtype=np.float32)
            n = min(192, arr.shape[0])
            fixed[:n] = arr[:n]
            arr = fixed
        
        return arr
    
    def load_now(self, db: Session):
        # PHASE 2 OPTIMIZATION: Load centroids as numpy arrays + cache names
        rows = db.execute(text("""
            SELECT ec.emp_id, ec.dim, ec.centroid, e.full_name
            FROM employee_centroids ec
            LEFT JOIN employees e ON e.emp_id = ec.emp_id
            WHERE ec.model = :m
        """), {"m": MODEL}).fetchall()
        
        data = {}
        centroids = {}
        names = {}
        
        for r in rows:
            try:
                # Parse centroid từ DB (có thể là string, list, hoặc array)
                centroid_arr = self._coerce_centroid_from_db(r.centroid)
                
                # Normalize centroid (cần cho cosine similarity)
                centroid_norm = np.linalg.norm(centroid_arr)
                if centroid_norm > 1e-9:
                    centroid_arr = centroid_arr / centroid_norm
                else:
                    # Skip zero vector
                    import logging
                    logging.warning(f"Skipping zero centroid for {r.emp_id}")
                    continue
                
                centroid_list = centroid_arr.tolist()
                
                # Old format (backward compatible)
                data[r.emp_id] = (int(r.dim), centroid_list)
                # PHASE 2: Numpy arrays for fast matching (đã normalized)
                centroids[r.emp_id] = centroid_arr
                names[r.emp_id] = r.full_name if r.full_name else r.emp_id
            except Exception as e:
                # Skip invalid centroids
                import logging
                logging.warning(f"Error loading centroid for {r.emp_id}: {e}")
                continue
        
        with self._lock:
            self._data = data
            self._centroids = centroids
            self._emp_names = names
            self._last_load = time.time()

    def ensure_fresh(self, db: Session):
        if time.time() - self._last_load > REFRESH_SEC:
            self.load_now(db)

    def get_all(self):
        with self._lock:
            return self._data.copy()
    
    # PHASE 2 OPTIMIZATION: Fast in-memory matching
    def match(self, query_vec: np.ndarray, threshold: float) -> Optional[Tuple[str, str, float]]:
        """
        Fast in-memory centroid matching (không cần query DB)
        
        Args:
            query_vec: Query embedding vector (192d, normalized)
            threshold: Minimum cosine similarity threshold
            
        Returns:
            (emp_id, full_name, score) hoặc None nếu không match
        """
        # Normalize query vector
        query_vec = query_vec.astype(np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm < 1e-9:
            return None
        query_vec = query_vec / query_norm
        
        best_emp = None
        best_score = -1.0
        
        with self._lock:
            for emp_id, centroid in self._centroids.items():
                # Cosine similarity: dot product (cả 2 đã normalized)
                score = float(np.dot(query_vec, centroid))
                if score > best_score:
                    best_score = score
                    best_emp = emp_id
        
        if best_score >= threshold and best_emp:
            full_name = self._emp_names.get(best_emp, best_emp)
            return (best_emp, full_name, best_score)
        
        return None

cache = CentroidCache()
