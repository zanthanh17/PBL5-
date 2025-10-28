# app/utils/centroid_cache.py
import os, time, threading
from sqlalchemy.orm import Session
from sqlalchemy import text

MODEL = os.getenv("MODEL_NAME", "mobilefacenet-192d")
REFRESH_SEC = int(os.getenv("CENTROID_REFRESH_SEC", "60"))

class CentroidCache:
    def __init__(self):
        self._lock = threading.Lock()
        self._data = {}      # emp_id -> (dim, centroid_list)
        self._last_load = 0.0

    def load_now(self, db: Session):
        rows = db.execute(text("""
            SELECT emp_id, dim, centroid
            FROM employee_centroids
            WHERE model = :m
        """), {"m": MODEL}).fetchall()
        data = {}
        for r in rows:
            # r.centroid (pgvector) => Python list[float]
            data[r.emp_id] = (int(r.dim), list(r.centroid))
        with self._lock:
            self._data = data
            self._last_load = time.time()

    def ensure_fresh(self, db: Session):
        if time.time() - self._last_load > REFRESH_SEC:
            self.load_now(db)

    def get_all(self):
        with self._lock:
            return self._data.copy()

cache = CentroidCache()
