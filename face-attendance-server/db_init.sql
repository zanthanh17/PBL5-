CREATE EXTENSION IF NOT EXISTS vector;

-- Bảng nhân sự
CREATE TABLE IF NOT EXISTS employees (
  emp_id      TEXT PRIMARY KEY,
  full_name   TEXT NOT NULL,
  dept        TEXT,
  status      INT DEFAULT 1,
  photo_path  TEXT,
  updated_at  TIMESTAMPTZ DEFAULT now()
);

-- Bảng embeddings (pgvector 192 chiều, cosine)
CREATE TABLE IF NOT EXISTS embeddings (
  id          BIGSERIAL PRIMARY KEY,
  emp_id      TEXT NOT NULL REFERENCES employees(emp_id) ON DELETE CASCADE,
  embedding   vector(192) NOT NULL,
  model       TEXT NOT NULL DEFAULT 'mobilefacenet-192d',
  created_at  TIMESTAMPTZ DEFAULT now()
);

-- HNSW index cho cosine
CREATE INDEX IF NOT EXISTS idx_embeddings_embedding_hnsw
ON embeddings USING hnsw (embedding vector_cosine_ops);

-- Bảng employee centroids (tăng tốc recognition)
CREATE TABLE IF NOT EXISTS employee_centroids (
  emp_id      TEXT NOT NULL,
  model       TEXT NOT NULL,
  dim         INT NOT NULL DEFAULT 192,
  centroid    vector(192) NOT NULL,
  n_samples   INT DEFAULT 0,
  updated_at  TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (emp_id, model),
  FOREIGN KEY (emp_id) REFERENCES employees(emp_id) ON DELETE CASCADE
);

-- HNSW index cho centroid
CREATE INDEX IF NOT EXISTS idx_centroids_centroid_hnsw
ON employee_centroids USING hnsw (centroid vector_cosine_ops);

-- Bảng enroll jobs (cho việc enroll từ xa)
CREATE TABLE IF NOT EXISTS enroll_jobs (
  id          BIGSERIAL PRIMARY KEY,
  emp_id      TEXT NOT NULL REFERENCES employees(emp_id) ON DELETE CASCADE,
  device_id   TEXT NOT NULL,
  status      TEXT DEFAULT 'pending',  -- pending/running/done/failed
  created_at  TIMESTAMPTZ DEFAULT now(),
  started_at  TIMESTAMPTZ,
  finished_at TIMESTAMPTZ,
  notes       TEXT
);

CREATE INDEX IF NOT EXISTS idx_enroll_jobs_status ON enroll_jobs(status, device_id);

-- Bảng chấm công
CREATE TABLE IF NOT EXISTS attendance (
  id          BIGSERIAL PRIMARY KEY,
  emp_id      TEXT,
  ts          TIMESTAMPTZ NOT NULL DEFAULT now(),
  device_id   TEXT,
  score       REAL,
  liveness    REAL,
  image_path  TEXT,
  decision    TEXT,                 -- accepted/reject
  type        TEXT,                 -- checkin/checkout/auto
  extra       JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_attendance_emp_ts ON attendance(emp_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_attendance_ts ON attendance(ts DESC);
