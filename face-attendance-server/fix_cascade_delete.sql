-- Script để fix CASCADE DELETE cho database hiện tại
-- Chạy script này nếu xóa employee không xóa embeddings

-- 1. Drop constraints cũ (nếu có)
ALTER TABLE embeddings DROP CONSTRAINT IF EXISTS embeddings_emp_id_fkey;
ALTER TABLE employee_centroids DROP CONSTRAINT IF EXISTS employee_centroids_emp_id_fkey;
ALTER TABLE enroll_jobs DROP CONSTRAINT IF EXISTS enroll_jobs_emp_id_fkey;
ALTER TABLE attendance DROP CONSTRAINT IF EXISTS attendance_emp_id_fkey;

-- 2. Thêm lại constraints với CASCADE DELETE
ALTER TABLE embeddings 
  ADD CONSTRAINT embeddings_emp_id_fkey 
  FOREIGN KEY (emp_id) REFERENCES employees(emp_id) ON DELETE CASCADE;

ALTER TABLE employee_centroids 
  ADD CONSTRAINT employee_centroids_emp_id_fkey 
  FOREIGN KEY (emp_id) REFERENCES employees(emp_id) ON DELETE CASCADE;

ALTER TABLE enroll_jobs 
  ADD CONSTRAINT enroll_jobs_emp_id_fkey 
  FOREIGN KEY (emp_id) REFERENCES employees(emp_id) ON DELETE CASCADE;

-- 3. Thêm constraint cho attendance (QUAN TRỌNG!)
ALTER TABLE attendance 
  ADD CONSTRAINT attendance_emp_id_fkey 
  FOREIGN KEY (emp_id) REFERENCES employees(emp_id) ON DELETE CASCADE;

-- 3. Verify
SELECT 
  conname AS constraint_name,
  conrelid::regclass AS table_name,
  confrelid::regclass AS referenced_table,
  confdeltype AS delete_action
FROM pg_constraint
WHERE confrelid = 'employees'::regclass
  AND contype = 'f';

-- Delete action codes:
-- 'a' = NO ACTION
-- 'r' = RESTRICT  
-- 'c' = CASCADE
-- 'n' = SET NULL
-- 'd' = SET DEFAULT

-- Kết quả mong đợi: delete_action = 'c' (CASCADE)

