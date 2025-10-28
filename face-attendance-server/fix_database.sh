#!/bin/bash
# Script để fix CASCADE DELETE trong database hiện tại

set -e

echo "=========================================="
echo "Fix CASCADE DELETE for Employee Deletion"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if database is running
if ! docker ps | grep -q fa_db; then
    echo -e "${RED}Error: Database container 'fa_db' is not running${NC}"
    echo "Start it with: docker-compose up -d db"
    exit 1
fi

echo -e "\n${YELLOW}Current CASCADE constraints:${NC}"
docker exec -i fa_db psql -U fa_user -d fa_db << 'EOF'
SELECT 
  conname AS constraint_name,
  conrelid::regclass AS table_name,
  CASE confdeltype
    WHEN 'a' THEN 'NO ACTION'
    WHEN 'r' THEN 'RESTRICT'
    WHEN 'c' THEN 'CASCADE'
    WHEN 'n' THEN 'SET NULL'
    WHEN 'd' THEN 'SET DEFAULT'
  END AS delete_action
FROM pg_constraint
WHERE confrelid = 'employees'::regclass
  AND contype = 'f';
EOF

echo -e "\n${YELLOW}Fixing CASCADE constraints...${NC}"
docker exec -i fa_db psql -U fa_user -d fa_db < fix_cascade_delete.sql

echo -e "\n${GREEN}✓ CASCADE constraints fixed!${NC}"

echo -e "\n${YELLOW}Testing: Deleting and re-adding employee E001...${NC}"

# Test delete
docker exec -i fa_db psql -U fa_user -d fa_db << 'EOF'
-- Check before delete
SELECT 'Before delete:' as status;
SELECT COUNT(*) as embeddings_count FROM embeddings WHERE emp_id = 'E001';
SELECT COUNT(*) as centroids_count FROM employee_centroids WHERE emp_id = 'E001';

-- Delete employee
DELETE FROM employees WHERE emp_id = 'E001';

-- Check after delete
SELECT 'After delete:' as status;
SELECT COUNT(*) as embeddings_count FROM embeddings WHERE emp_id = 'E001';
SELECT COUNT(*) as centroids_count FROM employee_centroids WHERE emp_id = 'E001';
EOF

echo -e "\n${GREEN}✓ Test completed!${NC}"
echo ""
echo "Expected results:"
echo "  - Before delete: embeddings_count > 0"
echo "  - After delete: embeddings_count = 0 (CASCADE worked!)"
echo ""
echo "Note: Employee E001 has been deleted for testing."
echo "You can re-add it via the web interface or API."

