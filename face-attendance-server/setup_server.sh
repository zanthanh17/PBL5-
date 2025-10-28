#!/bin/bash
# Setup script for Face Attendance Server
# Run with: bash setup_server.sh

set -e

echo "=========================================="
echo "Face Attendance Server Setup"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
NC='\033[0m'

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "Docker not found. Installing..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    echo "Please log out and log back in for Docker permissions"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose not found. Installing..."
    sudo apt update
    sudo apt install -y docker-compose
fi

# Start database
echo -e "\n${GREEN}[1/4] Starting PostgreSQL with pgvector...${NC}"
docker-compose up -d db

# Wait for database
echo -e "\n${GREEN}[2/4] Waiting for database to be ready...${NC}"
sleep 10

# Initialize database schema
echo -e "\n${GREEN}[3/4] Initializing database schema...${NC}"
docker exec -i fa_db psql -U fa_user -d fa_db < db_init.sql

echo -e "\n${GREEN}[4/4] Installing Python dependencies...${NC}"
python3 -m pip install -r requirements.txt

echo -e "\n${GREEN}✓ Setup complete!${NC}"
echo ""
echo "Database is running on localhost:5433"
echo ""
echo "Next steps:"
echo "1. Start server: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
echo "2. Access API docs: http://localhost:8000/docs"
echo "3. Access dashboard: http://localhost:8000/"
echo ""
echo "To stop database: docker-compose down"

