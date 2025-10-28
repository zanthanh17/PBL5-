# app/db.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# DSN: nếu server chạy ngoài Docker, dùng 127.0.0.1:5433 theo map port bạn đã set
# Nếu server chạy trong cùng docker network với DB service 'db', dùng db:5432
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://fa_user:fa_pass@127.0.0.1:5433/fa_db"
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    pool_recycle=3600,
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine, autocommit=False, autoflush=False, future=True
)

# Base for ORM models
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
