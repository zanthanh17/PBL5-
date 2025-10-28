from sqlalchemy import Column, Integer, Text, DateTime, Float, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from .db import Base

class Employee(Base):
    __tablename__ = "employees"
    emp_id = Column(Text, primary_key=True)
    full_name = Column(Text, nullable=False)
    dept = Column(Text)
    status = Column(Integer, default=1)
    updated_at = Column(DateTime(timezone=True), server_default=func.now())

class Attendance(Base):
    __tablename__ = "attendance"
    id = Column(Integer, primary_key=True, autoincrement=True)
    emp_id = Column(Text)
    ts = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    device_id = Column(Text)
    score = Column(Float)       # đổi sang Float
    liveness = Column(Float)    # đổi sang Float
    image_path = Column(Text)
    decision = Column(Text)
    type = Column(Text)
    extra = Column(JSONB, server_default='{}')
