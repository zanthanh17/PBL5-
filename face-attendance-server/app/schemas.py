from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class RecognizeOptions(BaseModel):
    save_event_face: bool = False
    mode: Optional[Literal["auto","checkin","checkout"]] = "auto"

class RecognizeRequest(BaseModel):
    device_id: str
    ts: Optional[str] = None
    embedding: List[float] = Field(..., min_items=192, max_items=192)  # was 128
    liveness: float
    quality: float
    options: Optional[RecognizeOptions] = RecognizeOptions()

class RecognizeResult(BaseModel):
    emp_id: Optional[str]
    full_name: Optional[str]
    score: float
    threshold: float
    decision: Literal["accepted","reject"]

class RecognizeResponse(BaseModel):
    status: Literal["ok","reject","error"]
    result: Optional[RecognizeResult] = None
    reason: Optional[str] = None

class EnrollRequest(BaseModel):
    model: str = "mobilefacenet-192d"  # đổi nhãn cho rõ
    embeddings: List[List[float]]
