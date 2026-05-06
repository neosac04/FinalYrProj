from __future__ import annotations
from typing import Literal
from pydantic import BaseModel


class DetectionResponse(BaseModel):
    result_id: str
    final_score: float
    verdict: Literal["real", "fake"]
    univfd_score: float
    efficientnet_score: float | None
    face_detected: bool
    explanations: list[str]
    total_inference_time_ms: float
