from __future__ import annotations
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from app.models.registry import ModelRegistry
from app.preprocessing.face_detection import detect_largest_face
from app.schemas.response import DetectionResponse
from app.preprocessing.image_transforms import preprocess


class DetectionPipeline:
    def __init__(self) -> None:
        self._registry = ModelRegistry.get_instance()

    async def run(self, image: Image.Image) -> DetectionResponse:
        t_start = time.time()
        source_rgb = np.array(image.convert("RGB"))
        face_crop = detect_largest_face(source_rgb)
        face_detected = face_crop is not None
        used_full_image = not face_detected

        full_image = image.convert("RGB")
        face_image = Image.fromarray(face_crop) if face_detected else full_image
        univfd_input = preprocess(full_image)
        efficientnet_input = preprocess(face_image)

        univfd_det = self._registry.get("univfd")
        efficientnet_det = self._registry.get("efficientnet")
        if univfd_det is None:
            raise RuntimeError("UnivFD model is not loaded.")

        univfd_pred = univfd_det.predict(univfd_input)
        efficientnet_pred = efficientnet_det.predict(efficientnet_input) if efficientnet_det is not None else None

        univfd_score = float(univfd_pred.fake_prob)
        efficientnet_score = float(efficientnet_pred.fake_prob) if efficientnet_pred is not None else None

        if face_detected and efficientnet_score is not None:
            final_score = 0.6 * efficientnet_score + 0.4 * univfd_score
        else:
            final_score = univfd_score

        print("Face detected:", face_detected)
        print("Using fallback full image:", used_full_image)

        verdict = "fake" if final_score >= 0.5 else "real"
        explanations = self._build_explanations(
            univfd_score=univfd_score,
            efficientnet_score=efficientnet_score,
            face_detected=face_detected,
        )

        total_ms = (time.time() - t_start) * 1000

        return DetectionResponse(
            result_id=str(uuid.uuid4()),
            final_score=float(final_score),
            verdict=verdict,
            univfd_score=univfd_score,
            efficientnet_score=efficientnet_score,
            face_detected=face_detected,
            explanations=explanations,
            total_inference_time_ms=total_ms,
        )

    def detect_image(self, image_path: str) -> dict:
        image = Image.open(Path(image_path)).convert("RGB")
        import asyncio

        result = asyncio.run(self.run(image))
        return result.model_dump()

    def _build_explanations(self, univfd_score: float, efficientnet_score: float | None, face_detected: bool) -> list[str]:
        explanations: list[str] = []
        if efficientnet_score is not None and efficientnet_score >= 0.6:
            explanations.append("Facial texture inconsistencies detected")
        if univfd_score >= 0.6:
            explanations.append("Global semantic inconsistencies detected")
        if not face_detected:
            explanations.append("No face detected; EfficientNet fallback used full image")
        if not explanations:
            explanations.append("No strong manipulation indicators detected")
        return explanations

    def debug_test(self, image_path: str):
        result = self.detect_image(image_path)
        print("Face detection status:", result["face_detected"])
        print("UnivFD score:", result["univfd_score"])
        print("EfficientNet score:", result["efficientnet_score"])
        print("Final score:", result["final_score"])
        print("Final verdict:", result["verdict"])
        print("Explanations:", result["explanations"])
        return result
