from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.fusion import fuse
from app.models.registry import ModelRegistry
from app.preprocessing.face_detection import detect_largest_face
from app.preprocessing.image_transforms import preprocess
from app.schemas.response import DetectionResponse, ModelVote


# Which models get the face crop vs. the full image.
# ViT was trained on uncropped images → use full image.
_FACE_CROP_MODELS = {"efficientnet", "xceptionnet"}
_FULL_IMAGE_MODELS = {"f3net", "vit"}


class DetectionPipeline:
    def __init__(self) -> None:
        self._registry = ModelRegistry.get_instance()

    async def run(self, image: Image.Image) -> DetectionResponse:
        t_start = time.time()

        source_rgb = np.array(image.convert("RGB"))
        face_crop = detect_largest_face(source_rgb)
        face_detected = face_crop is not None

        full_image = image.convert("RGB")
        face_image = Image.fromarray(face_crop) if face_detected else full_image

        # Two preprocess dicts — share work where models agree on input view
        face_input = preprocess(face_image)
        full_input = preprocess(full_image)

        # Dispatch all loaded detectors in parallel via a thread pool.
        loop = asyncio.get_event_loop()
        tasks: dict[str, asyncio.Future] = {}
        for name, det in self._registry.all().items():
            chosen_input = full_input if name in _FULL_IMAGE_MODELS else face_input
            tasks[name] = loop.run_in_executor(None, det.predict, chosen_input)

        outputs = {name: await fut for name, fut in tasks.items()}

        if not outputs:
            raise RuntimeError("No detection models are loaded.")

        # Build model votes
        model_votes: dict[str, ModelVote] = {
            name: ModelVote(
                fake_prob=out.fake_prob,
                real_prob=out.real_prob,
                inference_time_ms=out.inference_time_ms,
            )
            for name, out in outputs.items()
        }

        # Fusion
        scores = {name: out.fake_prob for name, out in outputs.items()}
        fusion = fuse(scores, face_detected=face_detected)

        verdict: str = "fake" if fusion.final_score >= 0.5 else "real"

        explanations = self._build_explanations(
            scores=scores,
            face_detected=face_detected,
            is_uncertain=fusion.is_uncertain,
            final_score=fusion.final_score,
        )

        total_ms = (time.time() - t_start) * 1000

        return DetectionResponse(
            result_id=str(uuid.uuid4()),
            final_score=float(fusion.final_score),
            verdict=verdict,  # type: ignore[arg-type]
            face_detected=face_detected,
            is_uncertain=fusion.is_uncertain,
            model_votes=model_votes,
            fusion_weights=fusion.weights,
            explanations=explanations,
            total_inference_time_ms=total_ms,
        )

    def detect_image(self, image_path: str) -> dict:
        image = Image.open(Path(image_path)).convert("RGB")
        result = asyncio.run(self.run(image))
        return result.model_dump()

    def _build_explanations(
        self,
        scores: dict[str, float],
        face_detected: bool,
        is_uncertain: bool,
        final_score: float,
    ) -> list[str]:
        """
        Produces a short, ordered list of human-readable findings.
        Each line is independent — the frontend renders them as bullets.
        """
        explanations: list[str] = []

        # ── Top-line verdict statement ────────────────────────────────────────
        pct = round(final_score * 100)
        if final_score >= 0.5:
            explanations.append(
                f"Overall, this image is {pct}% likely to be a deepfake. "
                f"Multiple detectors agree on synthesis artifacts."
                if pct >= 70
                else f"Overall, this image leans fake ({pct}% confidence) but signals are mixed."
            )
        else:
            real_pct = 100 - pct
            explanations.append(
                f"Overall, this image looks authentic ({real_pct}% confidence). "
                f"No strong manipulation patterns detected."
                if real_pct >= 70
                else f"Overall, this image leans real ({real_pct}% confidence) but signals are mixed."
            )

        # ── Face-detection context ────────────────────────────────────────────
        if not face_detected:
            explanations.append(
                "No face was detected, so face-cropped models analysed the full image instead."
            )

        # ── Per-model findings, sorted strongest signal first ─────────────────
        eff = scores.get("efficientnet")
        f3 = scores.get("f3net")
        vit = scores.get("vit")
        xcep = scores.get("xceptionnet")

        # FAKE-direction model findings
        if vit is not None and vit >= 0.6:
            explanations.append(
                f"ViT (full-image transformer) is {round(vit * 100)}% confident the image is synthetic — "
                "it spotted learned generation patterns across the whole frame."
            )
        if f3 is not None and f3 >= 0.6:
            explanations.append(
                f"F3Net (frequency analyser) is {round(f3 * 100)}% confident it's fake — "
                "the DCT decomposition revealed unnatural frequency-band energy typical of GAN/diffusion synthesis."
            )
        if eff is not None and eff >= 0.6:
            explanations.append(
                f"EfficientNet (facial-texture CNN) is {round(eff * 100)}% confident it's fake — "
                "it found micro-texture inconsistencies in the facial region (skin pores, edge blending, eye reflections)."
            )
        if xcep is not None and xcep >= 0.6:
            explanations.append(
                f"XceptionNet flagged localised manipulation artifacts ({round(xcep * 100)}% fake)."
            )

        # REAL-direction model findings — only mention strong real signals
        if vit is not None and vit <= 0.2:
            explanations.append(
                f"ViT is {round((1 - vit) * 100)}% confident the image is authentic — "
                "no synthesis patterns visible in the global image structure."
            )
        if f3 is not None and f3 <= 0.2:
            explanations.append(
                f"F3Net is {round((1 - f3) * 100)}% confident the image is real — "
                "frequency-domain energy matches that of natural photographs."
            )

        # ── Uncertainty warning ───────────────────────────────────────────────
        if is_uncertain:
            explanations.append(
                f"⚠ The final score ({final_score:.2f}) falls inside the uncertainty band (0.38–0.62). "
                "Treat this verdict as low confidence — borderline cases benefit from human review."
            )

        return explanations
