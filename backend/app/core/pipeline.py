from __future__ import annotations
import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from PIL import Image

from app.config import settings
from app.models.registry import ModelRegistry
from app.analysis.facial import FacialAnalyzer
from app.analysis.frequency import FrequencyAnalyzer
from app.analysis.prnu import PRNUAnalyzer
from app.analysis.color import ColorAnalyzer
from app.analysis.compression import CompressionAnalyzer
from app.core.ensemble import WeightedEnsemble
from app.core.fake_type_classifier import FakeTypeClassifier
from app.schemas.response import (
    DetectionResponse, ModelPrediction, PCAVisualization,
)
from app.utils.image import preprocess


# PCA reference cluster centroids (pre-computed from labelled dataset statistics)
# Feature order: [univfd, efficientnet, xception, fft_anomaly, high_freq, prnu_corr, lm_consistency]
_REFERENCE_CENTROIDS: dict[str, list[float]] = {
    "real":            [0.08, 0.07, 0.09, 0.12, 0.62, 0.65, 0.78],
    "gan":             [0.82, 0.70, 0.88, 0.72, 0.38, 0.18, 0.55],
    "face_swap":       [0.75, 0.88, 0.60, 0.40, 0.55, 0.35, 0.30],
    "diffusion":       [0.78, 0.65, 0.52, 0.35, 0.28, 0.12, 0.50],
    "face_reenactment":[0.65, 0.80, 0.55, 0.38, 0.52, 0.40, 0.35],
}

_PCA_MATRIX = np.array([
    [0.45, 0.40, 0.42, 0.35, -0.28, -0.30, -0.33],
    [-0.33, -0.30, -0.35, 0.32, 0.44, 0.40, 0.37],
])


class DetectionPipeline:
    def __init__(self) -> None:
        self._registry   = ModelRegistry.get_instance()
        self._facial     = FacialAnalyzer.get_instance()
        self._frequency  = FrequencyAnalyzer()
        self._prnu       = PRNUAnalyzer()
        self._color      = ColorAnalyzer()
        self._compression= CompressionAnalyzer()
        self._ensemble   = WeightedEnsemble()
        self._classifier = FakeTypeClassifier()
        self._executor   = ThreadPoolExecutor(max_workers=6)

    async def run(self, image: Image.Image) -> DetectionResponse:
        t_start = time.time()
        preprocessed = preprocess(image)
        loop = asyncio.get_event_loop()

        def run_model(name: str):
            det = self._registry.get(name)
            if det is None:
                return None
            return det.predict(preprocessed)

        def run_facial():
            return self._facial.analyze(image)

        def run_frequency():
            return self._frequency.analyze(image)

        def run_prnu():
            return self._prnu.analyze(image)

        def run_color():
            return self._color.analyze(image)

        def run_compression():
            return self._compression.analyze(image)

        # Run everything in parallel
        tasks = [
            loop.run_in_executor(self._executor, run_model, "univfd"),
            loop.run_in_executor(self._executor, run_model, "efficientnet"),
            loop.run_in_executor(self._executor, run_model, "xception"),
            loop.run_in_executor(self._executor, run_facial),
            loop.run_in_executor(self._executor, run_frequency),
            loop.run_in_executor(self._executor, run_prnu),
            loop.run_in_executor(self._executor, run_color),
            loop.run_in_executor(self._executor, run_compression),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        model_outputs = [r for r in results[:3] if r is not None and not isinstance(r, Exception)]
        facial     = results[3] if not isinstance(results[3], Exception) else None
        frequency  = results[4] if not isinstance(results[4], Exception) else self._frequency.analyze(image)
        prnu       = results[5] if not isinstance(results[5], Exception) else self._prnu.analyze(image)
        color      = results[6] if not isinstance(results[6], Exception) else self._color.analyze(image)
        compression= results[7] if not isinstance(results[7], Exception) else self._compression.analyze(image)

        # Initial ensemble pass to get type hint
        fake_type = self._classifier.classify(
            model_outputs, facial, frequency, prnu, color, compression
        )
        ensemble = self._ensemble.combine(
            model_outputs,
            type_hint=fake_type.predicted_type,
            fake_threshold=settings.confidence_fake_threshold,
            real_threshold=settings.confidence_real_threshold,
        )

        pca_viz = self._build_pca(model_outputs, frequency, prnu, facial)

        # Build model prediction objects
        model_preds = []
        for mo in model_outputs:
            det = self._registry.get(mo.model_name)
            model_preds.append(ModelPrediction(
                model_name=mo.model_name,
                fake_probability=mo.fake_prob,
                real_probability=mo.real_prob,
                confidence=abs(mo.fake_prob - 0.5) * 2,
                inference_time_ms=mo.inference_time_ms,
                heatmap_available=True,
            ))

        warnings: list[str] = []
        if len(model_outputs) < 3:
            missing = 3 - len(model_outputs)
            warnings.append(f"{missing} model(s) unavailable (weights not downloaded). Run models/download_weights.py")
        if ensemble.disagreement > 0.30:
            warnings.append("High disagreement between models — result may be uncertain.")

        total_ms = (time.time() - t_start) * 1000

        return DetectionResponse(
            result_id=str(uuid.uuid4()),
            verdict=ensemble.verdict,
            fake_probability=ensemble.fake_probability,
            confidence=ensemble.confidence,
            model_predictions=model_preds,
            ensemble_weights=ensemble.weights_used,
            facial_analysis=facial,
            frequency_analysis=frequency,
            prnu_analysis=prnu,
            color_analysis=color,
            compression_analysis=compression,
            fake_type=fake_type,
            pca_visualization=pca_viz,
            total_inference_time_ms=total_ms,
            image_dimensions=[image.width, image.height],
            warnings=warnings,
        )

    def _build_pca(self, model_outputs, frequency, prnu, facial) -> PCAVisualization:
        by_name = {m.model_name: m.fake_prob for m in model_outputs}
        feat = np.array([
            by_name.get("univfd", 0.5),
            by_name.get("efficientnet", 0.5),
            by_name.get("xception", 0.5),
            frequency.fft_anomaly_score,
            frequency.high_freq_ratio,
            prnu.prnu_correlation,
            facial.landmark_consistency_score if facial and facial.face_detected else 0.5,
        ], dtype=np.float32)

        coords_2d = (_PCA_MATRIX @ feat).tolist()
        centroids_2d = {
            k: (_PCA_MATRIX @ np.array(v, dtype=np.float32)).tolist()
            for k, v in _REFERENCE_CENTROIDS.items()
        }

        return PCAVisualization(
            feature_vector=feat.tolist(),
            reference_centroids=centroids_2d,
            pca_2d_coords=coords_2d,
        )
