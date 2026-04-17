from __future__ import annotations
import os
import torch
import structlog
from app.models.base import BaseDetector
from app.models.univfd import UnivFDDetector
from app.models.efficientnet import EfficientNetDetector
from app.models.xception import XceptionDetector

log = structlog.get_logger()


class ModelRegistry:
    _instance: ModelRegistry | None = None

    def __init__(self) -> None:
        self._detectors: dict[str, BaseDetector] = {}
        self._loaded = False

    @classmethod
    def get_instance(cls) -> ModelRegistry:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_all(self, models_dir: str, device_str: str = "cpu") -> None:
        device = torch.device(
            device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu"
        )
        log.info("loading_models", device=str(device))

        detectors: list[tuple[str, BaseDetector, str]] = [
            ("univfd",       UnivFDDetector(),       os.path.join(models_dir, "univfd",       "fc_weights.pth")),
            ("efficientnet", EfficientNetDetector(), os.path.join(models_dir, "efficientnet", "efficientnet_b4_ff++.pth")),
            ("xception",     XceptionDetector(),     os.path.join(models_dir, "xception",     "xception_ff++.pth")),
            # DIRE disabled: requires 6-channel input (img + diffusion reconstruction)
            # which needs a full diffusion model at inference time.
        ]

        for name, detector, weights_path in detectors:
            if os.path.exists(weights_path):
                try:
                    detector.load(weights_path, device)
                    self._detectors[name] = detector
                    log.info("model_loaded", name=name)
                except Exception as exc:
                    log.warning("model_load_failed", name=name, error=str(exc))
            else:
                log.warning("weights_missing", name=name, path=weights_path)

        self._loaded = True

    def get(self, name: str) -> BaseDetector | None:
        return self._detectors.get(name)

    def all(self) -> dict[str, BaseDetector]:
        return dict(self._detectors)

    def status(self) -> dict:
        return {
            name: {"loaded": True, "model_name": det.model_name}
            for name, det in self._detectors.items()
        }
