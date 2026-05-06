from __future__ import annotations
from pathlib import Path
import torch
import structlog
from app.config.settings import MODEL_PATHS
from app.models.base import BaseDetector
from app.models.univfd import UnivFDDetector
from app.models.efficientnet import EfficientNetDetector

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

        repo_root = Path(__file__).resolve().parents[3]

        def resolve_path(path_str: str) -> Path:
            path = Path(path_str)
            return path if path.is_absolute() else (repo_root / path).resolve()

        detectors: list[tuple[str, BaseDetector, str]] = [
            ("univfd", UnivFDDetector(), MODEL_PATHS["univfd"]),
            ("efficientnet", EfficientNetDetector(), MODEL_PATHS["efficientnet"]),
        ]

        for name, detector, weights_path in detectors:
            resolved_path = resolve_path(weights_path)
            if resolved_path.exists():
                try:
                    detector.load(str(resolved_path), device)
                    self._detectors[name] = detector
                    log.info("model_loaded", name=name)
                except Exception as exc:
                    log.warning("model_load_failed", name=name, error=str(exc))
            else:
                print(f"Warning: missing weights for {name}: {resolved_path}")
                log.warning("weights_missing", name=name, path=str(resolved_path))

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
