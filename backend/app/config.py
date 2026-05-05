from __future__ import annotations

import os

from pydantic_settings import BaseSettings, SettingsConfigDict


def _env_path(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value else default


MODEL_PATHS = {
    "univfd": _env_path("UNIVFD_MODEL_PATH", "models/univfd.pth"),
    "efficientnet": _env_path("EFFICIENTNET_MODEL_PATH", "models/efficientnet.pth"),
    "xception": _env_path("XCEPTION_MODEL_PATH", "models/xception.pth"),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    models_dir: str = "./models"
    device: str = "cpu"
    max_file_size_mb: int = 10
    confidence_fake_threshold: float = 0.65
    confidence_real_threshold: float = 0.35


settings = Settings()
