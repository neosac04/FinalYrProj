from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    models_dir: str = "./models"
    device: str = "cpu"
    max_file_size_mb: int = 10
    confidence_fake_threshold: float = 0.65
    confidence_real_threshold: float = 0.35


settings = Settings()
