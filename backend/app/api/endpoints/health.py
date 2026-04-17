from fastapi import APIRouter
from app.models.registry import ModelRegistry

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/models/status")
async def models_status():
    registry = ModelRegistry.get_instance()
    return {"models": registry.status()}
