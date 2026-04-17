from __future__ import annotations
import io
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image

from app.explainability.overlay import heatmap_to_overlay, ensemble_heatmap
from app.models.registry import ModelRegistry
from app.storage.result_cache import ResultCache
from app.utils.image import preprocess

router = APIRouter()


@router.get("/heatmap/{result_id}/{model_name}")
async def get_heatmap(result_id: str, model_name: str):
    result = ResultCache.get_instance().get(result_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found or expired.")

    registry = ModelRegistry.get_instance()
    allowed = {"univfd", "efficientnet", "freqnet", "dire", "ensemble"}
    if model_name not in allowed:
        raise HTTPException(status_code=400, detail=f"model_name must be one of {allowed}")

    # We need to re-run to generate heatmap (image not stored; regenerate from stored dims)
    # For now return a 404 with explanation if image not re-accessible.
    # In production, store the image bytes in the cache.
    # Here we check for the stored image in the cache extension.
    cache = ResultCache.get_instance()
    image_bytes = getattr(cache, "_images", {}).get(result_id)
    if image_bytes is None:
        raise HTTPException(
            status_code=404,
            detail="Heatmap image data not available. Re-submit the image.",
        )

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    preprocessed = preprocess(image)

    if model_name == "ensemble":
        heatmaps = []
        weights  = []
        for name, weight in result.ensemble_weights.items():
            det = registry.get(name)
            if det is None:
                continue
            hm = det.get_heatmap(preprocessed)
            if hm is not None:
                heatmaps.append(hm)
                weights.append(weight)
        heatmap = ensemble_heatmap(heatmaps, weights)
    else:
        det = registry.get(model_name)
        if det is None:
            raise HTTPException(status_code=503, detail=f"Model '{model_name}' not loaded.")
        heatmap = det.get_heatmap(preprocessed)
        if heatmap is None:
            raise HTTPException(status_code=503, detail=f"Model '{model_name}' does not support heatmaps.")

    png_bytes = heatmap_to_overlay(image, heatmap)
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/png")
