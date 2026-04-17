from __future__ import annotations
import base64
import io
import cv2
import numpy as np
from PIL import Image


def heatmap_to_overlay(
    original: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> bytes:
    """
    Resize heatmap (H×W float32, range 0-1) to match original image,
    apply JET colormap (blue=real, red=fake), blend at alpha, return PNG bytes.
    """
    w, h = original.size
    hm_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    hm_uint8 = (np.clip(hm_resized, 0, 1) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)           # BGR
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    orig_np = np.array(original.convert("RGB")).astype(np.float32)
    blended = (orig_np * (1 - alpha) + colored_rgb.astype(np.float32) * alpha)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format="PNG")
    return buf.getvalue()


def ensemble_heatmap(heatmaps: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Weighted average of multiple heatmaps (all normalised to same spatial resolution)."""
    if not heatmaps:
        return np.zeros((14, 14), dtype=np.float32)
    total_w = sum(weights) or 1.0
    result = np.zeros_like(heatmaps[0], dtype=np.float32)
    for hm, w in zip(heatmaps, weights):
        result += hm * (w / total_w)
    mn, mx = result.min(), result.max()
    return ((result - mn) / (mx - mn + 1e-8)).astype(np.float32)
