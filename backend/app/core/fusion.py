"""
Adaptive confidence-weighted fusion for the model trio.

Base weights vary by whether a face was detected (F3Net carries more weight when no
face is found, since it works on the full image). Each model gets a per-prediction
confidence bonus proportional to how far its score sits from 0.5.
"""
from __future__ import annotations

from dataclasses import dataclass


# Per-model AUC on Validation/ (calibration set, 400 fake + 400 real):
#   - ViT (dima806):   AUC 0.999  → dominant signal
#   - F3Net (retrained head):  AUC 0.958  → strong frequency-domain second opinion
#   - EfficientNet:    AUC 0.764  → calibrated facial-texture detector
#   - XceptionNet:     AUC 0.475  → disabled in registry
# Weights scaled with (AUC − 0.5), then renormalised within the face / no-face
# routing groups. EfficientNet is downweighted heavily when no face is present
# because its face-crop assumption breaks down.
_BASE_WEIGHTS_FACE = {
    "vit": 0.50,
    "f3net": 0.35,
    "efficientnet": 0.15,
    "xceptionnet": 0.0,
}

_BASE_WEIGHTS_NO_FACE = {
    "vit": 0.55,
    "f3net": 0.40,
    "efficientnet": 0.05,
    "xceptionnet": 0.0,
}

# Uncertainty band — outside this range the local verdict is considered confident.
UNCERTAINTY_LOW = 0.38
UNCERTAINTY_HIGH = 0.62

# Per-model confidence bonus scale
_CONFIDENCE_BONUS = 0.1


@dataclass
class FusionResult:
    final_score: float
    weights: dict[str, float]   # post-normalisation weights actually used
    is_uncertain: bool


def fuse(
    model_scores: dict[str, float],
    face_detected: bool,
) -> FusionResult:
    """
    Confidence-weighted fusion of fake-probability scores.

    `model_scores` maps model_name → fake_prob in [0, 1]. Missing models are
    silently dropped — fusion re-normalises across whichever survived.
    """
    base = _BASE_WEIGHTS_FACE if face_detected else _BASE_WEIGHTS_NO_FACE

    adjusted: dict[str, float] = {}
    for name, score in model_scores.items():
        if name not in base:
            continue
        bonus = abs(score - 0.5) * _CONFIDENCE_BONUS
        adjusted[name] = base[name] + bonus

    total = sum(adjusted.values())
    if total <= 0 or not adjusted:
        return FusionResult(final_score=0.5, weights={}, is_uncertain=True)

    weights = {name: w / total for name, w in adjusted.items()}
    final_score = sum(weights[name] * model_scores[name] for name in weights)
    is_uncertain = UNCERTAINTY_LOW <= final_score <= UNCERTAINTY_HIGH

    return FusionResult(
        final_score=float(final_score),
        weights=weights,
        is_uncertain=is_uncertain,
    )
