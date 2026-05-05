from __future__ import annotations
from typing import Literal
from pydantic import BaseModel


class ModelPrediction(BaseModel):
    model_name: str
    fake_probability: float
    real_probability: float
    confidence: float
    inference_time_ms: float
    heatmap_available: bool


class FacialAnalysis(BaseModel):
    face_detected: bool
    face_count: int
    landmark_consistency_score: float
    eye_reflection_symmetry: float
    iris_regularity_score: float
    facial_geometry_score: float
    blending_boundary_score: float
    landmark_points: list[list[float]]


class FrequencyAnalysis(BaseModel):
    fft_anomaly_score: float
    dct_anomaly_score: float
    high_freq_ratio: float
    spectral_entropy: float
    fft_image_b64: str


class PRNUAnalysis(BaseModel):
    prnu_correlation: float
    noise_pattern_score: float
    prnu_map_b64: str


class ColorAnalysis(BaseModel):
    r_mean: float
    r_std: float
    g_mean: float
    g_std: float
    b_mean: float
    b_std: float
    channel_correlation_score: float
    histogram_uniformity: float


class CompressionAnalysis(BaseModel):
    ela_score: float
    block_artifact_score: float
    ela_image_b64: str


class FakeTypeClassification(BaseModel):
    predicted_type: Literal["gan", "face_swap", "face_reenactment", "diffusion", "photoshop", "real"]
    type_probabilities: dict[str, float]
    confidence: float
    reasoning: list[str]


class PCAVisualization(BaseModel):
    feature_vector: list[float]
    reference_centroids: dict[str, list[float]]
    pca_2d_coords: list[float]


class DetectionResponse(BaseModel):
    result_id: str
    verdict: Literal["REAL", "FAKE", "UNCERTAIN"]
    fake_probability: float
    confidence: float

    model_predictions: list[ModelPrediction]
    ensemble_weights: dict[str, float]
    used_full_image_for_face_models: bool

    facial_analysis: FacialAnalysis | None
    frequency_analysis: FrequencyAnalysis
    prnu_analysis: PRNUAnalysis
    color_analysis: ColorAnalysis
    compression_analysis: CompressionAnalysis

    fake_type: FakeTypeClassification
    pca_visualization: PCAVisualization

    total_inference_time_ms: float
    image_dimensions: list[int]
    warnings: list[str]
