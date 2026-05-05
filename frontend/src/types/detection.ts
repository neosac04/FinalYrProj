export interface ModelPrediction {
  model_name: string
  fake_probability: number
  real_probability: number
  confidence: number
  inference_time_ms: number
  heatmap_available: boolean
}

export interface FacialAnalysis {
  face_detected: boolean
  face_count: number
  landmark_consistency_score: number
  eye_reflection_symmetry: number
  iris_regularity_score: number
  facial_geometry_score: number
  blending_boundary_score: number
  landmark_points: number[][]
}

export interface FrequencyAnalysis {
  fft_anomaly_score: number
  dct_anomaly_score: number
  high_freq_ratio: number
  spectral_entropy: number
  fft_image_b64: string
}

export interface PRNUAnalysis {
  prnu_correlation: number
  noise_pattern_score: number
  prnu_map_b64: string
}

export interface ColorAnalysis {
  r_mean: number
  r_std: number
  g_mean: number
  g_std: number
  b_mean: number
  b_std: number
  channel_correlation_score: number
  histogram_uniformity: number
}

export interface CompressionAnalysis {
  ela_score: number
  block_artifact_score: number
  ela_image_b64: string
}

export type FakeType = 'gan' | 'face_swap' | 'face_reenactment' | 'diffusion' | 'photoshop' | 'real'

export interface FakeTypeClassification {
  predicted_type: FakeType
  type_probabilities: Record<string, number>
  confidence: number
  reasoning: string[]
}

export interface PCAVisualization {
  feature_vector: number[]
  reference_centroids: Record<string, number[]>
  pca_2d_coords: number[]
}

export type Verdict = 'REAL' | 'FAKE' | 'UNCERTAIN'

export interface DetectionResponse {
  result_id: string
  verdict: Verdict
  fake_probability: number
  confidence: number
  model_predictions: ModelPrediction[]
  ensemble_weights: Record<string, number>
  used_full_image_for_face_models: boolean
  facial_analysis: FacialAnalysis | null
  frequency_analysis: FrequencyAnalysis
  prnu_analysis: PRNUAnalysis
  color_analysis: ColorAnalysis
  compression_analysis: CompressionAnalysis
  fake_type: FakeTypeClassification
  pca_visualization: PCAVisualization
  total_inference_time_ms: number
  image_dimensions: number[]
  warnings: string[]
}
