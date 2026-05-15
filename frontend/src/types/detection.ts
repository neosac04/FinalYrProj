// Matches backend/app/schemas/response.py

export type Verdict = 'real' | 'fake'

export interface ModelVote {
  fake_prob: number
  real_prob: number
  inference_time_ms: number
}

export interface DetectionResponse {
  result_id: string
  final_score: number          // 0-1 fused fake probability
  verdict: Verdict
  face_detected: boolean
  is_uncertain: boolean
  model_votes: Record<string, ModelVote>     // { efficientnet, vit, f3net }
  fusion_weights: Record<string, number>     // post-normalisation
  explanations: string[]
  total_inference_time_ms: number
}
