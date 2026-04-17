import type { FacialAnalysis } from '@/types/detection'
import { Card } from '@/components/shared/Card'
import { ScoreBar } from '@/components/shared/ScoreBar'
import { Eye, Users } from 'lucide-react'

interface FacialAnalysisCardProps {
  data: FacialAnalysis | null
}

export function FacialAnalysisCard({ data }: FacialAnalysisCardProps) {
  if (!data) return null

  if (!data.face_detected) {
    return (
      <Card title="Facial Analysis">
        <div className="flex items-center gap-3 text-gray-500">
          <Eye size={20} />
          <p className="text-sm">No face detected — spatial analysis only.</p>
        </div>
      </Card>
    )
  }

  return (
    <Card title="Facial Anatomy Analysis">
      {data.face_count > 1 && (
        <div className="flex items-center gap-2 mb-4 text-sm text-blue-400 bg-blue-500/10 rounded-lg px-3 py-2">
          <Users size={14} />
          {data.face_count} faces detected — analysing most prominent face
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-8">
        <div>
          <ScoreBar
            label="Landmark Symmetry"
            value={data.landmark_consistency_score}
            tooltip="Left/right facial geometry consistency"
          />
          <ScoreBar
            label="Eye Reflection Symmetry"
            value={data.eye_reflection_symmetry}
            tooltip="Specular highlights should mirror between eyes"
          />
          <ScoreBar
            label="Iris Circularity"
            value={data.iris_regularity_score}
            tooltip="GAN irises deviate from perfect circles"
          />
        </div>
        <div>
          <ScoreBar
            label="Facial Geometry"
            value={data.facial_geometry_score}
            tooltip="Golden ratio proportions consistency"
          />
          <ScoreBar
            label="Jaw Boundary Integrity"
            value={data.blending_boundary_score}
            tooltip="Gradient discontinuity along face oval — face swap seam detector"
          />
        </div>
      </div>

      <p className="text-xs text-gray-600 mt-3">
        Higher scores = more authentic anatomical features · powered by MediaPipe FaceMesh (478 landmarks + iris)
      </p>
    </Card>
  )
}
