import type { CompressionAnalysis, ColorAnalysis } from '@/types/detection'
import { Card } from '@/components/shared/Card'
import { ScoreBar } from '@/components/shared/ScoreBar'

interface CompressionCardProps {
  compression: CompressionAnalysis
  color: ColorAnalysis
}

export function CompressionCard({ compression, color }: CompressionCardProps) {
  return (
    <Card title="Compression & Colour Forensics">
      <div className="flex flex-col lg:flex-row gap-6">
        <div className="flex-1 space-y-4">
          <div>
            <p className="label mb-2">JPEG / Compression</p>
            <ScoreBar
              label="ELA Spatial Variance"
              value={compression.ela_score}
              invert
              tooltip="Error Level Analysis — uniform ELA (very low) or edited regions (very high) are suspicious"
            />
            <ScoreBar
              label="Block Artifact Score"
              value={compression.block_artifact_score}
              invert
              tooltip="8×8 JPEG block discontinuities from double compression"
            />
          </div>
          <div>
            <p className="label mb-2">RGB Channel Statistics</p>
            <ScoreBar
              label="Inter-channel Correlation"
              value={color.channel_correlation_score}
              invert
              tooltip="Unnaturally high R-G-B correlations suggest synthetic generation"
            />
            <ScoreBar
              label="Histogram Uniformity"
              value={color.histogram_uniformity}
              tooltip="Real photos have rich histogram variance"
            />
          </div>
        </div>

        {compression.ela_image_b64 && (
          <div className="flex-shrink-0">
            <p className="label mb-2">ELA Visualisation</p>
            <img
              src={`data:image/png;base64,${compression.ela_image_b64}`}
              alt="Error Level Analysis"
              className="w-40 h-40 rounded-lg object-cover border border-gray-700"
            />
          </div>
        )}
      </div>
    </Card>
  )
}
