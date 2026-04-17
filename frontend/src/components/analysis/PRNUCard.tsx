import type { PRNUAnalysis } from '@/types/detection'
import { Card } from '@/components/shared/Card'
import { ScoreBar } from '@/components/shared/ScoreBar'

interface PRNUCardProps {
  data: PRNUAnalysis
}

export function PRNUCard({ data }: PRNUCardProps) {
  return (
    <Card title="Camera Noise Fingerprint (PRNU)">
      <div className="flex flex-col lg:flex-row gap-6">
        <div className="flex-1">
          <p className="text-xs text-gray-500 mb-4">
            Every real camera sensor leaves a unique Photo Response Non-Uniformity fingerprint.
            AI-generated images have no camera fingerprint — its presence confirms authenticity.
          </p>
          <ScoreBar
            label="PRNU Correlation"
            value={data.prnu_correlation}
            tooltip="Camera noise autocorrelation — real photos score higher"
          />
          <ScoreBar
            label="Noise Pattern Regularity"
            value={data.noise_pattern_score}
            tooltip="GAN periodic artifacts in noise domain lower this score"
          />
        </div>

        {data.prnu_map_b64 && (
          <div className="flex-shrink-0">
            <p className="label mb-2">Noise Residual Map</p>
            <img
              src={`data:image/png;base64,${data.prnu_map_b64}`}
              alt="PRNU noise map"
              className="w-40 h-40 rounded-lg object-cover border border-gray-700"
            />
          </div>
        )}
      </div>
    </Card>
  )
}
