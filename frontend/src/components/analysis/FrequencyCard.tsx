import type { FrequencyAnalysis } from '@/types/detection'
import { Card } from '@/components/shared/Card'
import { ScoreBar } from '@/components/shared/ScoreBar'

interface FrequencyCardProps {
  data: FrequencyAnalysis
}

export function FrequencyCard({ data }: FrequencyCardProps) {
  return (
    <Card title="Frequency Domain Analysis">
      <div className="flex flex-col lg:flex-row gap-6">
        <div className="flex-1">
          <p className="text-xs text-gray-500 mb-4">
            GAN upsampling leaves characteristic spectral artifacts.
            Diffusion models suppress high frequencies.
          </p>
          <ScoreBar
            label="FFT Anomaly Score"
            value={data.fft_anomaly_score}
            invert
            tooltip="Elevated energy at GAN artifact frequencies"
          />
          <ScoreBar
            label="DCT Anomaly Score"
            value={data.dct_anomaly_score}
            invert
            tooltip="Abnormal DC dominance (over-smoothing)"
          />
          <ScoreBar
            label="High-Freq Content"
            value={data.high_freq_ratio}
            tooltip="Real photos have rich high-freq detail; diffusion smooths it"
          />
          <ScoreBar
            label="Spectral Entropy"
            value={data.spectral_entropy}
            tooltip="Complexity of the frequency spectrum — synthetic images often lower"
          />
        </div>

        {data.fft_image_b64 && (
          <div className="flex-shrink-0">
            <p className="label mb-2">FFT Magnitude Spectrum</p>
            <img
              src={`data:image/png;base64,${data.fft_image_b64}`}
              alt="FFT spectrum"
              className="w-40 h-40 rounded-lg object-cover border border-gray-700"
            />
          </div>
        )}
      </div>
    </Card>
  )
}
