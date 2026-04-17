import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowLeft, Zap } from 'lucide-react'
import { useDetectionStore } from '@/store/detectionStore'
import { VerdictBanner } from '@/components/verdict/VerdictBanner'
import { HeatmapViewer } from '@/components/heatmap/HeatmapViewer'
import { ModelVoteTable } from '@/components/ensemble/ModelVoteTable'
import { FakeTypeCard } from '@/components/analysis/FakeTypeCard'
import { FacialAnalysisCard } from '@/components/analysis/FacialAnalysisCard'
import { FrequencyCard } from '@/components/analysis/FrequencyCard'
import { PRNUCard } from '@/components/analysis/PRNUCard'
import { CompressionCard } from '@/components/analysis/CompressionCard'
import { PCAPlot } from '@/components/visualization/PCAPlot'

export function ResultsPage() {
  const navigate = useNavigate()
  const { result, previewUrl, reset } = useDetectionStore()

  useEffect(() => {
    if (!result) navigate('/')
  }, [result, navigate])

  if (!result) return null

  const handleNewImage = () => {
    reset()
    navigate('/')
  }

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b border-gray-800 px-6 py-4 sticky top-0 z-20 bg-gray-950/90 backdrop-blur">
        <div className="max-w-6xl mx-auto flex items-center gap-4">
          <button
            onClick={handleNewImage}
            className="flex items-center gap-2 text-gray-400 hover:text-gray-200 transition-colors text-sm"
          >
            <ArrowLeft size={16} />
            New Image
          </button>
          <div className="flex items-center gap-2 ml-2">
            <div className="p-1.5 bg-brand-600 rounded-lg">
              <Zap size={14} className="text-white" />
            </div>
            <span className="font-semibold text-gray-200">Deepfake Detector</span>
          </div>
          <div className="ml-auto text-xs text-gray-600">
            {result.image_dimensions[0]}×{result.image_dimensions[1]}px · {Math.round(result.total_inference_time_ms)}ms
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-6xl mx-auto w-full px-4 py-8 space-y-6">
        {/* Verdict banner — full width */}
        <VerdictBanner
          verdict={result.verdict}
          fakeProb={result.fake_probability}
          confidence={result.confidence}
          fakeType={result.fake_type.predicted_type}
          fakeTypeConf={result.fake_type.confidence}
          inferenceMs={result.total_inference_time_ms}
          warnings={result.warnings}
        />

        {/* 2-column layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left column: heatmap + PCA */}
          <div className="space-y-6">
            {previewUrl && (
              <HeatmapViewer
                resultId={result.result_id}
                originalUrl={previewUrl}
                landmarkPoints={result.facial_analysis?.landmark_points ?? undefined}
              />
            )}
            <PCAPlot data={result.pca_visualization} />
          </div>

          {/* Right column: fake type + ensemble */}
          <div className="space-y-6">
            <FakeTypeCard data={result.fake_type} />
            <ModelVoteTable
              predictions={result.model_predictions}
              weights={result.ensemble_weights}
            />
          </div>
        </div>

        {/* Full-width analysis section */}
        <div className="space-y-6">
          <h2 className="text-xl font-semibold text-gray-200 border-b border-gray-800 pb-3">
            Detailed Forensic Analysis
          </h2>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <FacialAnalysisCard data={result.facial_analysis} />
            <FrequencyCard data={result.frequency_analysis} />
            <PRNUCard data={result.prnu_analysis} />
            <CompressionCard
              compression={result.compression_analysis}
              color={result.color_analysis}
            />
          </div>
        </div>
      </main>

      <footer className="border-t border-gray-800 px-6 py-4 text-center text-xs text-gray-700">
        Deepfake Detector · Multi-model ensemble: UnivFD · EfficientNet-B4 · Xception · DistilDIRE
      </footer>
    </div>
  )
}
