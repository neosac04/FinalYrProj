import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowLeft, Zap } from 'lucide-react'
import { useDetectionStore } from '@/store/detectionStore'
import { VerdictBanner } from '@/components/verdict/VerdictBanner'
import { ExplanationsCard } from '@/components/verdict/ExplanationsCard'
import { HeatmapViewer } from '@/components/heatmap/HeatmapViewer'
import { ModelVoteTable } from '@/components/ensemble/ModelVoteTable'

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

  const availableModels = Object.keys(result.model_votes)

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
            {Math.round(result.total_inference_time_ms)} ms · {availableModels.length} models
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-6xl mx-auto w-full px-4 py-8 space-y-6">
        {/* Verdict banner — full width */}
        <VerdictBanner
          verdict={result.verdict}
          finalScore={result.final_score}
          isUncertain={result.is_uncertain}
          faceDetected={result.face_detected}
          inferenceMs={result.total_inference_time_ms}
        />

        {/* Two-column layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left: heatmap explanation */}
          <div className="space-y-6">
            {previewUrl && (
              <HeatmapViewer
                resultId={result.result_id}
                originalUrl={previewUrl}
                availableModels={availableModels}
              />
            )}
          </div>

          {/* Right: model votes + text explanations */}
          <div className="space-y-6">
            <ModelVoteTable
              votes={result.model_votes}
              weights={result.fusion_weights}
            />
            <ExplanationsCard explanations={result.explanations} />
          </div>
        </div>
      </main>

      <footer className="border-t border-gray-800 px-6 py-4 text-center text-xs text-gray-700">
        Deepfake Detector · Multi-model ensemble: ViT · F3Net · EfficientNet-B4
      </footer>
    </div>
  )
}
