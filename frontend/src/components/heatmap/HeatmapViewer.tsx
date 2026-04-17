import { useEffect, useRef, useState } from 'react'
import { getHeatmapUrl } from '@/api/detection'
import { Spinner } from '@/components/shared/Spinner'
import clsx from 'clsx'

const MODELS = [
  { key: 'ensemble',    label: 'Ensemble' },
  { key: 'univfd',      label: 'UnivFD' },
  { key: 'efficientnet',label: 'EfficientNet-B4' },
  { key: 'xception',    label: 'Xception' },
]

interface HeatmapViewerProps {
  resultId: string
  originalUrl: string
  landmarkPoints?: number[][]
}

export function HeatmapViewer({ resultId, originalUrl, landmarkPoints }: HeatmapViewerProps) {
  const [activeModel, setActiveModel] = useState('ensemble')
  const [opacity, setOpacity] = useState(0.5)
  const [heatmapUrl, setHeatmapUrl] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    setLoading(true)
    setHeatmapUrl(null)
    const url = getHeatmapUrl(resultId, activeModel)
    const img = new Image()
    img.onload = () => { setHeatmapUrl(url); setLoading(false) }
    img.onerror = () => setLoading(false)
    img.src = url
  }, [resultId, activeModel])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const orig = new Image()
    orig.crossOrigin = 'anonymous'
    orig.onload = () => {
      canvas.width = orig.naturalWidth
      canvas.height = orig.naturalHeight
      ctx.drawImage(orig, 0, 0)

      if (heatmapUrl) {
        const hm = new Image()
        hm.crossOrigin = 'anonymous'
        hm.onload = () => {
          ctx.globalAlpha = opacity
          ctx.drawImage(hm, 0, 0, canvas.width, canvas.height)
          ctx.globalAlpha = 1

          // Draw landmark dots
          if (landmarkPoints && landmarkPoints.length > 0) {
            ctx.fillStyle = 'rgba(0,255,128,0.85)'
            for (const [x, y] of landmarkPoints) {
              ctx.beginPath()
              ctx.arc(x, y, 2, 0, Math.PI * 2)
              ctx.fill()
            }
          }
        }
        hm.src = heatmapUrl
      }
    }
    orig.src = originalUrl
  }, [originalUrl, heatmapUrl, opacity, landmarkPoints])

  return (
    <div className="card">
      <h3 className="section-title">Forgery Heatmap</h3>
      <p className="text-xs text-gray-500 mb-4">
        Red = high suspicion region · Blue = authentic region · Green dots = facial landmarks
      </p>

      {/* Model tabs */}
      <div className="flex gap-2 mb-4 flex-wrap">
        {MODELS.map((m) => (
          <button
            key={m.key}
            onClick={() => setActiveModel(m.key)}
            className={clsx(
              'px-3 py-1.5 rounded-lg text-sm font-medium transition-colors',
              activeModel === m.key
                ? 'bg-brand-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700',
            )}
          >
            {m.label}
          </button>
        ))}
      </div>

      {/* Opacity slider */}
      <div className="flex items-center gap-3 mb-4">
        <span className="text-xs text-gray-500 w-16">Overlay</span>
        <input
          type="range" min={0} max={1} step={0.05}
          value={opacity}
          onChange={(e) => setOpacity(Number(e.target.value))}
          className="flex-1 accent-brand-500"
        />
        <span className="text-xs text-gray-400 w-10 text-right">{Math.round(opacity * 100)}%</span>
      </div>

      {/* Canvas */}
      <div className="relative rounded-xl overflow-hidden bg-gray-800 min-h-48 flex items-center justify-center">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-900/60 z-10">
            <Spinner className="w-8 h-8" />
          </div>
        )}
        <canvas ref={canvasRef} className="w-full h-auto rounded-xl" />
      </div>

      {/* Legend */}
      <div className="flex items-center gap-6 mt-3 justify-center">
        {[
          { color: 'bg-blue-500', label: 'Authentic' },
          { color: 'bg-green-400', label: 'Neutral' },
          { color: 'bg-yellow-400', label: 'Suspicious' },
          { color: 'bg-red-500', label: 'High suspicion' },
        ].map(({ color, label }) => (
          <div key={label} className="flex items-center gap-1.5">
            <div className={`w-3 h-3 rounded-sm ${color}`} />
            <span className="text-xs text-gray-500">{label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
