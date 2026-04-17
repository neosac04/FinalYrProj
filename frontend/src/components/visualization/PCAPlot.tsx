import type { PCAVisualization } from '@/types/detection'
import { Card } from '@/components/shared/Card'
import {
  ScatterChart, Scatter, XAxis, YAxis, ZAxis,
  Tooltip, ResponsiveContainer, ReferenceLine, Cell,
} from 'recharts'

const CENTROID_COLORS: Record<string, string> = {
  real:            '#22c55e',
  gan:             '#a855f7',
  face_swap:       '#f97316',
  diffusion:       '#3b82f6',
  face_reenactment:'#eab308',
}

const CENTROID_LABELS: Record<string, string> = {
  real:            'Real',
  gan:             'GAN',
  face_swap:       'Face Swap',
  diffusion:       'Diffusion',
  face_reenactment:'Reenactment',
}

interface PCAPlotProps {
  data: PCAVisualization
}

interface DataPoint { x: number; y: number; name: string; color: string; size: number }

export function PCAPlot({ data }: PCAPlotProps) {
  const points: DataPoint[] = Object.entries(data.reference_centroids).map(([k, v]) => ({
    x: v[0],
    y: v[1],
    name: CENTROID_LABELS[k] ?? k,
    color: CENTROID_COLORS[k] ?? '#6b7280',
    size: 80,
  }))

  const thisPoint: DataPoint = {
    x: data.pca_2d_coords[0],
    y: data.pca_2d_coords[1],
    name: 'This image',
    color: '#f1f5f9',
    size: 200,
  }

  const allPoints = [...points, thisPoint]

  return (
    <Card title="Feature Space (PCA)">
      <p className="text-xs text-gray-500 mb-4">
        Position of this image relative to known cluster centroids in the 8-dim feature space,
        projected to 2D via PCA. Proximity to a cluster indicates similarity to that forgery type.
      </p>

      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 10, right: 10, bottom: 0, left: -20 }}>
            <XAxis type="number" dataKey="x" tick={{ fill: '#4b5563', fontSize: 10 }} />
            <YAxis type="number" dataKey="y" tick={{ fill: '#4b5563', fontSize: 10 }} />
            <ZAxis dataKey="size" range={[60, 200]} />
            <ReferenceLine x={0} stroke="#374151" strokeDasharray="3 3" />
            <ReferenceLine y={0} stroke="#374151" strokeDasharray="3 3" />
            <Tooltip
              cursor={false}
              contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8 }}
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              formatter={(_: unknown, __: unknown, item: any) => {
                const pt: DataPoint = item?.payload ?? { x: 0, y: 0, name: '' }
                return [`(${pt.x.toFixed(2)}, ${pt.y.toFixed(2)})`, pt.name]
              }}
            />
            <Scatter data={allPoints} isAnimationActive>
              {allPoints.map((pt, i) => (
                <Cell key={i} fill={pt.color} opacity={pt.name === 'This image' ? 1 : 0.7} />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-3 mt-3">
        {Object.entries(CENTROID_LABELS).map(([k, label]) => (
          <div key={k} className="flex items-center gap-1.5">
            <div className="w-2.5 h-2.5 rounded-full" style={{ background: CENTROID_COLORS[k] }} />
            <span className="text-xs text-gray-500">{label}</span>
          </div>
        ))}
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-full bg-white border border-gray-400" />
          <span className="text-xs text-gray-300 font-medium">This image</span>
        </div>
      </div>
    </Card>
  )
}
