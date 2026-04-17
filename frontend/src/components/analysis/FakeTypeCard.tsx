import type { FakeTypeClassification } from '@/types/detection'
import { Card } from '@/components/shared/Card'
import { CheckCircle2 } from 'lucide-react'
import clsx from 'clsx'
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer,
} from 'recharts'

const TYPE_COLORS: Record<string, string> = {
  gan:             'bg-purple-500/20 text-purple-300 border-purple-500/30',
  face_swap:       'bg-orange-500/20 text-orange-300 border-orange-500/30',
  face_reenactment:'bg-yellow-500/20 text-yellow-300 border-yellow-500/30',
  diffusion:       'bg-blue-500/20 text-blue-300 border-blue-500/30',
  photoshop:       'bg-pink-500/20 text-pink-300 border-pink-500/30',
  real:            'bg-real/10 text-real border-real/30',
}
const TYPE_LABELS: Record<string, string> = {
  gan:             'GAN Generated',
  face_swap:       'Face Swap',
  face_reenactment:'Face Reenactment',
  diffusion:       'Diffusion Model',
  photoshop:       'Photoshop / Edited',
  real:            'Authentic',
}

interface FakeTypeCardProps {
  data: FakeTypeClassification
}

export function FakeTypeCard({ data }: FakeTypeCardProps) {
  const radarData = Object.entries(data.type_probabilities).map(([k, v]) => ({
    type: TYPE_LABELS[k] ?? k,
    value: Math.round(v * 100),
  }))

  return (
    <Card title="Forgery Type Classification">
      <div className="flex flex-col lg:flex-row gap-6">
        {/* Left: badge + reasoning */}
        <div className="flex-1">
          <div className={clsx(
            'inline-flex items-center gap-2 px-4 py-2 rounded-full border text-sm font-semibold mb-4',
            TYPE_COLORS[data.predicted_type],
          )}>
            {TYPE_LABELS[data.predicted_type]}
            <span className="opacity-70">({Math.round(data.confidence * 100)}%)</span>
          </div>

          <ul className="space-y-2">
            {data.reasoning.map((r, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-gray-400">
                <CheckCircle2 size={14} className="text-brand-500 mt-0.5 flex-shrink-0" />
                {r}
              </li>
            ))}
          </ul>
        </div>

        {/* Right: radar */}
        <div className="w-full lg:w-56 h-48">
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart data={radarData} margin={{ top: 5, right: 20, bottom: 5, left: 20 }}>
              <PolarGrid stroke="#374151" />
              <PolarAngleAxis
                dataKey="type"
                tick={{ fill: '#6b7280', fontSize: 9 }}
              />
              <Radar
                dataKey="value"
                stroke="#6366f1"
                fill="#6366f1"
                fillOpacity={0.25}
                strokeWidth={2}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </Card>
  )
}
