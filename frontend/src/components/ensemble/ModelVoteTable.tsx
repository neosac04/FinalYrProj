import type { ModelPrediction } from '@/types/detection'
import clsx from 'clsx'
import { Card } from '@/components/shared/Card'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'

const MODEL_LABELS: Record<string, string> = {
  univfd:        'UnivFD (CLIP ViT-L/14)',
  efficientnet:  'EfficientNet-B4',
  xception:      'Xception (FF++)',
}

interface ModelVoteTableProps {
  predictions: ModelPrediction[]
  weights: Record<string, number>
}

export function ModelVoteTable({ predictions, weights }: ModelVoteTableProps) {
  const chartData = predictions.map((p) => ({
    name: MODEL_LABELS[p.model_name] ?? p.model_name,
    fake: Math.round(p.fake_probability * 100),
    weight: Math.round((weights[p.model_name] ?? 0) * 100),
    ms: Math.round(p.inference_time_ms),
  }))

  return (
    <Card title="Model Ensemble Breakdown">
      {/* Bar chart */}
      <div className="h-40 mb-6">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
            <XAxis dataKey="name" tick={{ fill: '#6b7280', fontSize: 10 }} />
            <YAxis tick={{ fill: '#6b7280', fontSize: 10 }} domain={[0, 100]} />
            <Tooltip
              contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8 }}
              labelStyle={{ color: '#e5e7eb' }}
              formatter={(v: number) => [`${v}%`]}
            />
            <Bar dataKey="fake" radius={[4, 4, 0, 0]}>
              {chartData.map((d) => (
                <Cell
                  key={d.name}
                  fill={d.fake >= 65 ? '#ef4444' : d.fake <= 35 ? '#22c55e' : '#f59e0b'}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-gray-500 text-xs uppercase border-b border-gray-800">
              <th className="text-left pb-2">Model</th>
              <th className="text-right pb-2">Fake %</th>
              <th className="text-right pb-2">Weight</th>
              <th className="text-right pb-2">Time</th>
            </tr>
          </thead>
          <tbody>
            {predictions.map((p) => {
              const fake = Math.round(p.fake_probability * 100)
              const color = fake >= 65 ? 'text-fake' : fake <= 35 ? 'text-real' : 'text-uncertain'
              return (
                <tr key={p.model_name} className="border-b border-gray-800/50">
                  <td className="py-2.5 text-gray-300 font-medium">
                    {MODEL_LABELS[p.model_name] ?? p.model_name}
                  </td>
                  <td className={clsx('py-2.5 text-right font-bold', color)}>{fake}%</td>
                  <td className="py-2.5 text-right text-gray-500">
                    {Math.round((weights[p.model_name] ?? 0) * 100)}%
                  </td>
                  <td className="py-2.5 text-right text-gray-600 text-xs">
                    {Math.round(p.inference_time_ms)}ms
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </Card>
  )
}
