import { ShieldCheck, ShieldX, ShieldAlert, Clock } from 'lucide-react'
import clsx from 'clsx'
import type { Verdict, FakeType } from '@/types/detection'

const FAKE_TYPE_LABELS: Record<FakeType, string> = {
  gan:             'GAN Generated',
  face_swap:       'Face Swap',
  face_reenactment:'Face Reenactment',
  diffusion:       'Diffusion Model',
  photoshop:       'Photoshop / Edited',
  real:            'Authentic',
}

interface VerdictBannerProps {
  verdict: Verdict
  fakeProb: number
  confidence: number
  fakeType: FakeType
  fakeTypeConf: number
  inferenceMs: number
  warnings: string[]
}

export function VerdictBanner({
  verdict, fakeProb, confidence, fakeType, fakeTypeConf, inferenceMs, warnings,
}: VerdictBannerProps) {
  const config = {
    FAKE: {
      icon: ShieldX,
      label: 'DEEPFAKE DETECTED',
      bg: 'bg-fake/10',
      border: 'border-fake/40',
      text: 'text-fake',
      ring: '#ef4444',
    },
    REAL: {
      icon: ShieldCheck,
      label: 'AUTHENTIC IMAGE',
      bg: 'bg-real/10',
      border: 'border-real/40',
      text: 'text-real',
      ring: '#22c55e',
    },
    UNCERTAIN: {
      icon: ShieldAlert,
      label: 'UNCERTAIN',
      bg: 'bg-uncertain/10',
      border: 'border-uncertain/40',
      text: 'text-uncertain',
      ring: '#f59e0b',
    },
  }[verdict]

  const Icon = config.icon
  const fakePercent = Math.round(fakeProb * 100)
  const confPercent = Math.round(confidence * 100)
  const circumference = 2 * Math.PI * 54

  return (
    <div className={clsx('card border', config.border, config.bg)}>
      <div className="flex flex-col md:flex-row items-center gap-8">
        {/* Circular confidence gauge */}
        <div className="relative flex-shrink-0">
          <svg width={128} height={128} className="rotate-[-90deg]">
            <circle cx={64} cy={64} r={54} fill="none" stroke="#1f2937" strokeWidth={10} />
            <circle
              cx={64} cy={64} r={54} fill="none"
              stroke={config.ring} strokeWidth={10}
              strokeLinecap="round"
              strokeDasharray={circumference}
              strokeDashoffset={circumference * (1 - fakeProb)}
              style={{ transition: 'stroke-dashoffset 1s ease' }}
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className={clsx('text-3xl font-bold', config.text)}>{fakePercent}%</span>
            <span className="text-xs text-gray-500">fake prob</span>
          </div>
        </div>

        {/* Verdict text */}
        <div className="flex-1 text-center md:text-left">
          <div className="flex items-center justify-center md:justify-start gap-3 mb-2">
            <Icon size={28} className={config.text} />
            <h2 className={clsx('text-2xl font-bold', config.text)}>{config.label}</h2>
          </div>

          <div className="flex flex-wrap gap-3 justify-center md:justify-start mb-4">
            <span className={clsx('badge', verdict === 'FAKE' ? 'badge-fake' : verdict === 'REAL' ? 'badge-real' : 'badge-uncertain')}>
              Confidence: {confPercent}%
            </span>
            <span className="bg-gray-800 text-gray-300 border border-gray-700 rounded-full px-3 py-1 text-sm font-medium">
              {FAKE_TYPE_LABELS[fakeType]} ({Math.round(fakeTypeConf * 100)}%)
            </span>
          </div>

          {warnings.length > 0 && (
            <div className="mt-3 space-y-1">
              {warnings.map((w, i) => (
                <p key={i} className="text-xs text-uncertain flex items-start gap-1">
                  <ShieldAlert size={12} className="mt-0.5 flex-shrink-0" />
                  {w}
                </p>
              ))}
            </div>
          )}

          <div className="flex items-center gap-1 mt-3 text-xs text-gray-600">
            <Clock size={12} />
            Analysis completed in {Math.round(inferenceMs)}ms
          </div>
        </div>
      </div>
    </div>
  )
}
