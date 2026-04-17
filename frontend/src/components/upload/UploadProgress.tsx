import { Spinner } from '@/components/shared/Spinner'

interface UploadProgressProps {
  status: 'uploading' | 'analyzing'
  progress: number
}

export function UploadProgress({ status, progress }: UploadProgressProps) {
  return (
    <div className="card flex flex-col items-center gap-6 py-10">
      <Spinner className="w-12 h-12" />
      <div className="text-center">
        <p className="text-lg font-semibold text-gray-200">
          {status === 'uploading' ? 'Uploading image…' : 'Running detection pipeline…'}
        </p>
        <p className="text-sm text-gray-500 mt-1">
          {status === 'uploading'
            ? `${progress}% uploaded`
            : 'Running 4 models + forensic analysis in parallel'}
        </p>
      </div>
      {status === 'uploading' && (
        <div className="w-64 h-2 bg-gray-800 rounded-full overflow-hidden">
          <div
            className="h-full bg-brand-500 rounded-full transition-all duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
      )}
      {status === 'analyzing' && (
        <div className="grid grid-cols-4 gap-3 text-xs text-gray-500">
          {['UnivFD', 'EfficientNet-B4', 'Xception', 'DIRE'].map((m) => (
            <div key={m} className="flex flex-col items-center gap-1">
              <div className="w-2 h-2 rounded-full bg-brand-500 animate-pulse" />
              {m}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
