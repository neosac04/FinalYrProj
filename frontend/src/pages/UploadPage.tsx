import { useNavigate } from 'react-router-dom'
import { X, Zap } from 'lucide-react'
import { DropZone } from '@/components/upload/DropZone'
import { UploadProgress } from '@/components/upload/UploadProgress'
import { useDetectionStore } from '@/store/detectionStore'
import { useEffect } from 'react'

export function UploadPage() {
  const navigate = useNavigate()
  const { status, uploadProgress, selectedFile, previewUrl, error, setFile, submit, reset } =
    useDetectionStore()

  useEffect(() => {
    if (status === 'complete') navigate('/results')
  }, [status, navigate])

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b border-gray-800 px-6 py-4">
        <div className="max-w-4xl mx-auto flex items-center gap-3">
          <div className="p-2 bg-brand-600 rounded-xl">
            <Zap size={20} className="text-white" />
          </div>
          <div>
            <h1 className="font-bold text-gray-100">Deepfake Detector</h1>
            <p className="text-xs text-gray-500">Multi-model AI Image Forensics</p>
          </div>
        </div>
      </header>

      <main className="flex-1 flex flex-col items-center justify-center px-4 py-12">
        <div className="w-full max-w-2xl space-y-6">
          {/* Title */}
          <div className="text-center">
            <h2 className="text-3xl font-bold text-gray-100 mb-2">
              Detect Deepfakes & AI-Generated Images
            </h2>
            <p className="text-gray-500 max-w-lg mx-auto">
              4 deep learning models — UnivFD, EfficientNet-B7, FreqNet, DIRE — analyse your image
              with facial forensics, frequency analysis, and PRNU camera fingerprinting.
            </p>
          </div>

          {/* Model pills */}
          <div className="flex flex-wrap gap-2 justify-center">
            {[
              { label: 'UnivFD (CLIP)', desc: 'Universal' },
              { label: 'EfficientNet-B4', desc: 'Face Forgeries' },
              { label: 'Xception (FF++)', desc: 'Texture Artifacts' },
              { label: 'DistilDIRE', desc: 'Diffusion' },
            ].map((m) => (
              <div key={m.label} className="bg-gray-900 border border-gray-800 rounded-full px-3 py-1.5 text-xs">
                <span className="text-gray-300 font-medium">{m.label}</span>
                <span className="text-gray-600 ml-1">· {m.desc}</span>
              </div>
            ))}
          </div>

          {/* Upload area */}
          {(status === 'idle' || status === 'error') && !selectedFile && (
            <DropZone onFile={setFile} />
          )}

          {/* Preview + submit */}
          {selectedFile && (status === 'idle' || status === 'error') && (
            <div className="card space-y-4">
              <div className="flex items-center gap-4">
                {previewUrl && (
                  <img
                    src={previewUrl}
                    alt="preview"
                    className="w-20 h-20 object-cover rounded-xl border border-gray-700"
                  />
                )}
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-gray-200 truncate">{selectedFile.name}</p>
                  <p className="text-sm text-gray-500">{(selectedFile.size / 1024).toFixed(1)} KB</p>
                </div>
                <button onClick={reset} className="text-gray-600 hover:text-gray-400 p-2">
                  <X size={18} />
                </button>
              </div>

              {error && (
                <div className="bg-fake/10 border border-fake/30 rounded-lg px-4 py-3 text-sm text-fake">
                  {error}
                </div>
              )}

              <button className="btn-primary w-full" onClick={submit}>
                Analyse Image
              </button>
            </div>
          )}

          {/* Progress */}
          {(status === 'uploading' || status === 'analyzing') && (
            <UploadProgress status={status} progress={uploadProgress} />
          )}

          {/* Feature grid */}
          {status === 'idle' && !selectedFile && (
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mt-4">
              {[
                { icon: '🔬', title: 'Explainability Heatmaps', desc: 'See exactly what regions look fake' },
                { icon: '🫀', title: 'Anatomical Analysis', desc: 'Iris circularity, jaw seams, eye reflections' },
                { icon: '📊', title: 'Frequency Forensics', desc: 'FFT/DCT GAN artifact detection' },
                { icon: '📷', title: 'Camera Fingerprint', desc: 'PRNU noise authenticity verification' },
                { icon: '🧬', title: 'Fake Type Classification', desc: 'GAN vs Diffusion vs Face Swap' },
                { icon: '📡', title: 'PCA Feature Space', desc: 'Visual cluster positioning' },
              ].map((f) => (
                <div key={f.title} className="card-sm">
                  <div className="text-2xl mb-2">{f.icon}</div>
                  <p className="text-sm font-medium text-gray-300">{f.title}</p>
                  <p className="text-xs text-gray-600 mt-1">{f.desc}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      </main>
    </div>
  )
}
