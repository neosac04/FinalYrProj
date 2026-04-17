import { create } from 'zustand'
import { immer } from 'zustand/middleware/immer'
import { submitDetection } from '@/api/detection'
import type { DetectionResponse } from '@/types/detection'

type Status = 'idle' | 'uploading' | 'analyzing' | 'complete' | 'error'

interface DetectionStore {
  status: Status
  uploadProgress: number
  selectedFile: File | null
  previewUrl: string | null
  result: DetectionResponse | null
  error: string | null
  selectedHeatmapModel: string

  setFile: (file: File) => void
  submit: () => Promise<void>
  setHeatmapModel: (model: string) => void
  reset: () => void
}

export const useDetectionStore = create<DetectionStore>()(
  immer((set, get) => ({
    status: 'idle',
    uploadProgress: 0,
    selectedFile: null,
    previewUrl: null,
    result: null,
    error: null,
    selectedHeatmapModel: 'ensemble',

    setFile: (file) => {
      const prev = get().previewUrl
      if (prev) URL.revokeObjectURL(prev)
      set((s) => {
        s.selectedFile = file
        s.previewUrl = URL.createObjectURL(file)
        s.result = null
        s.error = null
        s.status = 'idle'
      })
    },

    submit: async () => {
      const file = get().selectedFile
      if (!file) return
      set((s) => { s.status = 'uploading'; s.uploadProgress = 0; s.error = null })
      try {
        const result = await submitDetection(file, (pct) => {
          set((s) => {
            s.uploadProgress = pct
            if (pct === 100) s.status = 'analyzing'
          })
        })
        set((s) => { s.result = result; s.status = 'complete' })
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : 'Detection failed. Please try again.'
        set((s) => { s.error = msg; s.status = 'error' })
      }
    },

    setHeatmapModel: (model) => set((s) => { s.selectedHeatmapModel = model }),

    reset: () => {
      const prev = get().previewUrl
      if (prev) URL.revokeObjectURL(prev)
      set((s) => {
        s.status = 'idle'
        s.uploadProgress = 0
        s.selectedFile = null
        s.previewUrl = null
        s.result = null
        s.error = null
      })
    },
  })),
)
