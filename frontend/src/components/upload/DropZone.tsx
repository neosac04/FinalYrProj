import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, ImageIcon } from 'lucide-react'
import clsx from 'clsx'

interface DropZoneProps {
  onFile: (file: File) => void
  disabled?: boolean
}

export function DropZone({ onFile, disabled }: DropZoneProps) {
  const onDrop = useCallback(
    (accepted: File[]) => { if (accepted[0]) onFile(accepted[0]) },
    [onFile],
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'] },
    maxFiles: 1,
    disabled,
  })

  return (
    <div
      {...getRootProps()}
      className={clsx(
        'relative border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all duration-200',
        isDragActive
          ? 'border-brand-500 bg-brand-900/20'
          : 'border-gray-700 hover:border-gray-500 hover:bg-gray-900/50',
        disabled && 'opacity-50 cursor-not-allowed',
      )}
    >
      <input {...getInputProps()} />
      <div className="flex flex-col items-center gap-4">
        <div className={clsx(
          'p-4 rounded-2xl transition-colors',
          isDragActive ? 'bg-brand-600/20' : 'bg-gray-800',
        )}>
          {isDragActive ? (
            <ImageIcon size={40} className="text-brand-400" />
          ) : (
            <Upload size={40} className="text-gray-500" />
          )}
        </div>
        {isDragActive ? (
          <p className="text-brand-400 font-semibold text-lg">Drop to analyse</p>
        ) : (
          <>
            <div>
              <p className="text-gray-300 font-semibold text-lg">
                Drag & drop an image here
              </p>
              <p className="text-gray-500 text-sm mt-1">
                or click to browse — JPG, PNG, WebP, BMP (max 10 MB)
              </p>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
