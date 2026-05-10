import { useRef, useState, useCallback, useEffect } from 'react'
import { useDetection } from '../hooks/useDetection'
import type { Detection } from '../hooks/useDetection'
import './Camera.css'

interface CameraProps {
  onDetection: (detections: Detection[]) => void
  onRunningChange: (running: boolean) => void
  backendUrl: string
  onBackendStatus?: (ok: boolean) => void
  onTesseractReady?: (ready: boolean) => void
}

type CameraFacing = 'environment' | 'user'

export default function Camera({ onDetection, onRunningChange, backendUrl, onBackendStatus, onTesseractReady }: CameraProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)

  const [running, setRunning] = useState(false)
  const [facing, setFacing] = useState<CameraFacing>('environment')
  const [error, setError] = useState<string | null>(null)

  const { modelReady, loading, tesseractReady, start, stop } = useDetection({ backendUrl, onDetection, onBackendStatus, onTesseractReady })

  const startCamera = useCallback(async (facingMode: CameraFacing) => {
    setError(null)
    try {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop())
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode,
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      })
      streamRef.current = stream
      const video = videoRef.current!
      video.srcObject = stream
      await video.play()
    } catch (e) {
      setError(`Impossible d'accéder à la caméra : ${(e as Error).message}`)
    }
  }, [])

  const handleToggle = useCallback(async () => {
    if (running) {
      stop()
      streamRef.current?.getTracks().forEach(t => t.stop())
      streamRef.current = null
      const video = videoRef.current
      if (video) { video.srcObject = null }
      setRunning(false)
      onRunningChange(false)
      onDetection([])
    } else {
      await startCamera(facing)
      if (videoRef.current && canvasRef.current) {
        setRunning(true)
        onRunningChange(true)
        start(videoRef.current, canvasRef.current)
      }
    }
  }, [running, facing, start, stop, startCamera, onRunningChange, onDetection])

  const handleFlip = useCallback(async () => {
    const next: CameraFacing = facing === 'environment' ? 'user' : 'environment'
    setFacing(next)
    if (running) {
      stop()
      await startCamera(next)
      if (videoRef.current && canvasRef.current) {
        start(videoRef.current, canvasRef.current)
      }
    }
  }, [facing, running, start, stop, startCamera])

  useEffect(() => {
    return () => {
      stop()
      streamRef.current?.getTracks().forEach(t => t.stop())
    }
  }, [stop])

  return (
    <div className="camera-wrapper">
      <video
        ref={videoRef}
        className="camera-video"
        playsInline
        muted
        autoPlay
      />
      <canvas ref={canvasRef} className="camera-canvas" />

      {!running && (
        <div className="camera-placeholder">
          <div className="placeholder-icon">📷</div>
          <p className="placeholder-text">Caméra inactive</p>
          <p className="placeholder-sub">
            {loading
              ? 'Chargement du modèle IA…'
              : modelReady
                ? tesseractReady
                  ? '✓ Prêt (IA + OCR)'
                  : '✓ IA prête — OCR en chargement…'
                : 'Initialisation…'}
          </p>
        </div>
      )}

      {loading && (
        <div className="loading-overlay">
          <div className="spinner" />
          <p>Chargement TensorFlow…</p>
        </div>
      )}

      {!loading && modelReady && !tesseractReady && running && (
        <div className="ocr-loading-badge">
          <div className="spinner-sm" /> OCR en chargement…
        </div>
      )}

      {error && (
        <div className="error-overlay">
          <p>⚠️ {error}</p>
          <button className="btn-dismiss" onClick={() => setError(null)}>OK</button>
        </div>
      )}

      <div className="camera-controls">
        <button
          className={`btn-main ${running ? 'btn-stop' : 'btn-start'}`}
          onClick={handleToggle}
          disabled={loading}
        >
          {running ? '⏹ Arrêter' : '▶ Démarrer'}
        </button>

        <button className="btn-icon" onClick={handleFlip} title="Retourner caméra">
          🔄
        </button>
      </div>
    </div>
  )
}
