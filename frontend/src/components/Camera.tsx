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
        video: { facingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
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
      if (videoRef.current) videoRef.current.srcObject = null
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
      if (videoRef.current && canvasRef.current) start(videoRef.current, canvasRef.current)
    }
  }, [facing, running, start, stop, startCamera])

  useEffect(() => {
    return () => {
      stop()
      streamRef.current?.getTracks().forEach(t => t.stop())
    }
  }, [stop])

  const statusLabel = loading
    ? 'Chargement du modèle IA…'
    : modelReady
      ? tesseractReady ? 'IA + OCR prêts' : 'IA prête — OCR en chargement…'
      : 'Initialisation…'

  return (
    <div className={`camera-wrapper ${running ? 'running' : ''}`}>
      <video ref={videoRef} className="camera-video" playsInline muted autoPlay />
      <canvas ref={canvasRef} className="camera-canvas" />

      {/* Placeholder quand caméra off */}
      {!running && !loading && (
        <div className="camera-placeholder">
          <div className="placeholder-icon-wrap">📷</div>
          <p className="placeholder-text">Caméra inactive</p>
          <p className="placeholder-sub">
            {modelReady
              ? <><strong>Modèle IA prêt</strong> — appuyez sur Démarrer</>
              : statusLabel}
          </p>
        </div>
      )}

      {/* Chargement TensorFlow */}
      {loading && (
        <div className="loading-overlay">
          <div className="loading-content">
            <div className="spinner" />
            <p className="loading-title">Chargement du modèle IA</p>
            <p className="loading-sub">TensorFlow.js COCO-SSD…</p>
          </div>
        </div>
      )}

      {/* OCR en cours de chargement (pendant la détection) */}
      {!loading && modelReady && !tesseractReady && running && (
        <div className="ocr-loading-badge">
          <div className="spinner-sm" />
          OCR en chargement…
        </div>
      )}

      {/* Erreur caméra */}
      {error && (
        <div className="error-overlay">
          <div className="error-icon">⚠️</div>
          <p className="error-msg">{error}</p>
          <button className="btn-dismiss" onClick={() => setError(null)}>Fermer</button>
        </div>
      )}

      {/* Contrôles */}
      <div className="camera-controls">
        <button
          className={`btn-main ${running ? 'btn-stop' : 'btn-start'}`}
          onClick={handleToggle}
          disabled={loading}
        >
          {running ? '⏹ Arrêter' : '▶ Démarrer'}
        </button>
        <button className="btn-icon" onClick={handleFlip} title="Changer de caméra">
          🔄
        </button>
      </div>
    </div>
  )
}
