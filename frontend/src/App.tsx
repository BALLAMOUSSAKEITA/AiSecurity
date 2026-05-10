import { useState, useCallback, useRef } from 'react'
import Camera from './components/Camera'
import ResultsPanel from './components/ResultsPanel'
import type { Detection } from './hooks/useDetection'
import './App.css'

export default function App() {
  const [detections, setDetections] = useState<Detection[]>([])
  const [isRunning, setIsRunning] = useState(false)
  const [backendUrl, setBackendUrl] = useState(
    import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:8000'
  )
  const [backendOk, setBackendOk] = useState<boolean | null>(null)
  const [tesseractReady, setTesseractReady] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleDetection = useCallback((d: Detection[]) => {
    setDetections(d)
  }, [])

  const backendStatus = backendOk === null
    ? { label: 'Vérification du backend…', cls: 'status-unknown' }
    : backendOk
      ? { label: 'Backend connecté — OCR haute précision actif', cls: 'status-ok' }
      : { label: 'Backend hors-ligne — Tesseract.js (navigateur) actif', cls: 'status-warn' }

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-left">
          <div className="logo-mark">⚡</div>
          <div className="header-title">
            <h1>AiSecurity</h1>
            <span className="byline">by Balla Moussa Keita</span>
          </div>
          <span className="badge">Détection route</span>
        </div>
        <div className="header-right">
          <div className={`live-indicator ${isRunning ? 'active' : ''}`}>
            <span className="live-dot" />
            {isRunning ? 'LIVE' : 'OFF'}
          </div>
        </div>
      </header>

      <div className={`backend-bar ${backendStatus.cls}`}>
        <span className="backend-dot" />
        <span>{backendStatus.label}</span>
      </div>

      <main className="app-main">
        <div className="camera-container">
          <Camera
            onDetection={handleDetection}
            onRunningChange={setIsRunning}
            backendUrl={backendUrl}
            onBackendStatus={setBackendOk}
            onTesseractReady={setTesseractReady}
          />
        </div>

        <aside className="sidebar">
          <div className="sidebar-section">
            <ResultsPanel detections={detections} tesseractReady={tesseractReady} />
          </div>

          <div className="sidebar-section config-panel">
            <label className="config-label">URL serveur backend</label>
            <input
              ref={inputRef}
              className="config-input"
              defaultValue={backendUrl}
              onBlur={e => setBackendUrl(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && setBackendUrl((e.target as HTMLInputElement).value)}
              placeholder="http://localhost:8000"
            />
            <p className="config-hint">
              Sans backend : Tesseract.js (~60% précision).<br />
              Avec backend Python + Plate Recognizer : ~95%.
            </p>
          </div>

          <div className="sidebar-footer">
            <span className="creator-label">Créé par</span>
            <span className="creator-name">Balla Moussa Keita</span>
          </div>
        </aside>
      </main>
    </div>
  )
}
