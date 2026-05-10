import { useState, useCallback, useRef } from 'react'
import Camera from './components/Camera'
import ResultsPanel from './components/ResultsPanel'
import type { Detection } from './hooks/useDetection'
import './App.css'

export default function App() {
  const [detections, setDetections] = useState<Detection[]>([])
  const [isRunning, setIsRunning] = useState(false)
  const [backendUrl, setBackendUrl] = useState('http://localhost:8000')
  const [backendOk, setBackendOk] = useState<boolean | null>(null)
  const [tesseractReady, setTesseractReady] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleDetection = useCallback((d: Detection[]) => {
    setDetections(d)
  }, [])

  const backendStatus = backendOk === null
    ? { label: '…', cls: 'status-unknown' }
    : backendOk
      ? { label: 'Backend connecté', cls: 'status-ok' }
      : { label: 'Backend hors-ligne (Tesseract.js actif)', cls: 'status-warn' }

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-left">
          <span className="logo">⚡</span>
          <h1>AiSecurity</h1>
          <span className="badge">Détection route</span>
        </div>
        <div className="header-right">
          <div className={`status-dot ${isRunning ? 'active' : ''}`} />
          <span className="status-label">{isRunning ? 'Actif' : 'Inactif'}</span>
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
          <ResultsPanel detections={detections} tesseractReady={tesseractReady} />
          <div className="config-panel">
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
              Sans backend : OCR via Tesseract.js (navigateur, précision ~60%).<br />
              Avec backend Python : OCR via Plate Recognizer (précision ~95%).
            </p>
          </div>
        </aside>
      </main>
    </div>
  )
}
