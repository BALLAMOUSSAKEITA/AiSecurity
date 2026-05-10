import type { Detection } from '../hooks/useDetection'
import './ResultsPanel.css'

interface ResultsPanelProps {
  detections: Detection[]
  tesseractReady: boolean
}

const CLASS_LABELS: Record<string, string> = {
  car:       '🚗 Voiture',
  truck:     '🚛 Camion',
  bus:       '🚌 Bus',
  motorbike: '🏍 Moto',
  bicycle:   '🚲 Vélo',
}

export default function ResultsPanel({ detections, tesseractReady }: ResultsPanelProps) {
  return (
    <div>
      <div className="results-header">
        <span className="results-title">
          Détections
          {detections.length > 0 && (
            <span className="count-badge">{detections.length}</span>
          )}
        </span>
      </div>

      {detections.length === 0 ? (
        <div className="results-empty">
          <div className="empty-icon">🚫</div>
          <p className="results-none">Aucun véhicule détecté</p>
        </div>
      ) : (
        <ul className="results-list">
          {detections.map(d => (
            <li key={d.id} className="result-card">

              <div className="result-header">
                <span className="result-class">{CLASS_LABELS[d.class] ?? d.class}</span>
                <span className="result-score">{Math.round(d.score * 100)}%</span>
              </div>

              {d.speedKmh !== undefined && (
                <div className={`result-speed ${d.speedKmh > 80 ? 'speed-high' : d.speedKmh > 50 ? 'speed-mid' : 'speed-ok'}`}>
                  <span className="speed-icon">⚡</span>
                  <span className="speed-value">{Math.round(d.speedKmh)} km/h</span>
                </div>
              )}

              {/* Plaque validée */}
              {d.plate ? (
                <div className="result-plate">
                  <span className="plate-label">Plaque</span>
                  <span className="plate-value">{d.plate}</span>
                  {d.plateConfidence !== undefined && (
                    <span className="plate-conf">{Math.round(d.plateConfidence * 100)}%</span>
                  )}
                  {d.ocrEngine && <span className="ocr-engine">{d.ocrEngine}</span>}
                </div>
              ) : (
                <div className="result-plate result-plate--pending">
                  <span className="plate-label">Plaque</span>
                  <span className="plate-pending">
                    {tesseractReady ? 'Lecture en cours…' : 'OCR en chargement…'}
                  </span>
                </div>
              )}

              {/* Debug : texte brut OCR */}
              {d.ocrRaw && !d.plate && (
                <div className="ocr-debug">
                  <span className="ocr-debug-label">{d.ocrEngine ?? 'OCR'} lit :</span>
                  <span className="ocr-debug-raw">"{d.ocrRaw.slice(0, 30)}"</span>
                </div>
              )}

            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
