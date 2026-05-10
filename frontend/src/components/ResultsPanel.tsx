import type { Detection } from '../hooks/useDetection'
import './ResultsPanel.css'

interface ResultsPanelProps {
  detections: Detection[]
  tesseractReady: boolean
}

const CLASS_LABELS: Record<string, string> = {
  car: '🚗 Voiture',
  truck: '🚛 Camion',
  bus: '🚌 Bus',
  motorbike: '🏍 Moto',
  bicycle: '🚲 Vélo',
}

export default function ResultsPanel({ detections, tesseractReady }: ResultsPanelProps) {
  if (detections.length === 0) {
    return (
      <div className="results-empty">
        <p className="results-title">Détections</p>
        <p className="results-none">Aucun véhicule détecté</p>
      </div>
    )
  }

  return (
    <div className="results-panel">
      <p className="results-title">
        Détections <span className="count">{detections.length}</span>
      </p>
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

            {d.plate ? (
              <div className="result-plate">
                <span className="plate-label">Plaque</span>
                <span className="plate-value">{d.plate}</span>
                {d.plateConfidence && (
                  <span className="plate-conf">{Math.round(d.plateConfidence * 100)}%</span>
                )}
              </div>
            ) : (
              <div className="result-plate result-plate--pending">
                <span className="plate-label">Plaque</span>
                <span className="plate-pending">
                  {tesseractReady ? 'Lecture en cours…' : 'OCR en chargement…'}
                </span>
              </div>
            )}
          </li>
        ))}
      </ul>
    </div>
  )
}
