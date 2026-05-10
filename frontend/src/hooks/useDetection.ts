import { useRef, useState, useCallback, useEffect } from 'react'
import * as cocoSsd from '@tensorflow-models/coco-ssd'
import '@tensorflow/tfjs'
import { createWorker, PSM } from 'tesseract.js'

export interface BoundingBox {
  x: number
  y: number
  width: number
  height: number
}

export interface Detection {
  id: string
  bbox: BoundingBox
  class: string
  score: number
  plate?: string
  plateConfidence?: number
  speedKmh?: number
  timestamp: number
}

interface PlateCache {
  plate: string
  confidence: number
  ts: number
}

interface TrackedVehicle {
  id: string
  cx: number
  cy: number
  ts: number
  plate?: PlateCache
}

interface TextDetectorResult {
  rawValue: string
}

declare global {
  class TextDetector {
    detect(image: CanvasImageSource): Promise<TextDetectorResult[]>
  }
}

interface UseDetectionOptions {
  backendUrl: string
  onDetection: (detections: Detection[]) => void
  onBackendStatus?: (ok: boolean) => void
  onTesseractReady?: (ready: boolean) => void
}

const VEHICLE_CLASSES   = new Set(['car', 'truck', 'bus', 'motorbike', 'bicycle'])
const FRAME_INTERVAL_MS = 100
const PLATE_INTERVAL_MS = 1800   // lecture toutes les 1.8s
const PLATE_CACHE_TTL   = 12000  // garde la plaque 12s
const IOU_THRESHOLD     = 0.20

// ── helpers ──────────────────────────────────────────────────────────────────

function iou(a: BoundingBox, b: BoundingBox): number {
  const ix = Math.max(0, Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x))
  const iy = Math.max(0, Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y))
  const inter = ix * iy
  const union = a.width * a.height + b.width * b.height - inter
  return union <= 0 ? 0 : inter / union
}

/** Garde uniquement les caractères de plaque valides. */
function cleanPlate(text: string): string {
  const cleaned = text.toUpperCase().replace(/[^A-Z0-9]/g, '')
  // Filtre basique : longueur entre 4 et 9 car, au moins 1 chiffre et 1 lettre
  if (cleaned.length < 4 || cleaned.length > 9) return ''
  if (!/[A-Z]/.test(cleaned) || !/[0-9]/.test(cleaned)) return ''
  return cleaned
}

/** Zone plaque brute upscalée — couleur naturelle (pour TextDetector). */
function buildPlateCanvasRaw(video: HTMLVideoElement, bbox: BoundingBox): HTMLCanvasElement {
  const px = bbox.x + bbox.width  * 0.08
  const py = bbox.y + bbox.height * 0.55
  const pw = bbox.width  * 0.84
  const ph = bbox.height * 0.42
  const scale = Math.max(2.5, 360 / Math.max(pw, 1))
  const dw = Math.round(pw * scale)
  const dh = Math.round(ph * scale)
  const c  = document.createElement('canvas')
  c.width = dw; c.height = dh
  c.getContext('2d')!.drawImage(video, px, py, pw, ph, 0, 0, dw, dh)
  return c
}

/** Zone plaque binarisée — pour Tesseract.js. */
function buildPlateCanvas(video: HTMLVideoElement, bbox: BoundingBox): HTMLCanvasElement {
  const raw  = buildPlateCanvasRaw(video, bbox)
  const { width: dw, height: dh } = raw
  const out  = document.createElement('canvas')
  out.width  = dw; out.height = dh
  const oCtx = out.getContext('2d')!
  oCtx.filter = 'contrast(3.5) brightness(1.1) saturate(0)'
  oCtx.drawImage(raw, 0, 0)
  oCtx.filter = 'none'
  const id   = oCtx.getImageData(0, 0, dw, dh)
  const d    = id.data
  let sum = 0
  for (let i = 0; i < d.length; i += 4) sum += 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]
  const thr  = sum / (dw * dh)
  for (let i = 0; i < d.length; i += 4) {
    const v = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]
    d[i] = d[i + 1] = d[i + 2] = v > thr ? 255 : 0; d[i + 3] = 255
  }
  oCtx.putImageData(id, 0, 0)
  return out
}

// ── hook ─────────────────────────────────────────────────────────────────────

export function useDetection({ backendUrl, onDetection, onBackendStatus, onTesseractReady }: UseDetectionOptions) {
  const modelRef      = useRef<cocoSsd.ObjectDetection | null>(null)
  const tesseractRef  = useRef<Awaited<ReturnType<typeof createWorker>> | null>(null)
  const textDetRef    = useRef<TextDetector | null>(null)

  const [modelReady,    setModelReady]    = useState(false)
  const [loading,       setLoading]       = useState(false)
  const [tesseractReady, setTesseractReady] = useState(false)

  const runningRef       = useRef(false)
  const animRef          = useRef<number>(0)
  const lastFrameRef     = useRef<number>(0)
  const lastPlateRef     = useRef<number>(0)
  const tracksRef        = useRef<Map<string, TrackedVehicle>>(new Map())
  const nextIdRef        = useRef(0)
  const backendOkRef     = useRef<boolean | null>(null)

  // ── OCR engines ───────────────────────────────────────────────────────────

  /** Moteur 1 : TextDetector (OCR natif Chrome/Android — pas de download) */
  const initTextDetector = useCallback(() => {
    try {
      if ('TextDetector' in window) {
        textDetRef.current = new TextDetector()
        setTesseractReady(true)
        onTesseractReady?.(true)
        console.info('[OCR] TextDetector natif disponible ✓')
      } else {
        console.info('[OCR] TextDetector non disponible (non-Chrome), essai Tesseract.js…')
      }
    } catch (e) {
      console.warn('[OCR] TextDetector init error:', e)
    }
  }, [onTesseractReady])

  /** Moteur 2 : Tesseract.js (fallback WASM) */
  const initTesseract = useCallback(async () => {
    if (tesseractRef.current || textDetRef.current) return
    try {
      const w = await createWorker('eng', 1)
      await w.setParameters({
        tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        tessedit_pageseg_mode: PSM.SINGLE_LINE,
      })
      tesseractRef.current = w
      setTesseractReady(true)
      onTesseractReady?.(true)
    } catch (err) {
      console.warn('[OCR] Tesseract init failed:', err)
    }
  }, [onTesseractReady])

  // ── Backend health ────────────────────────────────────────────────────────

  const checkBackend = useCallback(async () => {
    try {
      const res = await fetch(`${backendUrl}/api/health`, { signal: AbortSignal.timeout(2000) })
      const ok  = res.ok
      if (ok !== backendOkRef.current) { backendOkRef.current = ok; onBackendStatus?.(ok) }
    } catch {
      if (backendOkRef.current !== false) { backendOkRef.current = false; onBackendStatus?.(false) }
    }
  }, [backendUrl, onBackendStatus])

  const loadModel = useCallback(async () => {
    if (modelRef.current) return
    setLoading(true)
    try {
      modelRef.current = await cocoSsd.load({ base: 'lite_mobilenet_v2' })
      setModelReady(true)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadModel()
    initTextDetector()        // synchrone, disponible immédiatement si supporté
    initTesseract()           // asynchrone, fallback si TextDetector absent
    checkBackend()
    const iv = setInterval(checkBackend, 6000)
    return () => {
      clearInterval(iv)
      runningRef.current = false
      if (animRef.current) cancelAnimationFrame(animRef.current)
      tesseractRef.current?.terminate()
      tesseractRef.current = null
    }
  }, [loadModel, initTextDetector, initTesseract, checkBackend])

  // ── Plate reading pipeline ────────────────────────────────────────────────

  const readPlateTextDetector = useCallback(
    async (rawCanvas: HTMLCanvasElement): Promise<PlateCache | null> => {
      const det = textDetRef.current
      if (!det) return null
      try {
        // TextDetector fonctionne mieux sur l'image couleur naturelle
        const results = await det.detect(rawCanvas)
        // Trier par longueur décroissante du texte détecté
        const sorted  = [...results].sort((a, b) => b.rawValue.length - a.rawValue.length)
        for (const r of sorted) {
          const plate = cleanPlate(r.rawValue)
          if (plate) return { plate, confidence: 0.85, ts: Date.now() }
        }
        return null
      } catch { return null }
    },
    [],
  )

  const readPlateTesseract = useCallback(
    async (canvas: HTMLCanvasElement): Promise<PlateCache | null> => {
      const w = tesseractRef.current
      if (!w) return null
      try {
        const { data } = await w.recognize(canvas)
        const plate    = cleanPlate(data.text)
        if (!plate) return null
        return { plate, confidence: data.confidence / 100, ts: Date.now() }
      } catch { return null }
    },
    [],
  )

  const readPlateBackend = useCallback(
    async (video: HTMLVideoElement, bbox: BoundingBox): Promise<PlateCache | null> => {
      try {
        const canvas = buildPlateCanvas(video, bbox)
        const blob   = await new Promise<Blob | null>(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.92))
        if (!blob) return null
        const form   = new FormData()
        form.append('image', blob, 'plate.jpg')
        const res    = await fetch(`${backendUrl}/api/plate`, { method: 'POST', body: form, signal: AbortSignal.timeout(4000) })
        if (!res.ok) return null
        const json   = await res.json() as { plate?: string; confidence?: number }
        const plate  = cleanPlate(json.plate ?? '')
        if (!plate) return null
        return { plate, confidence: json.confidence ?? 0.6, ts: Date.now() }
      } catch { return null }
    },
    [backendUrl],
  )

  const readPlate = useCallback(
    async (video: HTMLVideoElement, bbox: BoundingBox): Promise<PlateCache | null> => {
      // Priorité : backend → TextDetector (couleur) → Tesseract (binarisé)
      if (backendOkRef.current) {
        const r = await readPlateBackend(video, bbox)
        if (r) return r
      }
      const rawCanvas  = buildPlateCanvasRaw(video, bbox)
      const r1 = await readPlateTextDetector(rawCanvas)
      if (r1) return r1
      const binCanvas  = buildPlateCanvas(video, bbox)
      return readPlateTesseract(binCanvas)
    },
    [readPlateBackend, readPlateTextDetector, readPlateTesseract],
  )

  // ── Track management ──────────────────────────────────────────────────────

  const matchOrCreate = useCallback((bbox: BoundingBox, cx: number, cy: number, now: number): TrackedVehicle => {
    let best: TrackedVehicle | null = null
    let bestScore = IOU_THRESHOLD
    for (const t of tracksRef.current.values()) {
      if (now - t.ts > 4000) continue
      const tb: BoundingBox = { x: t.cx - bbox.width / 2, y: t.cy - bbox.height / 2, width: bbox.width, height: bbox.height }
      const s = iou(bbox, tb)
      if (s > bestScore) { bestScore = s; best = t }
    }
    if (best) { best.cx = cx; best.cy = cy; best.ts = now; return best }
    const t: TrackedVehicle = { id: `v${nextIdRef.current++}`, cx, cy, ts: now }
    tracksRef.current.set(t.id, t)
    return t
  }, [])

  const estimateSpeed = useCallback((track: TrackedVehicle, cx: number, cy: number, now: number): number | undefined => {
    const dt = (now - track.ts) / 1000
    if (dt < 0.05 || dt > 3) return undefined
    const dpx = Math.hypot(cx - track.cx, cy - track.cy)
    return Math.min((dpx * 0.005 / dt) * 3.6, 300)
  }, [])

  // ── Detection loop ────────────────────────────────────────────────────────

  const runDetection = useCallback(
    (video: HTMLVideoElement, canvas: HTMLCanvasElement) => {
      if (!runningRef.current) return

      const now = performance.now()
      if (now - lastFrameRef.current < FRAME_INTERVAL_MS) {
        animRef.current = requestAnimationFrame(() => runDetection(video, canvas))
        return
      }
      lastFrameRef.current = now

      if (!modelRef.current || video.readyState < 2) {
        animRef.current = requestAnimationFrame(() => runDetection(video, canvas))
        return
      }

      modelRef.current.detect(video).then(async preds => {
        if (!runningRef.current) return
        const ctx = canvas.getContext('2d')
        if (!ctx) return

        canvas.width  = video.videoWidth
        canvas.height = video.videoHeight
        ctx.clearRect(0, 0, canvas.width, canvas.height)

        const vehicles   = preds.filter(p => VEHICLE_CLASSES.has(p.class))
        const sendPlates = now - lastPlateRef.current > PLATE_INTERVAL_MS
        if (sendPlates) lastPlateRef.current = now

        for (const [id, t] of tracksRef.current) {
          if (now - t.ts > 6000) tracksRef.current.delete(id)
        }

        const results: Detection[] = await Promise.all(
          vehicles.map(async pred => {
            const [bx, by, bw, bh] = pred.bbox
            const bbox: BoundingBox = { x: bx, y: by, width: bw, height: bh }
            const cx = bx + bw / 2
            const cy = by + bh / 2

            const track = matchOrCreate(bbox, cx, cy, now)
            const speed = estimateSpeed(track, cx, cy, now)
            track.cx = cx; track.cy = cy; track.ts = now

            let cache = track.plate
            const expired = !cache || (now - cache.ts > PLATE_CACHE_TTL)
            if (sendPlates && expired && bw > 40 && bh > 35) {
              const r = await readPlate(video, bbox)
              if (r) { track.plate = r; cache = r }
            }

            // ── draw ────────────────────────────────────────────────────────
            const hc = pred.score > 0.65
            ctx.strokeStyle = hc ? '#4f8ef7' : '#fbbf24'
            ctx.lineWidth   = 2
            ctx.strokeRect(bx, by, bw, bh)

            // Zone plaque (debug)
            ctx.strokeStyle = 'rgba(34,211,160,0.5)'
            ctx.lineWidth   = 1
            ctx.setLineDash([4, 3])
            ctx.strokeRect(bx + bw * 0.08, by + bh * 0.58, bw * 0.84, bh * 0.38)
            ctx.setLineDash([])

            ctx.font = 'bold 12px Inter,sans-serif'
            const label = `${pred.class} ${Math.round(pred.score * 100)}%`
            const lw    = ctx.measureText(label).width + 10
            ctx.fillStyle = hc ? '#4f8ef7' : '#fbbf24'
            ctx.fillRect(bx, by - 20, lw, 20)
            ctx.fillStyle = '#fff'
            ctx.fillText(label, bx + 5, by - 5)

            if (cache?.plate) {
              ctx.font = 'bold 12px JetBrains Mono,monospace'
              const pl = cache.plate
              const pw2 = ctx.measureText(pl).width + 12
              ctx.fillStyle = 'rgba(34,211,160,0.85)'
              ctx.fillRect(bx, by + bh + 2, pw2, 20)
              ctx.fillStyle = '#fff'
              ctx.fillText(pl, bx + 6, by + bh + 15)
            }

            if (speed !== undefined) {
              ctx.font = 'bold 11px JetBrains Mono,monospace'
              const sl  = `${Math.round(speed)} km/h`
              const sw2 = ctx.measureText(sl).width + 10
              ctx.fillStyle = speed > 80 ? '#f87171' : '#fbbf24'
              ctx.fillRect(bx + bw - sw2, by - 20, sw2, 20)
              ctx.fillStyle = '#fff'
              ctx.fillText(sl, bx + bw - sw2 + 5, by - 5)
            }

            return { id: track.id, bbox, class: pred.class, score: pred.score,
              plate: cache?.plate, plateConfidence: cache?.confidence,
              speedKmh: speed, timestamp: Date.now() }
          }),
        )

        onDetection(results)
        animRef.current = requestAnimationFrame(() => runDetection(video, canvas))
      }).catch(() => {
        animRef.current = requestAnimationFrame(() => runDetection(video, canvas))
      })
    },
    [matchOrCreate, estimateSpeed, readPlate, onDetection],
  )

  const start = useCallback((video: HTMLVideoElement, canvas: HTMLCanvasElement) => {
    runningRef.current = true
    runDetection(video, canvas)
  }, [runDetection])

  const stop = useCallback(() => {
    runningRef.current = false
    tracksRef.current.clear()
    if (animRef.current) cancelAnimationFrame(animRef.current)
  }, [])

  return { modelReady, loading, tesseractReady, start, stop }
}
