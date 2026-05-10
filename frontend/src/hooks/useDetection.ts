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

interface UseDetectionOptions {
  backendUrl: string
  onDetection: (detections: Detection[]) => void
  onBackendStatus?: (ok: boolean) => void
  onTesseractReady?: (ready: boolean) => void
}

const VEHICLE_CLASSES = new Set(['car', 'truck', 'bus', 'motorbike', 'bicycle'])
const FRAME_INTERVAL_MS = 100
const PLATE_SEND_INTERVAL_MS = 2000
const PLATE_CACHE_TTL_MS = 8000
const IOU_MATCH_THRESHOLD = 0.3

function iou(a: BoundingBox, b: BoundingBox): number {
  const ax2 = a.x + a.width
  const ay2 = a.y + a.height
  const bx2 = b.x + b.width
  const by2 = b.y + b.height
  const ix = Math.max(0, Math.min(ax2, bx2) - Math.max(a.x, b.x))
  const iy = Math.max(0, Math.min(ay2, by2) - Math.max(a.y, b.y))
  const inter = ix * iy
  const union = a.width * a.height + b.width * b.height - inter
  return union <= 0 ? 0 : inter / union
}

function cleanPlateText(text: string): string {
  return text
    .toUpperCase()
    .replace(/[^A-Z0-9-]/g, '')
    .trim()
}

export function useDetection({ backendUrl, onDetection, onBackendStatus, onTesseractReady }: UseDetectionOptions) {
  const modelRef = useRef<cocoSsd.ObjectDetection | null>(null)
  const tesseractRef = useRef<Awaited<ReturnType<typeof createWorker>> | null>(null)
  const [modelReady, setModelReady] = useState(false)
  const [loading, setLoading] = useState(false)
  const [tesseractReady, setTesseractReady] = useState(false)
  const runningRef = useRef(false)
  const animRef = useRef<number>(0)
  const lastFrameRef = useRef<number>(0)
  const lastPlateSendRef = useRef<number>(0)
  const tracksRef = useRef<Map<string, TrackedVehicle>>(new Map())
  const nextIdRef = useRef(0)
  const backendOkRef = useRef<boolean | null>(null)

  const checkBackend = useCallback(async () => {
    try {
      const res = await fetch(`${backendUrl}/api/health`, { signal: AbortSignal.timeout(2000) })
      const ok = res.ok
      if (ok !== backendOkRef.current) {
        backendOkRef.current = ok
        onBackendStatus?.(ok)
      }
    } catch {
      if (backendOkRef.current !== false) {
        backendOkRef.current = false
        onBackendStatus?.(false)
      }
    }
  }, [backendUrl, onBackendStatus])

  const initTesseract = useCallback(async () => {
    if (tesseractRef.current) return
    try {
      // v7 : pas de chemins personnalisés, on utilise les défauts CDN jsdelivr
      const worker = await createWorker('eng', 1)
      await worker.setParameters({
        tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-',
        tessedit_pageseg_mode: PSM.SINGLE_WORD,
      })
      tesseractRef.current = worker
      setTesseractReady(true)
      onTesseractReady?.(true)
    } catch (err) {
      console.error('[Tesseract] init failed:', err)
    }
  }, [])

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
    initTesseract()
    checkBackend()
    const interval = setInterval(checkBackend, 5000)
    return () => {
      clearInterval(interval)
      runningRef.current = false
      if (animRef.current) cancelAnimationFrame(animRef.current)
      tesseractRef.current?.terminate()
      tesseractRef.current = null
    }
  }, [loadModel, initTesseract, checkBackend])

  const cropPlateArea = useCallback(
    (video: HTMLVideoElement, bbox: BoundingBox): HTMLCanvasElement => {
      // La plaque est dans le tiers inférieur du véhicule, centré horizontalement
      const plateX = bbox.x + bbox.width * 0.1
      const plateY = bbox.y + bbox.height * 0.6   // 60% depuis le haut
      const plateW = bbox.width * 0.8
      const plateH = bbox.height * 0.38

      // Upscale agressif : on veut au moins 300px de large pour l'OCR
      const scale = Math.max(1, 300 / plateW)
      const dstW = Math.round(plateW * scale)
      const dstH = Math.round(plateH * scale)

      // Canvas 1 : crop brut upscalé
      const raw = document.createElement('canvas')
      raw.width = dstW
      raw.height = dstH
      const rCtx = raw.getContext('2d')!
      rCtx.drawImage(video, plateX, plateY, plateW, plateH, 0, 0, dstW, dstH)

      // Canvas 2 : preprocessing (contraste + binarisation)
      const out = document.createElement('canvas')
      out.width = dstW
      out.height = dstH
      const oCtx = out.getContext('2d')!

      // Filtre CSS : augmenter contraste et saturation avant la lecture
      oCtx.filter = 'contrast(2.5) brightness(1.1) grayscale(1)'
      oCtx.drawImage(raw, 0, 0)
      oCtx.filter = 'none'

      // Binarisation manuelle via ImageData
      const img = oCtx.getImageData(0, 0, dstW, dstH)
      const d = img.data
      for (let i = 0; i < d.length; i += 4) {
        const gray = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]
        const bin = gray > 128 ? 255 : 0
        d[i] = d[i + 1] = d[i + 2] = bin
      }
      oCtx.putImageData(img, 0, 0)

      return out
    },
    [],
  )

  const cropVehicle = useCallback(
    (video: HTMLVideoElement, bbox: BoundingBox): HTMLCanvasElement => {
      const canvas = document.createElement('canvas')
      const scale = Math.max(1, 200 / bbox.width)
      canvas.width = Math.round(bbox.width * scale)
      canvas.height = Math.round(bbox.height * scale)
      const ctx = canvas.getContext('2d')!
      ctx.drawImage(video, bbox.x, bbox.y, bbox.width, bbox.height, 0, 0, canvas.width, canvas.height)
      return canvas
    },
    [],
  )

  const readPlateBackend = useCallback(
    async (video: HTMLVideoElement, bbox: BoundingBox): Promise<PlateCache | null> => {
      try {
        const canvas = cropVehicle(video, bbox)
        const blob = await new Promise<Blob | null>(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.9))
        if (!blob) return null
        const form = new FormData()
        form.append('image', blob, 'vehicle.jpg')
        const res = await fetch(`${backendUrl}/api/plate`, {
          method: 'POST',
          body: form,
          signal: AbortSignal.timeout(3000),
        })
        if (!res.ok) return null
        const data = await res.json() as { plate?: string; confidence?: number }
        const plate = cleanPlateText(data.plate ?? '')
        if (plate.length < 4) return null
        return { plate, confidence: data.confidence ?? 0.5, ts: Date.now() }
      } catch {
        return null
      }
    },
    [backendUrl, cropVehicle],
  )

  const readPlateTesseract = useCallback(
    async (video: HTMLVideoElement, bbox: BoundingBox): Promise<PlateCache | null> => {
      const worker = tesseractRef.current
      if (!worker) return null
      try {
        // Crop précis sur la zone plaque + preprocessing
        const canvas = cropPlateArea(video, bbox)
        const { data } = await worker.recognize(canvas)
        const plate = cleanPlateText(data.text)
        if (plate.length < 4) return null
        return { plate, confidence: data.confidence / 100, ts: Date.now() }
      } catch {
        return null
      }
    },
    [cropPlateArea],
  )

  const readPlate = useCallback(
    async (video: HTMLVideoElement, bbox: BoundingBox): Promise<PlateCache | null> => {
      // Essaie le backend en premier, tombe sur Tesseract.js sinon
      if (backendOkRef.current) {
        const result = await readPlateBackend(video, bbox)
        if (result) return result
      }
      return readPlateTesseract(video, bbox)
    },
    [readPlateBackend, readPlateTesseract],
  )

  const matchOrCreateTrack = useCallback(
    (bbox: BoundingBox, cx: number, cy: number, now: number): TrackedVehicle => {
      let bestMatch: TrackedVehicle | null = null
      let bestIou = IOU_MATCH_THRESHOLD

      for (const track of tracksRef.current.values()) {
        // Expire les pistes trop vieilles
        if (now - track.ts > 3000) continue
        const trackBbox: BoundingBox = {
          x: track.cx - bbox.width / 2,
          y: track.cy - bbox.height / 2,
          width: bbox.width,
          height: bbox.height,
        }
        const score = iou(bbox, trackBbox)
        if (score > bestIou) {
          bestIou = score
          bestMatch = track
        }
      }

      if (bestMatch) {
        bestMatch.cx = cx
        bestMatch.cy = cy
        bestMatch.ts = now
        return bestMatch
      }

      const newTrack: TrackedVehicle = { id: `v${nextIdRef.current++}`, cx, cy, ts: now }
      tracksRef.current.set(newTrack.id, newTrack)
      return newTrack
    },
    [],
  )

  const estimateSpeed = useCallback(
    (track: TrackedVehicle, cx: number, cy: number, now: number): number | undefined => {
      const dt = (now - track.ts) / 1000
      if (dt < 0.05 || dt > 3) return undefined
      const dpx = Math.hypot(cx - track.cx, cy - track.cy)
      const metersPerPixel = 0.005
      const speedMs = (dpx * metersPerPixel) / dt
      return Math.min(speedMs * 3.6, 300)
    },
    [],
  )

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

      modelRef.current.detect(video).then(async predictions => {
        if (!runningRef.current) return

        const ctx = canvas.getContext('2d')
        if (!ctx) return

        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
        ctx.clearRect(0, 0, canvas.width, canvas.height)

        const vehicles = predictions.filter(p => VEHICLE_CLASSES.has(p.class))
        const sendPlates = now - lastPlateSendRef.current > PLATE_SEND_INTERVAL_MS
        if (sendPlates) lastPlateSendRef.current = now

        // Nettoyage des pistes expirées
        for (const [id, track] of tracksRef.current.entries()) {
          if (now - track.ts > 5000) tracksRef.current.delete(id)
        }

        const results: Detection[] = await Promise.all(
          vehicles.map(async pred => {
            const [bx, by, bw, bh] = pred.bbox
            const bbox: BoundingBox = { x: bx, y: by, width: bw, height: bh }
            const cx = bx + bw / 2
            const cy = by + bh / 2

            const track = matchOrCreateTrack(bbox, cx, cy, now)
            const speed = estimateSpeed(track, cx, cy, now)

            // Met à jour la piste avec la vitesse
            track.cx = cx
            track.cy = cy
            track.ts = now

            // Lecture plaque : seulement si véhicule assez grand et pas trop récente
            let plateCache = track.plate
            const plateExpired = !plateCache || (now - plateCache.ts > PLATE_CACHE_TTL_MS)

            if (sendPlates && plateExpired && bw > 80 && bh > 60) {
              const result = await readPlate(video, bbox)
              if (result) {
                track.plate = result
                plateCache = result
              }
            }

            // === Dessin ===
            const highConf = pred.score > 0.7
            ctx.strokeStyle = highConf ? '#3b82f6' : '#f59e0b'
            ctx.lineWidth = 2
            ctx.strokeRect(bx, by, bw, bh)

            // Zone plaque en surbrillance (debug visuel)
            const pzX = bx + bw * 0.1
            const pzY = by + bh * 0.6
            const pzW = bw * 0.8
            const pzH = bh * 0.38
            ctx.strokeStyle = 'rgba(34,197,94,0.6)'
            ctx.lineWidth = 1
            ctx.setLineDash([4, 3])
            ctx.strokeRect(pzX, pzY, pzW, pzH)
            ctx.setLineDash([])

            ctx.font = 'bold 13px Inter, system-ui, sans-serif'
            const label = `${pred.class} ${Math.round(pred.score * 100)}%`
            const lw = ctx.measureText(label).width + 10
            ctx.fillStyle = highConf ? '#3b82f6' : '#f59e0b'
            ctx.fillRect(bx, by - 22, lw, 22)
            ctx.fillStyle = '#fff'
            ctx.fillText(label, bx + 5, by - 6)

            if (plateCache?.plate) {
              const plateLabel = `🔢 ${plateCache.plate}`
              const pw = ctx.measureText(plateLabel).width + 10
              ctx.fillStyle = '#22c55e'
              ctx.fillRect(bx, by + bh + 2, pw, 22)
              ctx.fillStyle = '#fff'
              ctx.fillText(plateLabel, bx + 5, by + bh + 16)
            }

            if (speed !== undefined) {
              ctx.font = 'bold 12px Inter, system-ui, sans-serif'
              const speedLabel = `${Math.round(speed)} km/h`
              const sw = ctx.measureText(speedLabel).width + 10
              ctx.fillStyle = speed > 80 ? '#ef4444' : '#f59e0b'
              ctx.fillRect(bx + bw - sw, by - 22, sw, 22)
              ctx.fillStyle = '#fff'
              ctx.fillText(speedLabel, bx + bw - sw + 5, by - 6)
            }

            return {
              id: track.id,
              bbox,
              class: pred.class,
              score: pred.score,
              plate: plateCache?.plate,
              plateConfidence: plateCache?.confidence,
              speedKmh: speed,
              timestamp: Date.now(),
            }
          }),
        )

        onDetection(results)
        animRef.current = requestAnimationFrame(() => runDetection(video, canvas))
      }).catch(() => {
        animRef.current = requestAnimationFrame(() => runDetection(video, canvas))
      })
    },
    [matchOrCreateTrack, estimateSpeed, readPlate, onDetection],
  )

  const start = useCallback(
    (video: HTMLVideoElement, canvas: HTMLCanvasElement) => {
      runningRef.current = true
      runDetection(video, canvas)
    },
    [runDetection],
  )

  const stop = useCallback(() => {
    runningRef.current = false
    tracksRef.current.clear()
    if (animRef.current) cancelAnimationFrame(animRef.current)
  }, [])

  return { modelReady, loading, tesseractReady, start, stop }
}
