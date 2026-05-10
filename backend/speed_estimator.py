"""
Estimation de vitesse par optical flow (Lucas-Kanade).
Principe : compare deux frames successives pour estimer le déplacement
en pixels, convertit en km/h avec une calibration approximative.

Note : sans calibration fixe (hauteur caméra, angle, focale),
la précision est indicative (~±30%). Pour une mesure certifiée,
il faudrait un radar homologué.
"""
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


# Facteur de conversion pixels/s → km/h (calibration par défaut)
# Suppose : caméra à ~3m de hauteur, angle 30°, route à ~10m
# 1% de la largeur de frame ≈ ~0.3m réels → ajuster selon contexte
DEFAULT_METERS_PER_PIXEL_FRACTION = 0.003  # à 1280px de large


@dataclass
class VehicleTrack:
    vehicle_id: str
    speed_samples: deque = field(default_factory=lambda: deque(maxlen=8))
    last_frame: Optional[np.ndarray] = None
    last_time: float = field(default_factory=time.time)
    last_cx: float = 0.0
    last_cy: float = 0.0

    @property
    def smoothed_speed(self) -> Optional[float]:
        if len(self.speed_samples) < 2:
            return None
        arr = np.array(self.speed_samples)
        # Médiane pour résistance aux outliers
        return float(np.median(arr))


class SpeedEstimator:
    def __init__(self, frame_width: int = 1280, meters_per_pixel_fraction: float = DEFAULT_METERS_PER_PIXEL_FRACTION):
        self.frame_width = frame_width
        self.mppf = meters_per_pixel_fraction
        self._tracks: dict[str, VehicleTrack] = {}

    def _meters_per_pixel(self) -> float:
        return self.frame_width * self.mppf / self.frame_width

    def update(
        self,
        vehicle_id: str,
        frame_gray: np.ndarray,
        cx: float,
        cy: float,
        bbox_w: float,
        bbox_h: float,
    ) -> Optional[float]:
        now = time.time()
        mpp = self._meters_per_pixel()

        if vehicle_id not in self._tracks:
            self._tracks[vehicle_id] = VehicleTrack(
                vehicle_id=vehicle_id,
                last_frame=frame_gray.copy(),
                last_time=now,
                last_cx=cx,
                last_cy=cy,
            )
            return None

        track = self._tracks[vehicle_id]
        dt = now - track.last_time

        if dt < 0.05 or dt > 5.0:
            track.last_frame = frame_gray.copy()
            track.last_time = now
            track.last_cx = cx
            track.last_cy = cy
            return track.smoothed_speed

        # Optical flow sur la région du véhicule
        roi_x1 = max(0, int(cx - bbox_w * 0.6))
        roi_y1 = max(0, int(cy - bbox_h * 0.6))
        roi_x2 = min(frame_gray.shape[1], int(cx + bbox_w * 0.6))
        roi_y2 = min(frame_gray.shape[0], int(cy + bbox_h * 0.6))

        if roi_x2 - roi_x1 < 20 or roi_y2 - roi_y1 < 20:
            # ROI trop petite : fall back sur centroïde
            dpx = np.hypot(cx - track.last_cx, cy - track.last_cy)
            speed_ms = (dpx * mpp) / dt
        else:
            prev_roi = track.last_frame[roi_y1:roi_y2, roi_x1:roi_x2]
            curr_roi = frame_gray[roi_y1:roi_y2, roi_x1:roi_x2]

            try:
                prev_pts = cv2.goodFeaturesToTrack(
                    prev_roi, maxCorners=50, qualityLevel=0.01, minDistance=5,
                )
                if prev_pts is not None and len(prev_pts) >= 3:
                    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_roi, curr_roi, prev_pts, None)
                    if status is not None:
                        good_prev = prev_pts[status.flatten() == 1]
                        good_next = next_pts[status.flatten() == 1]
                        if len(good_next) >= 2:
                            flow = good_next - good_prev
                            median_flow = np.median(np.linalg.norm(flow, axis=1))
                            speed_ms = (median_flow * mpp) / dt
                        else:
                            speed_ms = 0.0
                    else:
                        speed_ms = 0.0
                else:
                    dpx = np.hypot(cx - track.last_cx, cy - track.last_cy)
                    speed_ms = (dpx * mpp) / dt
            except cv2.error:
                dpx = np.hypot(cx - track.last_cx, cy - track.last_cy)
                speed_ms = (dpx * mpp) / dt

        speed_kmh = min(speed_ms * 3.6, 250.0)
        track.speed_samples.append(speed_kmh)
        track.last_frame = frame_gray.copy()
        track.last_time = now
        track.last_cx = cx
        track.last_cy = cy

        return track.smoothed_speed

    def cleanup_stale(self, max_age_s: float = 3.0) -> None:
        now = time.time()
        stale = [k for k, v in self._tracks.items() if now - v.last_time > max_age_s]
        for k in stale:
            del self._tracks[k]
