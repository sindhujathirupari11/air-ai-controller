"""
utils.py
--------
Utility functions shared across the Air AI Controller.

Why NumPy here?
  - np.interp  → fast linear interpolation for coordinate remapping
  - np.hypot   → vectorised Euclidean distance
  - np.clip    → clean clamping without if/else chains
  - ndarray ops → in-place pixel blending for glitter effect

Why not OpenCV for maths?
  OpenCV is great for image ops; NumPy is more ergonomic for scalar maths
  like distance and smoothing — no unnecessary overhead.
"""

import math
import random
import time
from collections import deque

import cv2
import numpy as np


# ─────────────────────────────────────────────
#  Distance helpers
# ─────────────────────────────────────────────

def euclidean_distance(p1: tuple, p2: tuple) -> float:
    """Straight-line pixel distance between two (x,y) points."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def midpoint(p1: tuple, p2: tuple) -> tuple[int, int]:
    """Integer midpoint between two (x,y) points."""
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)


# ─────────────────────────────────────────────
#  Smoothing / jitter reduction
# ─────────────────────────────────────────────

class SmoothFilter:
    """
    Exponential moving average (EMA) smoother for x,y mouse coords.
    
    EMA formula:  smoothed = alpha * current + (1 - alpha) * previous
    
    Why EMA over a simple rolling average?
      - EMA is O(1) memory and O(1) compute per frame.
      - It reacts quickly to deliberate movement (alpha ~ 0.2–0.4 is a
        good balance) while dampening high-frequency jitter.
      - A rolling average introduces lag proportional to window size.
    
    Args:
        alpha: Smoothing factor in (0, 1].
               0 → never moves (infinite lag), 1 → no smoothing (raw).
               Default 0.25 gives smooth cursor without noticeable lag.
    """

    def __init__(self, alpha: float = 0.25):
        self.alpha   = alpha
        self.prev_x  = None
        self.prev_y  = None

    def smooth(self, x: float, y: float) -> tuple[float, float]:
        if self.prev_x is None:
            self.prev_x, self.prev_y = x, y
        sx = self.alpha * x + (1 - self.alpha) * self.prev_x
        sy = self.alpha * y + (1 - self.alpha) * self.prev_y
        self.prev_x, self.prev_y = sx, sy
        return sx, sy

    def reset(self):
        self.prev_x = self.prev_y = None


# ─────────────────────────────────────────────
#  Coordinate mapping
# ─────────────────────────────────────────────

def map_to_screen(px: float, py: float,
                  cam_w: int, cam_h: int,
                  screen_w: int, screen_h: int,
                  margin: int = 100) -> tuple[int, int]:
    """
    Map a camera pixel (px, py) to a screen pixel (sx, sy).
    
    We shrink the active camera region by `margin` pixels on each side
    so reaching screen edges is comfortable without moving to camera corners.
    
    np.interp does the linear interpolation:
        output = out_min + (value - in_min) / (in_max - in_min) * (out_max - out_min)
    then np.clip ensures we never go out of screen bounds.
    """
    sx = int(np.interp(px, [margin, cam_w - margin], [0, screen_w]))
    sy = int(np.interp(py, [margin, cam_h - margin], [0, screen_h]))
    sx = int(np.clip(sx, 0, screen_w - 1))
    sy = int(np.clip(sy, 0, screen_h - 1))
    return sx, sy


# ─────────────────────────────────────────────
#  Volume level mapping
# ─────────────────────────────────────────────

def distance_to_volume(dist: float,
                        min_dist: float = 30,
                        max_dist: float = 200) -> int:
    """
    Convert thumb-index distance (pixels) to a volume percentage [0, 100].
    
    np.interp handles the linear mapping; clip guards against finger
    positions outside the expected range.
    """
    vol = int(np.interp(dist, [min_dist, max_dist], [0, 100]))
    return int(np.clip(vol, 0, 100))


# ─────────────────────────────────────────────
#  Click debounce
# ─────────────────────────────────────────────

class ClickDebounce:
    """
    Prevents accidental rapid-fire clicks.
    
    A click is registered only if at least `cooldown` seconds have elapsed
    since the last registered click. Simple time-based guard.
    """

    def __init__(self, cooldown: float = 0.4):
        self.cooldown   = cooldown
        self.last_click = 0.0

    def can_click(self) -> bool:
        now = time.time()
        if now - self.last_click >= self.cooldown:
            self.last_click = now
            return True
        return False


# ─────────────────────────────────────────────
#  Drawing canvas helpers
# ─────────────────────────────────────────────

def draw_rounded_rect(img: np.ndarray, pt1: tuple, pt2: tuple,
                      color: tuple, thickness: int = -1,
                      radius: int = 12) -> None:
    """
    Draw a rectangle with rounded corners on `img` (in-place).
    
    OpenCV has no native rounded-rect function, so we decompose it into
    four arcs + three filled rectangles (or three stroked segments).
    Used for UI panels so they feel modern rather than boxy.
    """
    x1, y1 = pt1
    x2, y2 = pt2

    if thickness == -1:  # filled
        # Fill three rectangles to cover interior
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        # Four corner circles
        for cx, cy in [(x1 + radius, y1 + radius),
                       (x2 - radius, y1 + radius),
                       (x1 + radius, y2 - radius),
                       (x2 - radius, y2 - radius)]:
            cv2.circle(img, (cx, cy), radius, color, -1)
    else:
        # Outline only
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
        for cx, cy in [(x1 + radius, y1 + radius),
                       (x2 - radius, y1 + radius),
                       (x1 + radius, y2 - radius),
                       (x2 - radius, y2 - radius)]:
            cv2.ellipse(img, (cx, cy), (radius, radius), 0, 0, 0, color, thickness)


def overlay_alpha(bg: np.ndarray, overlay: np.ndarray,
                  x: int, y: int, alpha: float = 0.6) -> None:
    """
    Blend `overlay` onto `bg` at position (x, y) with given alpha.
    
    Uses NumPy in-place ops for speed — avoids creating temporary arrays
    by using cv2.addWeighted on the ROI slice only.
    """
    h, w = overlay.shape[:2]
    roi = bg[y:y+h, x:x+w]
    if roi.shape[:2] != (h, w):
        return
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)


# ─────────────────────────────────────────────
#  Glitter / sparkle effect
# ─────────────────────────────────────────────

class GlitterEffect:
    """
    Renders animated sparkle particles that follow the drawing cursor.
    
    Each frame we:
      1. Spawn N new particles near the current tip position.
      2. Age existing particles (reduce life, drift position randomly).
      3. Draw each live particle as a small filled circle with fading opacity.
    
    Stored as a deque so old particles are automatically evicted — O(1) append/pop.
    """

    def __init__(self, max_particles: int = 120):
        self.max_particles = max_particles
        # Each particle: [x, y, radius, life (0–1), r, g, b]
        self.particles: deque = deque(maxlen=max_particles)

    def emit(self, x: int, y: int, color: tuple, intensity: int = 1):
        """Spawn `intensity` new sparkles near (x, y)."""
        for _ in range(intensity):
            jx = x + random.randint(-12, 12)
            jy = y + random.randint(-12, 12)
            r  = random.randint(2, 6)
            # Random tint blended with base drawing color
            rc = min(255, color[2] + random.randint(-40, 40))
            gc = min(255, color[1] + random.randint(-40, 40))
            bc = min(255, color[0] + random.randint(-40, 40))
            self.particles.append([jx, jy, r, 1.0, rc, gc, bc])

    def update_and_draw(self, frame: np.ndarray) -> None:
        """Age particles and render them onto `frame`."""
        alive = []
        for p in self.particles:
            p[3] -= 0.06        # age: reduce life
            p[0] += random.randint(-2, 2)   # drift x
            p[1] += random.randint(-2, 1)   # drift y (slight upward)
            if p[3] > 0:
                alive.append(p)
                alpha_val = p[3]            # life doubles as alpha
                rad  = max(1, int(p[2] * p[3]))
                color = (int(p[6] * alpha_val),
                         int(p[5] * alpha_val),
                         int(p[4] * alpha_val))
                cv2.circle(frame, (int(p[0]), int(p[1])), rad, color, -1)

        # Replace deque content with only alive particles
        self.particles.clear()
        self.particles.extend(alive)


# ─────────────────────────────────────────────
#  FPS counter
# ─────────────────────────────────────────────

class FPSCounter:
    """Rolling FPS estimate over a short window."""

    def __init__(self, window: int = 20):
        self.times: deque = deque(maxlen=window)

    def tick(self) -> float:
        self.times.append(time.time())
        if len(self.times) < 2:
            return 0.0
        return (len(self.times) - 1) / (self.times[-1] - self.times[0])
