"""
utils.py
--------
Utility functions: coordinate smoothing, interpolation, distance helpers,
OpenCV drawing helpers, and glitter-effect generator.
"""

import cv2
import numpy as np
import random
import math


# ---------------------------------------------------------------------------
# Smoothing / Interpolation
# ---------------------------------------------------------------------------

class SmoothCursor:
    """
    Exponential moving-average smoother for (x, y) coordinates.
    Reduces jitter while keeping cursor responsive.
    """

    def __init__(self, alpha=0.25):
        """
        alpha: smoothing factor (0 = frozen, 1 = no smoothing).
        Lower alpha → smoother but more lag. 0.20-0.30 is a good range.
        """
        self.alpha = alpha
        self.sx = None
        self.sy = None

    def update(self, x, y):
        """Feed a new raw coordinate and return the smoothed value."""
        if self.sx is None:
            self.sx, self.sy = float(x), float(y)
        else:
            self.sx = self.alpha * x + (1 - self.alpha) * self.sx
            self.sy = self.alpha * y + (1 - self.alpha) * self.sy
        return int(self.sx), int(self.sy)

    def reset(self):
        self.sx = self.sy = None


def interpolate(value, in_min, in_max, out_min, out_max):
    """
    Map a value from one range to another (like Arduino's map()).
    Clamps output to [out_min, out_max].
    """
    value   = max(in_min, min(in_max, value))
    ratio   = (value - in_min) / (in_max - in_min + 1e-6)
    result  = out_min + ratio * (out_max - out_min)
    return float(np.clip(result, min(out_min, out_max), max(out_min, out_max)))


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------

def euclidean(p1, p2):
    """Return Euclidean distance between two (x, y) tuples."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


# ---------------------------------------------------------------------------
# OpenCV drawing helpers
# ---------------------------------------------------------------------------

def draw_rounded_rect(img, x1, y1, x2, y2, radius, color, thickness=-1, alpha=1.0):
    """
    Draw a filled or outlined rectangle with rounded corners on img.
    Supports per-call transparency via alpha blend onto a copy.
    """
    overlay = img.copy()
    # Four corner circles
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, thickness)
    # Fill rectangles
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    else:
        np.copyto(img, overlay)


def put_text_shadow(img, text, pos, font, scale, color, thickness=1, shadow_color=(0, 0, 0)):
    """Draw text with a subtle drop shadow for readability on any background."""
    ox, oy = pos
    cv2.putText(img, text, (ox + 1, oy + 1), font, scale, shadow_color, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, pos,              font, scale, color,        thickness,     cv2.LINE_AA)


def draw_gradient_rect(img, x1, y1, x2, y2, color_top, color_bottom):
    """Fill a rectangle with a vertical gradient between two BGR colors."""
    for row in range(y1, y2):
        t = (row - y1) / max(y2 - y1 - 1, 1)
        b = tuple(int(color_top[i] * (1 - t) + color_bottom[i] * t) for i in range(3))
        cv2.line(img, (x1, row), (x2, row), b, 1)


# ---------------------------------------------------------------------------
# Glitter / sparkle effect
# ---------------------------------------------------------------------------

class GlitterEffect:
    """
    Maintains a pool of sparkle particles that decay over time.
    Call `spawn(x, y)` each frame when drawing, then `render(canvas)` to paint.
    """

    def __init__(self, max_particles=120, lifespan=18):
        self.particles   = []   # Each: [x, y, life, max_life, size, color]
        self.max_p       = max_particles
        self.lifespan    = lifespan
        self.COLORS      = [
            (255, 220, 80),   # Gold
            (255, 255, 255),  # White
            (80, 200, 255),   # Cyan-blue
            (255, 120, 220),  # Pink
            (150, 255, 150),  # Mint
        ]

    def spawn(self, x, y):
        """Spawn a burst of sparkles near (x, y)."""
        count = random.randint(3, 7)
        for _ in range(count):
            if len(self.particles) >= self.max_p:
                break
            ox = x + random.randint(-12, 12)
            oy = y + random.randint(-12, 12)
            life  = random.randint(self.lifespan // 2, self.lifespan)
            size  = random.randint(2, 5)
            color = random.choice(self.COLORS)
            self.particles.append([ox, oy, life, life, size, color])

    def render(self, canvas):
        """Draw all alive particles onto canvas and age them."""
        alive = []
        for p in self.particles:
            x, y, life, max_life, size, color = p
            if life > 0:
                alpha_ratio = life / max_life
                # Fade color as particle ages
                faded = tuple(int(c * alpha_ratio) for c in color)
                cv2.circle(canvas, (x, y), size, faded, -1, cv2.LINE_AA)
                p[2] -= 1
                alive.append(p)
        self.particles = alive

    def clear(self):
        self.particles = []


# ---------------------------------------------------------------------------
# Click debouncer
# ---------------------------------------------------------------------------

class ClickDebounce:
    """
    Prevents accidental rapid-fire clicks by enforcing a minimum interval
    between successive click events (frame-count based).
    """

    def __init__(self, cooldown_frames=20):
        self.cooldown = cooldown_frames
        self.counter  = 0

    def ready(self):
        """Return True if a click is allowed right now."""
        if self.counter == 0:
            return True
        self.counter -= 1
        return False

    def trigger(self):
        """Call after a successful click to start the cooldown."""
        self.counter = self.cooldown