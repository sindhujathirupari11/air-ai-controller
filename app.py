"""
app.py  —  Air AI Controller
─────────────────────────────
Entry point. Orchestrates camera capture, hand tracking, gesture
recognition, mode switching, and UI rendering.

Technology choices (summary):
  cv2 (OpenCV) — camera capture, all drawing, window management
  mediapipe    — real-time 21-landmark hand detection on CPU
  pyautogui    — cross-platform mouse move / click / scroll
  numpy        — fast coordinate interpolation and array maths
  utils.py     — smoothing (EMA), distance, glitter, FPS
  hand_tracker.py — MediaPipe wrapper + gesture helpers

Modes:
  MOUSE   — index finger moves cursor; pinch (thumb+index) = click
  DRAW    — index-only draws on a persistent canvas layer
            press 'd' to toggle, or pinch two specific fingers

Controls (keyboard):
  d        — toggle DRAW / MOUSE
  c        — clear drawing canvas
  q / ESC  — quit
"""

import sys
import time

import cv2
import numpy as np
import pyautogui

from hand_tracker import HandTracker
from utils import (
    SmoothFilter, ClickDebounce, FPSCounter, GlitterEffect,
    euclidean_distance, midpoint, map_to_screen,
    distance_to_volume, draw_rounded_rect, overlay_alpha,
)

# ─── Safety: pyautogui will raise exception on mouse fail rather than crash ──
pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0           # remove built-in 0.1s delay for low latency

# ─── Screen dimensions ───────────────────────────────────────────────────────
SCREEN_W, SCREEN_H = pyautogui.size()

# ─── Camera / window settings ────────────────────────────────────────────────
CAM_W,  CAM_H  = 1280, 720      # request this resolution from camera
WIN_W,  WIN_H  = 1280, 720      # display window size (mirrors cam)
MARGIN         = 120            # deadzone margin for coordinate mapping

# ─── Gesture thresholds (pixels) ─────────────────────────────────────────────
CLICK_DIST        = 38          # thumb-index distance to trigger click
VOLUME_MIN_DIST   = 30
VOLUME_MAX_DIST   = 220

# ─── Drawing presets ─────────────────────────────────────────────────────────
COLORS = {
    "White"  : (255, 255, 255),
    "Red"    : (50,  50,  240),
    "Green"  : (50, 200,  50),
    "Blue"   : (230, 100,  30),
    "Yellow" : (30, 220, 220),
    "Purple" : (200,  60, 180),
    "Cyan"   : (220, 200,  30),
    "Eraser" : (0,   0,   0),
}
COLOR_NAMES  = list(COLORS.keys())
BRUSH_WIDTHS = [3, 6, 10, 16, 24]      # selectable stroke widths

# ─── Modes ────────────────────────────────────────────────────────────────────
MODE_MOUSE = "MOUSE"
MODE_DRAW  = "DRAW"


# ══════════════════════════════════════════════════════════════════════════════
#  UI renderer  (draws HUD panels onto a frame)
# ══════════════════════════════════════════════════════════════════════════════

class UI:
    """
    All on-screen HUD drawing lives here to keep app.py readable.
    Drawn directly onto the BGR frame using OpenCV primitives.
    """

    # Fonts
    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    FONT_BOLD  = cv2.FONT_HERSHEY_DUPLEX

    # Palette
    BG_DARK    = (18, 18, 28)
    ACCENT     = (100, 220, 255)       # cyan-ish
    ACCENT2    = (180, 100, 255)       # purple
    TEXT_LIGHT = (230, 230, 240)
    RED        = (60,  60, 240)
    GREEN      = (60, 210,  80)
    PANEL_BG   = (30, 30, 46)

    # ── color swatches (top bar) ──────────────────────────────────────────────

    @staticmethod
    def draw_color_bar(frame: np.ndarray, selected_color: str,
                       selected_width: int, glitter_level: int) -> dict:
        """
        Draw the top toolbar: color swatches + brush width + glitter selector.
        Returns a dict mapping color name → (x1,y1,x2,y2) for hit-testing.
        """
        h, w = frame.shape[:2]
        bar_h = 64

        # Semi-transparent panel
        panel = np.full((bar_h, w, 3), UI.PANEL_BG, dtype=np.uint8)
        overlay_alpha(frame, panel, 0, 0, alpha=0.85)

        # Border line
        cv2.line(frame, (0, bar_h), (w, bar_h), UI.ACCENT, 1)

        hit_zones = {}

        # ── color swatches ────────────────────────────────────────────────────
        sw = 36; gap = 6; ox = 14; oy = 14
        for i, name in enumerate(COLOR_NAMES):
            color_bgr = COLORS[name]
            x1 = ox + i * (sw + gap)
            y1 = oy
            x2 = x1 + sw
            y2 = y1 + sw

            # Draw swatch
            if name == "Eraser":
                cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 80), -1)
                cv2.putText(frame, "E", (x1 + 10, y2 - 9),
                            UI.FONT, 0.5, UI.TEXT_LIGHT, 1)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, -1)

            # Highlight selected
            border_col = UI.ACCENT if name == selected_color else (80, 80, 100)
            thick = 3 if name == selected_color else 1
            cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1),
                          border_col, thick)

            hit_zones[name] = (x1, y1, x2, y2)

        # ── brush width selector ──────────────────────────────────────────────
        bx = ox + len(COLOR_NAMES) * (sw + gap) + 20
        cv2.putText(frame, "W:", (bx, 42), UI.FONT, 0.45, UI.ACCENT, 1)
        bx += 28
        for i, bw in enumerate(BRUSH_WIDTHS):
            cx_ = bx + i * 36 + 14
            cy_ = 32
            col = UI.ACCENT if bw == selected_width else (80, 80, 100)
            cv2.circle(frame, (cx_, cy_), bw // 2 + 3, col, -1)
            hit_zones[f"W{bw}"] = (cx_ - 14, cy_ - 20, cx_ + 14, cy_ + 20)

        # ── glitter selector ──────────────────────────────────────────────────
        gx = bx + len(BRUSH_WIDTHS) * 36 + 16
        cv2.putText(frame, "Glitter:", (gx, 26), UI.FONT, 0.42, UI.ACCENT2, 1)
        for i, level in enumerate([0, 1, 2, 3]):
            rx1 = gx + i * 30
            ry1 = 32
            rx2 = rx1 + 22
            ry2 = ry1 + 16
            col = UI.ACCENT2 if level == glitter_level else (60, 60, 80)
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), col, -1)
            label = "✦" if level > 0 else "○"
            cv2.putText(frame, str(level), (rx1 + 6, ry2 - 3),
                        UI.FONT, 0.38, UI.TEXT_LIGHT, 1)
            hit_zones[f"G{level}"] = (rx1, ry1, rx2, ry2)

        return hit_zones

    # ── mode badge ────────────────────────────────────────────────────────────

    @staticmethod
    def draw_mode_badge(frame: np.ndarray, mode: str) -> None:
        h, w = frame.shape[:2]
        label = f"  {mode} MODE  "
        col   = UI.GREEN if mode == MODE_DRAW else UI.ACCENT
        # Draw filled pill
        (tw, th), _ = cv2.getTextSize(label, UI.FONT_BOLD, 0.65, 2)
        x1, y1 = w - tw - 30, h - 44
        x2, y2 = w - 10, h - 14
        draw_rounded_rect(frame, (x1, y1), (x2, y2), col, thickness=-1, radius=10)
        cv2.putText(frame, label, (x1 + 6, y2 - 8), UI.FONT_BOLD, 0.55,
                    UI.BG_DARK, 2)

    # ── FPS + gesture info ────────────────────────────────────────────────────

    @staticmethod
    def draw_info(frame: np.ndarray, fps: float, vol: int,
                  hand_visible: bool, clicking: bool) -> None:
        h, w = frame.shape[:2]
        # Left side info panel
        lines = [
            f"FPS: {fps:.0f}",
            f"Vol: {vol}%",
            f"Hand: {'YES' if hand_visible else 'NO'}",
            f"Click: {'●' if clicking else '○'}",
        ]
        oy = h - 110
        for i, ln in enumerate(lines):
            col = UI.ACCENT if i < 2 else (UI.GREEN if "YES" in ln or "●" in ln else UI.TEXT_LIGHT)
            cv2.putText(frame, ln, (14, oy + i * 22), UI.FONT, 0.52, col, 1)

    # ── volume bar ────────────────────────────────────────────────────────────

    @staticmethod
    def draw_volume_bar(frame: np.ndarray, vol: int) -> None:
        h, w = frame.shape[:2]
        bar_x, bar_y = 16, 80
        bar_h_total  = h - 200
        filled_h     = int(bar_h_total * vol / 100)

        # Background
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + 18, bar_y + bar_h_total), (40, 40, 60), -1)
        # Fill
        fy = bar_y + bar_h_total - filled_h
        color = UI.ACCENT if vol < 80 else UI.RED
        cv2.rectangle(frame, (bar_x, fy),
                      (bar_x + 18, bar_y + bar_h_total), color, -1)
        cv2.putText(frame, "VOL", (bar_x - 2, bar_y - 6),
                    UI.FONT, 0.38, UI.ACCENT, 1)

    # ── shortcut hints ────────────────────────────────────────────────────────

    @staticmethod
    def draw_hints(frame: np.ndarray) -> None:
        hints = ["d: toggle mode", "c: clear", "q: quit"]
        h, w  = frame.shape[:2]
        for i, hint in enumerate(hints):
            cv2.putText(frame, hint, (w - 150, h - 80 + i * 20),
                        UI.FONT, 0.38, (100, 100, 130), 1)

    # ── finger gesture indicators ─────────────────────────────────────────────

    @staticmethod
    def draw_finger_dots(frame: np.ndarray, fingers: list) -> None:
        names = ["T", "I", "M", "R", "P"]
        ox    = 50
        for i, (name, up) in enumerate(zip(names, fingers)):
            col = UI.GREEN if up else (60, 60, 80)
            cv2.circle(frame, (ox + i * 22, frame.shape[0] - 24), 8, col, -1)
            cv2.putText(frame, name, (ox + i * 22 - 5, frame.shape[0] - 16),
                        UI.FONT, 0.32, UI.BG_DARK, 1)


# ══════════════════════════════════════════════════════════════════════════════
#  Welcome / splash screen
# ══════════════════════════════════════════════════════════════════════════════

def draw_splash(frame: np.ndarray, cam_index: int) -> np.ndarray:
    """
    Render the welcome screen overlay on a camera preview frame.
    Shows title, instructions, and a "PRESS ENTER to Start" prompt.
    """
    h, w = frame.shape[:2]

    # Dim the camera preview
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Title
    title      = "AIR AI CONTROLLER"
    sub        = "Gesture-Driven Mouse, Volume & Air Drawing"
    start_hint = "  PRESS  ENTER  TO  START  "
    quit_hint  = "Press Q to quit"

    # Animated pulse border
    pulse = int(abs(math.sin(time.time() * 2) * 80) + 40)
    cv2.rectangle(frame, (40, 40), (w - 40, h - 40),
                  (pulse, pulse // 2, 255 - pulse // 2), 2)

    # Title text
    (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 1.6, 3)
    cv2.putText(frame, title, ((w - tw) // 2, h // 2 - 80),
                cv2.FONT_HERSHEY_DUPLEX, 1.6, (100, 220, 255), 3)

    (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
    cv2.putText(frame, sub, ((w - sw) // 2, h // 2 - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 200), 1)

    # Feature list
    features = [
        "✦  Index finger  →  move mouse",
        "✦  Pinch (thumb + index)  →  click",
        "✦  Spread thumb + index  →  control volume",
        "✦  [D]  →  toggle air drawing mode",
        "✦  [C]  →  clear canvas",
    ]
    fy = h // 2 + 10
    for feat in features:
        (fw, _), _ = cv2.getTextSize(feat, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.putText(frame, feat, ((w - fw) // 2, fy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 200, 160), 1)
        fy += 26

    # Start button
    bp = int(abs(math.sin(time.time() * 3) * 30))
    btn_col = (40 + bp, 100 + bp, 40 + bp)
    (bw, _), _ = cv2.getTextSize(start_hint, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
    bx = (w - bw) // 2 - 14
    by = fy + 20
    draw_rounded_rect(frame, (bx, by - 26), (bx + bw + 28, by + 10),
                      btn_col, thickness=-1, radius=10)
    cv2.putText(frame, start_hint, (bx + 14, by),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 255, 200), 2)

    (qw, _), _ = cv2.getTextSize(quit_hint, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.putText(frame, quit_hint, ((w - qw) // 2, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 120), 1)

    return frame


# Import math for splash (sin)
import math


# ══════════════════════════════════════════════════════════════════════════════
#  Main application
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("   AIR AI CONTROLLER  —  starting up")
    print("=" * 60)

    # ── init camera ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check device permissions.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 60)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"   Camera: {actual_w}x{actual_h}")
    print(f"   Screen: {SCREEN_W}x{SCREEN_H}")

    # ── init modules ─────────────────────────────────────────────────────────
    tracker  = HandTracker(detection_confidence=0.80, tracking_confidence=0.75)
    smoother = SmoothFilter(alpha=0.25)
    debounce = ClickDebounce(cooldown=0.35)
    fps_ctr  = FPSCounter(window=30)
    glitter  = GlitterEffect(max_particles=160)

    # ── drawing canvas — persistent BGRA layer ────────────────────────────────
    # We maintain a separate black canvas and blend it onto each camera frame.
    # This lets us keep strokes between frames without redrawing every point.
    canvas = np.zeros((actual_h, actual_w, 3), dtype=np.uint8)

    # ── state ─────────────────────────────────────────────────────────────────
    mode           = MODE_MOUSE
    draw_color     = "White"
    brush_width    = 6
    glitter_level  = 0              # 0=off, 1=light, 2=medium, 3=heavy
    volume_pct     = 50
    clicking       = False
    prev_draw_pt   = None           # last drawing point for line interpolation

    ui_hit_zones: dict = {}         # populated by UI.draw_color_bar each frame

    # ══════════════════════════════════════════════════════════════════════════
    #  SPLASH SCREEN LOOP
    # ══════════════════════════════════════════════════════════════════════════
    print("   Showing splash screen — press ENTER to continue…")
    cv2.namedWindow("Air AI Controller", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Air AI Controller", WIN_W, WIN_H)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)          # mirror (selfie view)
        splash_frame = draw_splash(frame.copy(), cam_index=0)
        cv2.imshow("Air AI Controller", splash_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key == 10:          # ENTER
            break
        if key == ord('q') or key == 27:    # Q or ESC
            cap.release()
            cv2.destroyAllWindows()
            return

    print("   Starting main loop — enjoy! 🖐")

    # ══════════════════════════════════════════════════════════════════════════
    #  MAIN LOOP
    # ══════════════════════════════════════════════════════════════════════════
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed — camera disconnected?")
            break

        # ── pre-process ───────────────────────────────────────────────────────
        frame = cv2.flip(frame, 1)          # mirror for intuitive control
        h, w  = frame.shape[:2]

        # ── hand detection (MediaPipe) ────────────────────────────────────────
        tracker.process_frame(frame)
        tracker.draw_landmarks(frame)       # draws skeleton on frame

        fingers      = tracker.fingers_up()
        hand_visible = tracker.hand_visible
        clicking     = False

        # Landmarks we care about most
        thumb_tip  = tracker.get_landmark(4)
        index_tip  = tracker.get_landmark(8)
        middle_tip = tracker.get_landmark(12)

        # ── gesture recognition ───────────────────────────────────────────────
        if hand_visible and index_tip:

            # --- 1. Volume control: thumb + index spread (any mode) -----------
            if thumb_tip:
                dist = euclidean_distance(thumb_tip, index_tip)
                volume_pct = distance_to_volume(dist,
                                                VOLUME_MIN_DIST,
                                                VOLUME_MAX_DIST)
                # Draw stretch line between thumb and index
                cv2.line(frame, thumb_tip, index_tip, (180, 180, 200), 2)
                mp_pt = midpoint(thumb_tip, index_tip)
                cv2.circle(frame, mp_pt, 6, (100, 220, 255), -1)

                # --- 2. Click: pinch gesture ----------------------------------
                if dist < CLICK_DIST and mode == MODE_MOUSE:
                    clicking = True
                    if debounce.can_click():
                        pyautogui.click()
                        # Visual feedback circle
                        cv2.circle(frame, index_tip, 20, (60, 60, 240), 3)

            # --- 3. Mode toggle via gesture: middle+index up, rest down -------
            #     (user can also press 'd' keyboard key)

            # --- 4. MOUSE MODE: move cursor ----------------------------------
            if mode == MODE_MOUSE:
                # Only move when index finger is up (prevents drift during pinch)
                if fingers[1] == 1:
                    sx, sy = map_to_screen(index_tip[0], index_tip[1],
                                           w, h,
                                           SCREEN_W, SCREEN_H,
                                           MARGIN)
                    # Apply EMA smoothing to reduce jitter
                    sx, sy = smoother.smooth(sx, sy)
                    pyautogui.moveTo(int(sx), int(sy))

                # Highlight cursor fingertip
                cv2.circle(frame, index_tip, 12, (100, 220, 255), -1)
                cv2.circle(frame, index_tip, 14, (255, 255, 255), 2)

            # --- 5. DRAW MODE: paint on canvas when ONLY index is up ---------
            elif mode == MODE_DRAW:
                # Only draw when index up and middle down (pen-up = stop)
                is_drawing = (fingers[1] == 1 and fingers[2] == 0)
                tip_color  = COLORS.get(draw_color, (255, 255, 255))
                thickness  = brush_width if draw_color != "Eraser" else 40

                if is_drawing:
                    if prev_draw_pt is not None:
                        cv2.line(canvas, prev_draw_pt, index_tip,
                                 tip_color, thickness,
                                 lineType=cv2.LINE_AA)
                    prev_draw_pt = index_tip

                    # Glitter particles
                    if glitter_level > 0:
                        glitter.emit(index_tip[0], index_tip[1],
                                     tip_color, intensity=glitter_level * 2)
                else:
                    prev_draw_pt = None     # lift pen

                # Show fingertip indicator
                dot_col = (50, 220, 50) if is_drawing else (80, 80, 120)
                cv2.circle(frame, index_tip, 10, dot_col, -1)

        else:
            prev_draw_pt = None             # hand lost: lift pen

        # ── blend canvas onto frame ───────────────────────────────────────────
        # Only blend non-black pixels from canvas (mask = any channel > 0)
        canvas_mask = np.any(canvas > 0, axis=2)
        frame[canvas_mask] = cv2.addWeighted(
            frame, 0.15, canvas, 0.85, 0
        )[canvas_mask]

        # ── glitter update ────────────────────────────────────────────────────
        if glitter_level > 0:
            glitter.update_and_draw(frame)

        # ── draw UI elements ─────────────────────────────────────────────────
        ui_hit_zones = UI.draw_color_bar(frame, draw_color,
                                         brush_width, glitter_level)
        UI.draw_mode_badge(frame, mode)
        UI.draw_info(frame, fps_ctr.tick(), volume_pct,
                     hand_visible, clicking)
        UI.draw_volume_bar(frame, volume_pct)
        UI.draw_hints(frame)
        if hand_visible:
            UI.draw_finger_dots(frame, fingers)

        # Bounding box around hand
        bb = tracker.bounding_box()
        if bb:
            bx, by, bw, bh = bb
            cv2.rectangle(frame, (bx, 64), (bx + bw, by + bh),
                          (60, 60, 100), 1)

        # ── show frame ────────────────────────────────────────────────────────
        cv2.imshow("Air AI Controller", frame)

        # ── keyboard shortcuts ────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('d') or key == ord('D'):
            mode = MODE_DRAW if mode == MODE_MOUSE else MODE_MOUSE
            smoother.reset()
            prev_draw_pt = None
            print(f"   Mode → {mode}")

        elif key == ord('c') or key == ord('C'):
            canvas[:] = 0
            glitter.particles.clear()
            print("   Canvas cleared.")

        elif key == ord('q') or key == 27:     # Q or ESC
            print("   Quit requested.")
            break

        # ── UI toolbar hit-testing (mouse click on toolbar) ──────────────────
        # We use OpenCV mouse callback for toolbar interaction
        # (set up below via cv2.setMouseCallback)

    # ── cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print("   Air AI Controller shut down cleanly. ✓")


# ══════════════════════════════════════════════════════════════════════════════
#  Mouse callback for toolbar clicks
# ══════════════════════════════════════════════════════════════════════════════

# We need shared state between callback and main loop:
_toolbar_state = {
    "color"   : "White",
    "width"   : 6,
    "glitter" : 0,
    "zones"   : {},
}

def _mouse_callback(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    zones = _toolbar_state["zones"]
    for key, (x1, y1, x2, y2) in zones.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            if key in COLOR_NAMES:
                _toolbar_state["color"] = key
            elif key.startswith("W"):
                _toolbar_state["width"] = int(key[1:])
            elif key.startswith("G"):
                _toolbar_state["glitter"] = int(key[1:])


# ══════════════════════════════════════════════════════════════════════════════
#  Re-entrant main with toolbar state
# ══════════════════════════════════════════════════════════════════════════════

def main_v2():
    """
    Enhanced main that also supports clicking the toolbar with the mouse.
    This replaces `main()` as the real entry point.
    """
    print("=" * 60)
    print("   AIR AI CONTROLLER  —  starting up (v2)")
    print("=" * 60)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 60)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"   Camera resolution: {actual_w}x{actual_h}")
    print(f"   Screen resolution: {SCREEN_W}x{SCREEN_H}")

    tracker  = HandTracker()
    smoother = SmoothFilter(alpha=0.25)
    debounce = ClickDebounce(cooldown=0.35)
    fps_ctr  = FPSCounter(window=30)
    glitter  = GlitterEffect(max_particles=180)

    canvas   = np.zeros((actual_h, actual_w, 3), dtype=np.uint8)

    mode         = MODE_MOUSE
    volume_pct   = 50
    clicking     = False
    prev_draw_pt = None

    # ── window setup ─────────────────────────────────────────────────────────
    cv2.namedWindow("Air AI Controller", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Air AI Controller", WIN_W, WIN_H)
    cv2.setMouseCallback("Air AI Controller", _mouse_callback)

    # ══ Splash loop ══════════════════════════════════════════════════════════
    print("   Splash screen → press ENTER to begin")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        cv2.imshow("Air AI Controller", draw_splash(frame.copy(), 0))
        key = cv2.waitKey(1) & 0xFF
        if key in (13, 10):
            break
        if key == ord('q') or key == 27:
            cap.release()
            cv2.destroyAllWindows()
            return

    print("   ▶  Main loop running — wave your hand!")

    # ══ Main loop ════════════════════════════════════════════════════════════
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # Sync toolbar state
        draw_color    = _toolbar_state["color"]
        brush_width   = _toolbar_state["width"]
        glitter_level = _toolbar_state["glitter"]

        tracker.process_frame(frame)
        tracker.draw_landmarks(frame)

        fingers      = tracker.fingers_up()
        hand_visible = tracker.hand_visible
        clicking     = False

        thumb_tip  = tracker.get_landmark(4)
        index_tip  = tracker.get_landmark(8)

        if hand_visible and index_tip:

            # Volume / pinch distance
            if thumb_tip:
                dist = euclidean_distance(thumb_tip, index_tip)
                volume_pct = distance_to_volume(dist, VOLUME_MIN_DIST, VOLUME_MAX_DIST)

                cv2.line(frame, thumb_tip, index_tip, (160, 160, 200), 2)
                cv2.circle(frame, midpoint(thumb_tip, index_tip), 5, UI.ACCENT, -1)

                if dist < CLICK_DIST and mode == MODE_MOUSE:
                    clicking = True
                    if debounce.can_click():
                        pyautogui.click()
                        cv2.circle(frame, index_tip, 22, UI.RED, 3)

            if mode == MODE_MOUSE:
                if fingers[1]:
                    sx, sy = map_to_screen(index_tip[0], index_tip[1],
                                           w, h, SCREEN_W, SCREEN_H, MARGIN)
                    sx, sy = smoother.smooth(sx, sy)
                    pyautogui.moveTo(int(sx), int(sy))
                cv2.circle(frame, index_tip, 12, UI.ACCENT, -1)
                cv2.circle(frame, index_tip, 14, (255, 255, 255), 2)

            elif mode == MODE_DRAW:
                tip_color = COLORS.get(draw_color, (255, 255, 255))
                thick     = brush_width if draw_color != "Eraser" else 42
                drawing   = fingers[1] == 1 and fingers[2] == 0

                if drawing:
                    if prev_draw_pt:
                        cv2.line(canvas, prev_draw_pt, index_tip,
                                 tip_color, thick, cv2.LINE_AA)
                    prev_draw_pt = index_tip
                    if glitter_level > 0:
                        glitter.emit(index_tip[0], index_tip[1],
                                     tip_color, intensity=glitter_level * 2)
                else:
                    prev_draw_pt = None

                cv2.circle(frame, index_tip, 10,
                           (50, 220, 50) if drawing else (80, 80, 120), -1)
        else:
            prev_draw_pt = None

        # Blend drawing canvas
        mask = np.any(canvas > 0, axis=2)
        frame[mask] = cv2.addWeighted(frame, 0.12, canvas, 0.88, 0)[mask]

        # Glitter
        if glitter_level > 0:
            glitter.update_and_draw(frame)

        # UI
        zones = UI.draw_color_bar(frame, draw_color, brush_width, glitter_level)
        _toolbar_state["zones"] = zones
        UI.draw_mode_badge(frame, mode)
        UI.draw_info(frame, fps_ctr.tick(), volume_pct, hand_visible, clicking)
        UI.draw_volume_bar(frame, volume_pct)
        UI.draw_hints(frame)
        if hand_visible:
            UI.draw_finger_dots(frame, fingers)

        bb = tracker.bounding_box()
        if bb:
            bx, by, bw2, bh = bb
            cv2.rectangle(frame, (bx, 68), (bx + bw2, by + bh), (60, 60, 100), 1)

        cv2.imshow("Air AI Controller", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('d') or key == ord('D'):
            mode = MODE_DRAW if mode == MODE_MOUSE else MODE_MOUSE
            smoother.reset()
            prev_draw_pt = None
            print(f"   Mode → {mode}")
        elif key == ord('c') or key == ord('C'):
            canvas[:] = 0
            glitter.particles.clear()
            print("   Canvas cleared.")
        elif key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("   Bye! ✓")


# ─── entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main_v2()
