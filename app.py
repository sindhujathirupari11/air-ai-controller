"""
app.py
------
Air AI Controller — Main application entry point.

Modes
-----
  M → Mouse Mode   : Index finger = cursor | Thumb+Index pinch = left click
  D → Draw Mode    : Index draws | Index+Middle = pen up | C = clear canvas
  V → Volume Mode  : Thumb-Index distance controls system volume

Controls
--------
  M / D / V  : switch mode
  C          : clear canvas (Draw mode only)
  Q          : quit
"""

import cv2
import numpy as np
import pyautogui
import time
import sys

from hand_tracker import HandTracker
from utils        import (SmoothCursor, GlitterEffect, ClickDebounce,
                          interpolate, put_text_shadow, draw_rounded_rect,
                          draw_gradient_rect)

# ── Pycaw (Windows volume control) ─────────────────────────────────────────
try:
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    from comtypes import CLSCTX_ALL
    from ctypes   import cast, POINTER
    _PYCAW_OK = True
except ImportError:
    _PYCAW_OK = False

# ── PyAutoGUI safety ────────────────────────────────────────────────────────
pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0.0


# ===========================================================================
#  Constants & Theme
# ===========================================================================

# Window
WIN_NAME  = "Air AI Controller"
CAM_W, CAM_H = 1280, 720          # Capture resolution
UI_W,  UI_H  = 1280, 720          # Display resolution

# Panels
HEADER_H  = 70
LEFT_W    = 220
RIGHT_W   = 230
BOTTOM_H  = 0                      # reserved / unused

# Camera area inside panels
CAM_X1 = LEFT_W
CAM_X2 = UI_W - RIGHT_W
CAM_Y1 = HEADER_H
CAM_Y2 = UI_H

# Mouse mode: active region inside camera frame (avoids panel overlap)
MOUSE_PAD_X1, MOUSE_PAD_X2 = 80,  CAM_W - 80
MOUSE_PAD_Y1, MOUSE_PAD_Y2 = 80,  CAM_H - 80

# Screen size (for PyAutoGUI)
SCR_W, SCR_H = pyautogui.size()

# ── Colour palette ───────────────────────────────────────────────────────────
C = {
    "bg"          : (18,  18,  24),
    "panel"       : (26,  26,  36),
    "header_top"  : (45,  30,  90),
    "header_bot"  : (20,  20,  50),
    "accent"      : (130, 90, 255),   # purple
    "accent2"     : (60, 200, 255),   # cyan
    "green"       : (60, 220, 120),
    "red"         : (60,  80, 230),
    "yellow"      : (40, 210, 240),
    "white"       : (230, 230, 240),
    "subtext"     : (140, 130, 160),
    "divider"     : (50,  45,  70),
    "vol_bar_bg"  : (40,  35,  60),
    "vol_bar_fg"  : (100, 180, 255),
}

FONT      = cv2.FONT_HERSHEY_DUPLEX
FONT_MONO = cv2.FONT_HERSHEY_PLAIN

# ── Draw palette colors (BGR) ────────────────────────────────────────────────
DRAW_COLORS = [
    ("White",   (240, 240, 240)),
    ("Cyan",    (255, 220,  60)),
    ("Magenta", (230,  60, 200)),
    ("Yellow",  ( 30, 220, 240)),
    ("Green",   ( 60, 220, 100)),
    ("Red",     ( 40,  60, 230)),
    ("Blue",    (230,  80,  40)),
    ("Orange",  ( 30, 140, 255)),
]

BRUSH_SIZES = [3, 6, 10, 16, 24]


# ===========================================================================
#  Volume Initialisation (Pycaw)
# ===========================================================================

def init_volume():
    if not _PYCAW_OK:
        return None, (-65.25, 0.0)
    try:
        devices   = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume    = cast(interface, POINTER(IAudioEndpointVolume))
        vol_range = volume.GetVolumeRange()          # (min_dB, max_dB, step)
        return volume, (vol_range[0], vol_range[1])
    except Exception:
        return None, (-65.25, 0.0)


# ===========================================================================
#  UI Drawing helpers
# ===========================================================================

def draw_header(frame, mode):
    draw_gradient_rect(frame, 0, 0, UI_W, HEADER_H,
                       C["header_top"], C["header_bot"])
    cv2.line(frame, (0, HEADER_H), (UI_W, HEADER_H), C["accent"], 2)

    # Title
    put_text_shadow(frame, "AIR AI CONTROLLER",
                    (LEFT_W + 20, 44), FONT, 0.90, C["white"], 2)

    # Mode badge
    mode_colors = {"MOUSE": C["green"], "DRAW": C["accent"], "VOLUME": C["yellow"]}
    badge_color = mode_colors.get(mode, C["white"])
    label = f"  {mode} MODE  "
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.65, 1)
    bx = UI_W - RIGHT_W - tw - 30
    draw_rounded_rect(frame, bx - 8, 16, bx + tw + 8, 54, 8, badge_color, -1)
    cv2.putText(frame, label, (bx, 44), FONT, 0.65, C["bg"], 2, cv2.LINE_AA)


def draw_left_panel(frame, mode):
    # Panel background
    cv2.rectangle(frame, (0, HEADER_H), (LEFT_W, UI_H), C["panel"], -1)
    cv2.line(frame, (LEFT_W, HEADER_H), (LEFT_W, UI_H), C["divider"], 1)

    y = HEADER_H + 28
    put_text_shadow(frame, "CONTROLS", (14, y), FONT, 0.52, C["accent"], 1)
    y += 8
    cv2.line(frame, (10, y), (LEFT_W - 10, y), C["divider"], 1)
    y += 22

    INST = {
        "MOUSE": [
            (" Index Finger", "Move cursor"),
            (" Pinch (Thumb+", "Index)  = Click"),
            ("", ""),
            (" M ", "Mouse Mode"),
            (" D ", "Draw  Mode"),
            (" V ", "Volume Mode"),
            (" Q ", "Quit"),
        ],
        "DRAW": [
            (" Index Finger", "Draw on screen"),
            (" Index+Middle", "Pen up (stop)"),
            (" C Key ", "Clear canvas"),
            ("", ""),
            (" M ", "Mouse Mode"),
            (" D ", "Draw  Mode"),
            (" V ", "Volume Mode"),
            (" Q ", "Quit"),
        ],
        "VOLUME": [
            (" Thumb+Index", "Distance ="),
            ("", "Volume level"),
            ("", ""),
            (" Spread wide", "Max volume"),
            (" Pinch close", "Min volume"),
            ("", ""),
            (" M ", "Mouse Mode"),
            (" D ", "Draw  Mode"),
            (" V ", "Volume Mode"),
            (" Q ", "Quit"),
        ],
    }

    for key, val in INST.get(mode, []):
        if not key and not val:
            y += 10
            continue
        if key in (" M ", " D ", " V ", " Q ", " C Key "):
            # Key badge
            draw_rounded_rect(frame, 12, y - 14, 12 + 40, y + 6, 4, C["accent"], -1)
            cv2.putText(frame, key.strip(), (16, y), FONT_MONO, 1.1,
                        C["bg"], 1, cv2.LINE_AA)
            cv2.putText(frame, val, (60, y), FONT_MONO, 1.1,
                        C["subtext"], 1, cv2.LINE_AA)
        else:
            put_text_shadow(frame, key, (14, y), FONT_MONO, 1.0, C["accent2"], 1)
            if val:
                put_text_shadow(frame, val, (14, y + 16), FONT_MONO, 1.0,
                                C["subtext"], 1)
                y += 16
        y += 28


def draw_right_panel_mouse(frame):
    cv2.rectangle(frame, (UI_W - RIGHT_W, HEADER_H), (UI_W, UI_H), C["panel"], -1)
    cv2.line(frame, (UI_W - RIGHT_W, HEADER_H),
             (UI_W - RIGHT_W, UI_H), C["divider"], 1)
    y = HEADER_H + 30
    put_text_shadow(frame, "MOUSE MODE", (UI_W - RIGHT_W + 14, y),
                    FONT, 0.48, C["green"], 1)
    y += 30
    tips = [
        "Point index finger",
        "to move the cursor.",
        "",
        "Pinch thumb and",
        "index together",
        "to left-click.",
        "",
        "Works with real",
        "OS windows,",
        "browsers, files.",
    ]
    for line in tips:
        if line:
            cv2.putText(frame, line, (UI_W - RIGHT_W + 12, y),
                        FONT_MONO, 1.0, C["subtext"], 1, cv2.LINE_AA)
        y += 20


def draw_right_panel_draw(frame, color_idx, brush_idx, glitter_on,
                          palette_rects, brush_rects, glitter_rect):
    cv2.rectangle(frame, (UI_W - RIGHT_W, HEADER_H), (UI_W, UI_H), C["panel"], -1)
    cv2.line(frame, (UI_W - RIGHT_W, HEADER_H),
             (UI_W - RIGHT_W, UI_H), C["divider"], 1)

    px = UI_W - RIGHT_W + 14
    y  = HEADER_H + 26
    put_text_shadow(frame, "COLOR PALETTE", (px, y), FONT, 0.45, C["accent"], 1)
    y += 16

    # Color swatches (2 columns)
    palette_rects.clear()
    col_w = (RIGHT_W - 28) // 2
    for i, (name, bgr) in enumerate(DRAW_COLORS):
        col   = i % 2
        row   = i // 2
        sx    = px + col * (col_w + 4)
        sy    = y  + row * 38
        ex    = sx + col_w
        ey    = sy + 30
        draw_rounded_rect(frame, sx, sy, ex, ey, 6, bgr, -1)
        palette_rects.append((sx, sy, ex, ey, i))
        if i == color_idx:
            draw_rounded_rect(frame, sx - 2, sy - 2, ex + 2, ey + 2,
                              7, C["white"], 2)
        cv2.putText(frame, name[:6], (sx + 4, sy + 20),
                    FONT_MONO, 0.85, C["bg"], 1, cv2.LINE_AA)

    y += (len(DRAW_COLORS) // 2) * 38 + 14
    cv2.line(frame, (px - 4, y), (UI_W - 8, y), C["divider"], 1)
    y += 18

    # Brush sizes
    put_text_shadow(frame, "BRUSH SIZE", (px, y), FONT, 0.45, C["accent"], 1)
    y += 20
    brush_rects.clear()
    bx = px
    for i, sz in enumerate(BRUSH_SIZES):
        cx = bx + i * 40 + sz
        cy = y + 16
        cv2.circle(frame, (cx, cy), sz, C["white"], -1)
        if i == brush_idx:
            cv2.circle(frame, (cx, cy), sz + 3, C["accent"], 2)
        brush_rects.append((cx - sz - 3, cy - sz - 3,
                            cx + sz + 3, cy + sz + 3, i))
        bx += 4
    y += 50

    cv2.line(frame, (px - 4, y), (UI_W - 8, y), C["divider"], 1)
    y += 18

    # Glitter toggle
    put_text_shadow(frame, "GLITTER", (px, y), FONT, 0.45, C["accent"], 1)
    y += 24
    gx1, gy1, gx2, gy2 = px, y, px + 80, y + 28
    glitter_rect[:] = [gx1, gy1, gx2, gy2]
    tog_color = C["yellow"] if glitter_on else C["divider"]
    draw_rounded_rect(frame, gx1, gy1, gx2, gy2, 8, tog_color, -1)
    label = "ON " if glitter_on else "OFF"
    cv2.putText(frame, label, (gx1 + 22, gy1 + 20),
                FONT, 0.52, C["bg"] if glitter_on else C["subtext"],
                1, cv2.LINE_AA)


def draw_right_panel_volume(frame, vol_pct):
    cv2.rectangle(frame, (UI_W - RIGHT_W, HEADER_H), (UI_W, UI_H), C["panel"], -1)
    cv2.line(frame, (UI_W - RIGHT_W, HEADER_H),
             (UI_W - RIGHT_W, UI_H), C["divider"], 1)

    px = UI_W - RIGHT_W + 14
    y  = HEADER_H + 30
    put_text_shadow(frame, "VOLUME", (px, y), FONT, 0.52, C["yellow"], 1)
    y += 50

    # Vertical bar
    bar_x  = UI_W - RIGHT_W + RIGHT_W // 2 - 20
    bar_y1 = y
    bar_y2 = y + 340
    bar_w  = 40

    # Background track
    draw_rounded_rect(frame, bar_x, bar_y1, bar_x + bar_w, bar_y2,
                      10, C["vol_bar_bg"], -1)

    # Fill based on volume
    fill_h  = int((bar_y2 - bar_y1) * vol_pct / 100)
    fill_y1 = bar_y2 - fill_h
    if fill_h > 4:
        # Gradient fill: low=blue, high=green
        for row in range(fill_y1, bar_y2):
            t = (row - fill_y1) / max(fill_h, 1)
            b = int(255 * (1 - t) * (vol_pct / 100))
            g = int(180 * t)
            r = 40
            cv2.line(frame, (bar_x + 4, row), (bar_x + bar_w - 4, row),
                     (r, g, b), 1)

    # Border
    draw_rounded_rect(frame, bar_x, bar_y1, bar_x + bar_w, bar_y2,
                      10, C["accent"], 2)

    # Percentage text
    pct_label = f"{int(vol_pct)}%"
    (tw, _), _ = cv2.getTextSize(pct_label, FONT, 0.80, 2)
    cv2.putText(frame, pct_label,
                (bar_x + bar_w // 2 - tw // 2, bar_y2 + 34),
                FONT, 0.80, C["vol_bar_fg"], 2, cv2.LINE_AA)

    # "MUTE" label at low volumes
    if vol_pct < 3:
        put_text_shadow(frame, "MUTED", (px, bar_y2 + 60),
                        FONT, 0.55, C["red"], 1)


def draw_status_bar(frame, fps, hand_ok, mode):
    y = UI_H - 14
    fps_color = C["green"] if fps >= 24 else C["yellow"] if fps >= 15 else C["red"]
    cv2.putText(frame, f"FPS: {fps:.0f}", (LEFT_W + 10, y),
                FONT_MONO, 1.1, fps_color, 1, cv2.LINE_AA)

    hand_label = "Hand: DETECTED" if hand_ok else "Hand: NOT FOUND"
    hand_color = C["green"] if hand_ok else C["red"]
    cv2.putText(frame, hand_label, (LEFT_W + 100, y),
                FONT_MONO, 1.1, hand_color, 1, cv2.LINE_AA)


# ===========================================================================
#  Main Application
# ===========================================================================

def main():
    # ── Camera ──────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Exiting.")
        sys.exit(1)

    # ── Modules ─────────────────────────────────────────────────────────────
    tracker  = HandTracker(max_hands=1, detection_conf=0.75, tracking_conf=0.75)
    smoother = SmoothCursor(alpha=0.22)
    glitter  = GlitterEffect()
    debounce = ClickDebounce(cooldown_frames=18)

    # ── Volume ───────────────────────────────────────────────────────────────
    vol_obj, (vol_min, vol_max) = init_volume()
    vol_pct   = 50.0
    vol_db    = 0.0

    # ── State ────────────────────────────────────────────────────────────────
    mode       = "MOUSE"    # MOUSE | DRAW | VOLUME
    prev_x     = prev_y = 0
    draw_x     = draw_y = 0
    is_drawing = False

    # Draw canvas (transparent overlay, same size as camera region)
    cam_region_w = CAM_X2 - CAM_X1
    cam_region_h = CAM_Y2 - CAM_Y1
    canvas = np.zeros((cam_region_h, cam_region_w, 3), dtype=np.uint8)

    color_idx  = 0
    brush_idx  = 1
    glitter_on = False

    # UI state for right panel interactables
    palette_rects = []   # filled each frame
    brush_rects   = []
    glitter_rect  = [0, 0, 0, 0]

    # FPS tracking
    fps_list  = []
    t_prev    = time.time()

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, UI_W, UI_H)

    print("[INFO] Air AI Controller started.")
    print("[INFO] Press M=Mouse  D=Draw  V=Volume  C=Clear  Q=Quit")

    # ── Main Loop ────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame.")
            break

        frame = cv2.flip(frame, 1)                      # Mirror for natural feel
        frame = cv2.resize(frame, (CAM_W, CAM_H))

        # Hand tracking (draw landmarks on camera frame)
        frame = tracker.find_hands(frame, draw=True)
        fingers = tracker.fingers_up()
        hand_ok = tracker.hand_detected()

        # ── Camera feed crop to display region ──────────────────────────────
        cam_crop = frame[0:CAM_H, 0:CAM_W]             # full camera frame
        # Resize to fit the centre panel
        cam_display = cv2.resize(
            cam_crop,
            (cam_region_w, cam_region_h),
            interpolation=cv2.INTER_LINEAR
        )

        # ────────────────────────────────────────────────────────────────────
        #  Mode Logic
        # ────────────────────────────────────────────────────────────────────

        if mode == "MOUSE" and hand_ok:
            # ── Mouse mode ─────────────────────────────────────────────────
            ix, iy = tracker.get_landmark(8)   # Index fingertip

            # Map camera coords → screen coords
            sx = int(interpolate(ix, MOUSE_PAD_X1, MOUSE_PAD_X2, 0, SCR_W))
            sy = int(interpolate(iy, MOUSE_PAD_Y1, MOUSE_PAD_Y2, 0, SCR_H))
            sx, sy = smoother.update(sx, sy)

            pyautogui.moveTo(sx, sy)

            # Pinch detection: thumb(4) + index(8) distance
            dist, mx, my = tracker.distance_between(4, 8)
            if dist < 38 and debounce.ready():
                pyautogui.click()
                debounce.trigger()
                # Visual pinch indicator on camera
                # (mapped back to camera space)
                cmx = int(interpolate(mx, 0, CAM_W, 0, cam_region_w))
                cmy = int(interpolate(my, 0, CAM_H, 0, cam_region_h))
                cv2.circle(cam_display, (cmx, cmy), 20, C["green"], 3)
                cv2.putText(cam_display, "CLICK!", (cmx - 30, cmy - 28),
                            FONT_MONO, 1.4, C["green"], 1, cv2.LINE_AA)

        elif mode == "DRAW" and hand_ok:
            # ── Draw mode ──────────────────────────────────────────────────
            ix, iy = tracker.get_landmark(8)    # Index tip
            mx, my = tracker.get_landmark(12)   # Middle tip

            # Map to canvas space
            cx = int(interpolate(ix, 0, CAM_W, 0, cam_region_w))
            cy = int(interpolate(iy, 0, CAM_H, 0, cam_region_h))

            # Index + Middle up → pen up
            if fingers[1] == 1 and fingers[2] == 1:
                is_drawing = False
                draw_x = draw_y = 0
                cv2.circle(cam_display, (cx, cy), 12, C["accent2"], 2)

            # Only index up → drawing
            elif fingers[1] == 1 and fingers[2] == 0:
                brush_r = BRUSH_SIZES[brush_idx]
                color   = DRAW_COLORS[color_idx][1]

                if draw_x == 0 and draw_y == 0:
                    draw_x, draw_y = cx, cy

                cv2.line(canvas, (draw_x, draw_y), (cx, cy), color,
                         brush_r, cv2.LINE_AA)

                if glitter_on:
                    glitter.spawn(cx, cy)

                draw_x, draw_y = cx, cy
                is_drawing = True

                # Cursor ring
                cv2.circle(cam_display, (cx, cy), brush_r + 4, color, 2)
            else:
                draw_x = draw_y = 0
                is_drawing = False

            # Render glitter onto canvas
            if glitter_on:
                glitter.render(canvas)

            # Merge canvas onto camera display
            mask = canvas.astype(bool).any(axis=2)
            cam_display[mask] = cv2.addWeighted(
                cam_display, 0.15, canvas, 0.85, 0
            )[mask]

        elif mode == "VOLUME" and hand_ok:
            # ── Volume mode ────────────────────────────────────────────────
            dist, mx, my = tracker.distance_between(4, 8)  # thumb–index

            # Distance range ~20px (pinch) to ~220px (wide spread)
            vol_pct = interpolate(dist, 20, 220, 0, 100)

            if vol_obj is not None:
                try:
                    vol_db = interpolate(vol_pct, 0, 100, vol_min, vol_max)
                    vol_obj.SetMasterVolumeLevel(vol_db, None)
                except Exception:
                    pass

            # Visual line between thumb and index on camera display
            t4 = tracker.get_landmark(4)
            t8 = tracker.get_landmark(8)
            if t4 and t8:
                cmx4 = int(interpolate(t4[0], 0, CAM_W, 0, cam_region_w))
                cmy4 = int(interpolate(t4[1], 0, CAM_H, 0, cam_region_h))
                cmx8 = int(interpolate(t8[0], 0, CAM_W, 0, cam_region_w))
                cmy8 = int(interpolate(t8[1], 0, CAM_H, 0, cam_region_h))
                cv2.line(cam_display, (cmx4, cmy4), (cmx8, cmy8),
                         C["yellow"], 2)
                cv2.circle(cam_display, (cmx4, cmy4), 10, C["yellow"], -1)
                cv2.circle(cam_display, (cmx8, cmy8), 10, C["yellow"], -1)
                cv2.circle(cam_display,
                           ((cmx4 + cmx8) // 2, (cmy4 + cmy8) // 2),
                           6, C["white"], -1)

        # Also merge canvas when just displaying (other modes shouldn't clear it)
        if mode != "DRAW":
            mask = canvas.astype(bool).any(axis=2)
            cam_display[mask] = cv2.addWeighted(
                cam_display, 0.15, canvas, 0.85, 0
            )[mask]

        # ────────────────────────────────────────────────────────────────────
        #  Build UI frame
        # ────────────────────────────────────────────────────────────────────
        ui = np.full((UI_H, UI_W, 3), C["bg"], dtype=np.uint8)

        # Paste camera into centre region
        ui[CAM_Y1:CAM_Y2, CAM_X1:CAM_X2] = cam_display

        # Header
        draw_header(ui, mode)

        # Left panel
        draw_left_panel(ui, mode)

        # Right panel
        if mode == "MOUSE":
            draw_right_panel_mouse(ui)
        elif mode == "DRAW":
            draw_right_panel_draw(ui, color_idx, brush_idx, glitter_on,
                                  palette_rects, brush_rects, glitter_rect)
        elif mode == "VOLUME":
            draw_right_panel_volume(ui, vol_pct)

        # FPS
        t_now = time.time()
        fps   = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev = t_now
        fps_list.append(fps)
        if len(fps_list) > 15:
            fps_list.pop(0)
        avg_fps = sum(fps_list) / len(fps_list)

        draw_status_bar(ui, avg_fps, hand_ok, mode)

        # ────────────────────────────────────────────────────────────────────
        #  Show & Key Handling
        # ────────────────────────────────────────────────────────────────────
        cv2.imshow(WIN_NAME, ui)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord('m') or key == ord('M'):
            mode = "MOUSE"
            smoother.reset()
        elif key == ord('d') or key == ord('D'):
            mode = "DRAW"
        elif key == ord('v') or key == ord('V'):
            mode = "VOLUME"
        elif key == ord('c') or key == ord('C'):
            if mode == "DRAW":
                canvas[:] = 0
                glitter.clear()

        # ── Mouse clicks on right-panel UI (Draw mode) ──────────────────────
        # We intercept mouse events on the OpenCV window for palette selection
        # (done via a global state + setMouseCallback below)

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Air AI Controller closed.")


# ===========================================================================
#  Mouse callback for panel interactions (palette / brush / glitter)
# ===========================================================================

_click_state = {
    "palette_rects": [],
    "brush_rects"  : [],
    "glitter_rect" : [0, 0, 0, 0],
    "color_idx"    : [0],
    "brush_idx"    : [1],
    "glitter_on"   : [False],
    "mode"         : ["MOUSE"],
}


def _mouse_cb(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if _click_state["mode"][0] != "DRAW":
        return

    for (x1, y1, x2, y2, i) in _click_state["palette_rects"]:
        if x1 <= x <= x2 and y1 <= y <= y2:
            _click_state["color_idx"][0] = i
            return

    for (x1, y1, x2, y2, i) in _click_state["brush_rects"]:
        if x1 <= x <= x2 and y1 <= y <= y2:
            _click_state["brush_idx"][0] = i
            return

    gr = _click_state["glitter_rect"]
    if gr[0] <= x <= gr[2] and gr[1] <= y <= gr[3]:
        _click_state["glitter_on"][0] = not _click_state["glitter_on"][0]


# ===========================================================================
#  Main with mouse-callback wiring
# ===========================================================================

def main_with_ui():
    """
    Wrapper that wires up the OpenCV mouse callback for panel interactions
    and runs the main loop with shared state.
    """
    # ── Camera ──────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Exiting.")
        sys.exit(1)

    # ── Modules ─────────────────────────────────────────────────────────────
    tracker  = HandTracker(max_hands=1, detection_conf=0.75, tracking_conf=0.75)
    smoother = SmoothCursor(alpha=0.22)
    glitter  = GlitterEffect()
    debounce = ClickDebounce(cooldown_frames=18)

    # ── Volume ───────────────────────────────────────────────────────────────
    vol_obj, (vol_min, vol_max) = init_volume()
    vol_pct = 50.0

    # ── Mutable state (shared with mouse callback via dict) ──────────────────
    state = {
        "mode"        : "MOUSE",
        "color_idx"   : 0,
        "brush_idx"   : 1,
        "glitter_on"  : False,
        "palette_rects": [],
        "brush_rects"  : [],
        "glitter_rect" : [0, 0, 0, 0],
    }

    cam_region_w = CAM_X2 - CAM_X1
    cam_region_h = CAM_Y2 - CAM_Y1
    canvas = np.zeros((cam_region_h, cam_region_w, 3), dtype=np.uint8)

    prev_draw_x = prev_draw_y = 0

    fps_list = []
    t_prev   = time.time()

    # ── Window + callback ────────────────────────────────────────────────────
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, UI_W, UI_H)

    def mouse_cb(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if state["mode"] != "DRAW":
            return
        for (x1, y1, x2, y2, i) in state["palette_rects"]:
            if x1 <= x <= x2 and y1 <= y <= y2:
                state["color_idx"] = i
                return
        for (x1, y1, x2, y2, i) in state["brush_rects"]:
            if x1 <= x <= x2 and y1 <= y <= y2:
                state["brush_idx"] = i
                return
        gr = state["glitter_rect"]
        if gr[0] <= x <= gr[2] and gr[1] <= y <= gr[3]:
            state["glitter_on"] = not state["glitter_on"]

    cv2.setMouseCallback(WIN_NAME, mouse_cb)

    print("[INFO] Air AI Controller started.")
    print("[INFO] Press M=Mouse  D=Draw  V=Volume  C=Clear  Q=Quit")
    if not _PYCAW_OK:
        print("[WARN] Pycaw not found – volume control disabled (simulated).")

    smoother_x = smoother_y = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame read failed.")
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (CAM_W, CAM_H))

        frame   = tracker.find_hands(frame, draw=True)
        fingers = tracker.fingers_up()
        hand_ok = tracker.hand_detected()

        cam_display = cv2.resize(
            frame, (cam_region_w, cam_region_h),
            interpolation=cv2.INTER_LINEAR
        )

        mode = state["mode"]

        # ── MOUSE MODE ────────────────────────────────────────────────────
        if mode == "MOUSE" and hand_ok:
            pos = tracker.get_landmark(8)
            if pos:
                ix, iy = pos
                sx = int(interpolate(ix, MOUSE_PAD_X1, MOUSE_PAD_X2, 0, SCR_W))
                sy = int(interpolate(iy, MOUSE_PAD_Y1, MOUSE_PAD_Y2, 0, SCR_H))
                sx, sy = smoother.update(sx, sy)
                pyautogui.moveTo(sx, sy)

                dist, mx, my = tracker.distance_between(4, 8)
                if dist < 38 and debounce.ready():
                    pyautogui.click()
                    debounce.trigger()
                    cmx = int(interpolate(mx, 0, CAM_W, 0, cam_region_w))
                    cmy = int(interpolate(my, 0, CAM_H, 0, cam_region_h))
                    cv2.circle(cam_display, (cmx, cmy), 22, C["green"], 3)
                    put_text_shadow(cam_display, "CLICK",
                                   (cmx - 28, cmy - 30),
                                   FONT_MONO, 1.4, C["green"], 1)

        # ── DRAW MODE ─────────────────────────────────────────────────────
        elif mode == "DRAW" and hand_ok:
            pos8  = tracker.get_landmark(8)
            pos12 = tracker.get_landmark(12)

            if pos8:
                ix, iy = pos8
                cx = int(interpolate(ix, 0, CAM_W, 0, cam_region_w))
                cy = int(interpolate(iy, 0, CAM_H, 0, cam_region_h))

                # Pen up: index + middle both up
                if fingers[1] == 1 and fingers[2] == 1:
                    prev_draw_x = prev_draw_y = 0
                    cv2.circle(cam_display, (cx, cy), 14, C["accent2"], 2)

                # Drawing: only index up
                elif fingers[1] == 1 and fingers[2] == 0:
                    brush_r = BRUSH_SIZES[state["brush_idx"]]
                    color   = DRAW_COLORS[state["color_idx"]][1]

                    if prev_draw_x == 0 and prev_draw_y == 0:
                        prev_draw_x, prev_draw_y = cx, cy

                    cv2.line(canvas, (prev_draw_x, prev_draw_y),
                             (cx, cy), color, brush_r, cv2.LINE_AA)

                    if state["glitter_on"]:
                        glitter.spawn(cx, cy)

                    prev_draw_x, prev_draw_y = cx, cy
                    cv2.circle(cam_display, (cx, cy), brush_r + 4, color, 2)
                else:
                    prev_draw_x = prev_draw_y = 0

            if state["glitter_on"]:
                glitter.render(canvas)

            mask = canvas.astype(bool).any(axis=2)
            cam_display[mask] = cv2.addWeighted(
                cam_display, 0.15, canvas, 0.85, 0)[mask]

        # ── VOLUME MODE ───────────────────────────────────────────────────
        elif mode == "VOLUME" and hand_ok:
            dist, _, _ = tracker.distance_between(4, 8)
            vol_pct = interpolate(dist, 20, 220, 0, 100)

            if vol_obj is not None:
                try:
                    vdb = interpolate(vol_pct, 0, 100, vol_min, vol_max)
                    vol_obj.SetMasterVolumeLevel(vdb, None)
                except Exception:
                    pass

            t4 = tracker.get_landmark(4)
            t8 = tracker.get_landmark(8)
            if t4 and t8:
                p4 = (int(interpolate(t4[0], 0, CAM_W, 0, cam_region_w)),
                      int(interpolate(t4[1], 0, CAM_H, 0, cam_region_h)))
                p8 = (int(interpolate(t8[0], 0, CAM_W, 0, cam_region_w)),
                      int(interpolate(t8[1], 0, CAM_H, 0, cam_region_h)))
                cv2.line(cam_display, p4, p8, C["yellow"], 3)
                cv2.circle(cam_display, p4, 12, C["yellow"], -1)
                cv2.circle(cam_display, p8, 12, C["yellow"], -1)
                mid = ((p4[0] + p8[0]) // 2, (p4[1] + p8[1]) // 2)
                cv2.circle(cam_display, mid, 7, C["white"], -1)
                put_text_shadow(cam_display, f"{int(vol_pct)}%",
                               (mid[0] + 14, mid[1] - 8),
                               FONT, 0.65, C["yellow"], 1)

        # Overlay draw canvas in non-draw modes (preserve strokes)
        if mode != "DRAW":
            mask = canvas.astype(bool).any(axis=2)
            if mask.any():
                cam_display[mask] = cv2.addWeighted(
                    cam_display, 0.15, canvas, 0.85, 0)[mask]

        # ── Build UI ──────────────────────────────────────────────────────
        ui = np.full((UI_H, UI_W, 3), C["bg"], dtype=np.uint8)
        ui[CAM_Y1:CAM_Y2, CAM_X1:CAM_X2] = cam_display

        draw_header(ui, mode)
        draw_left_panel(ui, mode)

        if mode == "MOUSE":
            draw_right_panel_mouse(ui)
        elif mode == "DRAW":
            draw_right_panel_draw(
                ui,
                state["color_idx"],
                state["brush_idx"],
                state["glitter_on"],
                state["palette_rects"],
                state["brush_rects"],
                state["glitter_rect"],
            )
        elif mode == "VOLUME":
            draw_right_panel_volume(ui, vol_pct)

        # FPS counter
        t_now  = time.time()
        fps    = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev = t_now
        fps_list.append(fps)
        if len(fps_list) > 15:
            fps_list.pop(0)
        avg_fps = sum(fps_list) / len(fps_list)

        draw_status_bar(ui, avg_fps, hand_ok, mode)

        cv2.imshow(WIN_NAME, ui)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key in (ord('m'), ord('M')):
            state["mode"] = "MOUSE"
            smoother.reset()
        elif key in (ord('d'), ord('D')):
            state["mode"] = "DRAW"
            prev_draw_x = prev_draw_y = 0
        elif key in (ord('v'), ord('V')):
            state["mode"] = "VOLUME"
        elif key in (ord('c'), ord('C')):
            if state["mode"] == "DRAW":
                canvas[:] = 0
                glitter.clear()
                prev_draw_x = prev_draw_y = 0

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Air AI Controller closed.")


if __name__ == "__main__":
    main_with_ui()