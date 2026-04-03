"""
app.py
------
Air AI Controller — Optimized Version.
- Fixed NameError: Restored 'draw_right_panel_mouse'.
- Refined Selection Border: Highlight drawn behind colors.
- Highlighter Stroke: Edge-glow effect for Draw Mode.
- Simplified UI: Minimalist "C" label for Clear.
"""

import cv2
import numpy as np
import pyautogui
import time
import sys

# Import our custom modules
from hand_tracker import HandTracker
from utils import (SmoothCursor, ClickDebounce,
                          interpolate, put_text_shadow, draw_rounded_rect,
                          draw_gradient_rect)

# ── PyAutoGUI safety ────────────────────────────────────────────────────────
pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0.0

# ===========================================================================
#  Constants & Theme
# ===========================================================================

WIN_NAME  = "Air AI Controller"
CAM_W, CAM_H = 1280, 720          
UI_W,  UI_H  = 1280, 720          

HEADER_H  = 70
LEFT_W    = 220
RIGHT_W   = 230

CAM_X1, CAM_X2 = LEFT_W, UI_W - RIGHT_W
CAM_Y1, CAM_Y2 = HEADER_H, UI_H

# Mouse active region calibrated for full screen reach
MOUSE_PAD_X1, MOUSE_PAD_X2 = 180, 1100
MOUSE_PAD_Y1, MOUSE_PAD_Y2 = 120, 600

SCR_W, SCR_H = pyautogui.size()

C = {
    "bg"          : (18,  18,  24),
    "panel"       : (26,  26,  36),
    "header_top"  : (45,  30,  90),
    "header_bot"  : (20,  20,  50),
    "accent"      : (130, 90, 255),
    "accent2"     : (60, 200, 255),
    "green"       : (60, 220, 120),
    "red"         : (60,  80, 230),
    "white"       : (230, 230, 240),
    "subtext"     : (140, 130, 160),
    "divider"     : (50,  45,  70),
}

FONT      = cv2.FONT_HERSHEY_DUPLEX
FONT_MONO = cv2.FONT_HERSHEY_PLAIN

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
#  UI Drawing helpers
# ===========================================================================

def draw_header(frame, mode):
    draw_gradient_rect(frame, 0, 0, UI_W, HEADER_H, C["header_top"], C["header_bot"])
    cv2.line(frame, (0, HEADER_H), (UI_W, HEADER_H), C["accent"], 2)
    put_text_shadow(frame, "AIR AI CONTROLLER", (LEFT_W + 20, 44), FONT, 0.90, C["white"], 2)

    mode_colors = {"MOUSE": C["green"], "DRAW": C["accent"]}
    badge_color = mode_colors.get(mode, C["white"])
    label = f"  {mode} MODE  "
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.65, 1)
    bx = UI_W - RIGHT_W - tw - 30
    draw_rounded_rect(frame, bx - 8, 16, bx + tw + 8, 54, 8, badge_color, -1)
    cv2.putText(frame, label, (bx, 44), FONT, 0.65, C["bg"], 2, cv2.LINE_AA)

def draw_left_panel(frame, mode):
    cv2.rectangle(frame, (0, HEADER_H), (LEFT_W, UI_H), C["panel"], -1)
    cv2.line(frame, (LEFT_W, HEADER_H), (LEFT_W, UI_H), C["divider"], 1)

    y = HEADER_H + 35
    put_text_shadow(frame, "CONTROLS", (14, y), FONT, 0.52, C["accent"], 1)
    y += 50

    INST = {
        "MOUSE": [
            ("Index", "Move cursor"),
            ("Pinch", "Left Click"),
            ("", ""),
            (" M ", "Mouse Mode"),
            (" D ", "Draw Mode"),
            (" Q ", "Quit"),
        ],
        "DRAW": [
            ("Index", "Draw"),
            ("Idx+Mid", "Pen up"),
            (" C ", "Clear"),
            ("", ""),
            (" M ", "Mouse Mode"),
            (" D ", "Draw Mode"),
            (" Q ", "Quit"),
        ],
    }

    for key, val in INST.get(mode, []):
        if not key and not val:
            y += 10
            continue
        if len(key.strip()) == 1:
            draw_rounded_rect(frame, 12, y - 14, 12 + 40, y + 6, 4, C["accent"], -1)
            cv2.putText(frame, key.strip(), (16, y), FONT_MONO, 1.1, C["bg"], 1, cv2.LINE_AA)
            cv2.putText(frame, val, (60, y), FONT_MONO, 1.1, C["subtext"], 1, cv2.LINE_AA)
        else:
            put_text_shadow(frame, key, (14, y), FONT_MONO, 1.0, C["accent2"], 1)
            if val:
                put_text_shadow(frame, val, (14, y + 18), FONT_MONO, 1.0, C["subtext"], 1)
                y += 18
        y += 32

def draw_right_panel_mouse(frame):
    cv2.rectangle(frame, (UI_W - RIGHT_W, HEADER_H), (UI_W, UI_H), C["panel"], -1)
    cv2.line(frame, (UI_W - RIGHT_W, HEADER_H), (UI_W - RIGHT_W, UI_H), C["divider"], 1)
    y = HEADER_H + 30
    put_text_shadow(frame, "MOUSE MODE", (UI_W - RIGHT_W + 14, y), FONT, 0.48, C["green"], 1)
    y += 30
    tips = ["Point index finger", "to move cursor.", "", "Pinch thumb and", "index together", "to left-click."]
    for line in tips:
        if line: cv2.putText(frame, line, (UI_W - RIGHT_W + 12, y), FONT_MONO, 1.0, C["subtext"], 1, cv2.LINE_AA)
        y += 20

def draw_right_panel_draw(frame, state, palette_rects, brush_rects, high_rect):
    cv2.rectangle(frame, (UI_W - RIGHT_W, HEADER_H), (UI_W, UI_H), C["panel"], -1)
    cv2.line(frame, (UI_W - RIGHT_W, HEADER_H), (UI_W - RIGHT_W, UI_H), C["divider"], 1)

    px, y = UI_W - RIGHT_W + 14, HEADER_H + 26
    put_text_shadow(frame, "PALETTE", (px, y), FONT, 0.45, C["accent"], 1)
    y += 16

    palette_rects.clear()
    col_w = (RIGHT_W - 28) // 2
    for i, (name, bgr) in enumerate(DRAW_COLORS):
        col, row = i % 2, i // 2
        sx, sy = px + col * (col_w + 4), y + row * 38
        ex, ey = sx + col_w, sy + 30
        
        if i == state["color_idx"]:
            draw_rounded_rect(frame, sx - 3, sy - 3, ex + 3, ey + 3, 7, C["white"], 1)
            
        draw_rounded_rect(frame, sx, sy, ex, ey, 6, bgr, -1)
        palette_rects.append((sx, sy, ex, ey, i))
        cv2.putText(frame, name[:6], (sx + 4, sy + 20), FONT_MONO, 0.85, (20,20,20), 1, cv2.LINE_AA)

    y += (len(DRAW_COLORS) // 2) * 38 + 14
    cv2.line(frame, (px - 4, y), (UI_W - 8, y), C["divider"], 1)
    y += 18

    put_text_shadow(frame, "BRUSH SIZE", (px, y), FONT, 0.45, C["accent"], 1)
    y += 20
    brush_rects.clear()
    bx = px
    for i, sz in enumerate(BRUSH_SIZES):
        cx, cy = bx + i * 40 + sz, y + 16
        cv2.circle(frame, (cx, cy), sz, C["white"], -1)
        if i == state["brush_idx"]: 
            cv2.circle(frame, (cx, cy), sz + 3, C["accent"], 2)
        brush_rects.append((cx - sz - 3, cy - sz - 3, cx + sz + 3, cy + sz + 3, i))
        bx += 4

    y += 50
    cv2.line(frame, (px - 4, y), (UI_W - 8, y), C["divider"], 1)
    y += 18
    
    put_text_shadow(frame, "HIGHLIGHTER", (px, y), FONT, 0.45, C["accent"], 1)
    y += 10
    high_rect[:] = [px, y, px + 80, y + 28]
    tog_c = C["accent2"] if state["high_on"] else C["divider"]
    draw_rounded_rect(frame, px, y, px + 80, y + 28, 8, tog_c, -1)
    label = "ON" if state["high_on"] else "OFF"
    cv2.putText(frame, label, (px + 20, y + 20), FONT, 0.5, C["bg"] if state["high_on"] else C["white"], 1, cv2.LINE_AA)

def draw_status_bar(frame, fps, hand_ok):
    y = UI_H - 14
    fps_color = C["green"] if fps >= 24 else C["red"]
    cv2.putText(frame, f"FPS: {fps:.0f}", (LEFT_W + 10, y), FONT_MONO, 1.1, fps_color, 1, cv2.LINE_AA)
    hand_color = C["green"] if hand_ok else C["red"]
    cv2.putText(frame, "Hand: DETECTED" if hand_ok else "Hand: NOT FOUND", (LEFT_W + 100, y), FONT_MONO, 1.1, hand_color, 1, cv2.LINE_AA)

# ===========================================================================
#  Main Application
# ===========================================================================

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    tracker = HandTracker(max_hands=1); smoother = SmoothCursor(alpha=0.25)
    debounce = ClickDebounce(cooldown_frames=18)

    state = {"mode": "MOUSE", "color_idx": 0, "brush_idx": 1, "high_on": False,
             "palette_rects": [], "brush_rects": [], "high_rect": [0,0,0,0]}

    cam_region_w, cam_region_h = CAM_X2 - CAM_X1, CAM_Y2 - CAM_Y1
    canvas = np.zeros((cam_region_h, cam_region_w, 3), dtype=np.uint8)
    prev_draw_x = prev_draw_y = 0
    t_prev = time.time(); fps_list = []

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, UI_W, UI_H)

    def mouse_cb(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or state["mode"] != "DRAW": return
        for (x1, y1, x2, y2, i) in state["palette_rects"]:
            if x1 <= x <= x2 and y1 <= y <= y2: state["color_idx"] = i; return
        for (x1, y1, x2, y2, i) in state["brush_rects"]:
            if x1 <= x <= x2 and y1 <= y <= y2: state["brush_idx"] = i; return
        if state["high_rect"][0] <= x <= state["high_rect"][2] and state["high_rect"][1] <= y <= state["high_rect"][3]:
            state["high_on"] = not state["high_on"]

    cv2.setMouseCallback(WIN_NAME, mouse_cb)

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        frame = tracker.find_hands(frame, draw=True)
        fingers, hand_ok = tracker.fingers_up(), tracker.hand_detected()
        cam_display = cv2.resize(frame, (cam_region_w, cam_region_h))

        mode = state["mode"]

        if mode == "MOUSE" and hand_ok:
            pos = tracker.get_landmark(8)
            if pos:
                sx = int(interpolate(pos[0], MOUSE_PAD_X1, MOUSE_PAD_X2, 0, SCR_W))
                sy = int(interpolate(pos[1], MOUSE_PAD_Y1, MOUSE_PAD_Y2, 0, SCR_H))
                sx, sy = smoother.update(sx, sy)
                pyautogui.moveTo(sx, sy)
                dist, mx, my = tracker.distance_between(4, 8)
                if dist < 38 and debounce.ready():
                    pyautogui.click(); debounce.trigger()
                    cmx, cmy = int(interpolate(mx, 0, CAM_W, 0, cam_region_w)), int(interpolate(my, 0, CAM_H, 0, cam_region_h))
                    cv2.circle(cam_display, (cmx, cmy), 22, C["green"], 3)

        elif mode == "DRAW" and hand_ok:
            pos8 = tracker.get_landmark(8)
            if pos8:
                cx, cy = int(interpolate(pos8[0], 0, CAM_W, 0, cam_region_w)), int(interpolate(pos8[1], 0, CAM_H, 0, cam_region_h))
                if fingers[1] == 1 and fingers[2] == 1:
                    prev_draw_x = prev_draw_y = 0
                    cv2.circle(cam_display, (cx, cy), 14, C["accent2"], 2)
                elif fingers[1] == 1 and fingers[2] == 0:
                    brush_r, color = BRUSH_SIZES[state["brush_idx"]], DRAW_COLORS[state["color_idx"]][1]
                    if prev_draw_x == 0: prev_draw_x, prev_draw_y = cx, cy
                    
                    if state["high_on"]:
                        glow = tuple(min(255, val + 60) for val in color)
                        cv2.line(canvas, (prev_draw_x, prev_draw_y), (cx, cy), glow, brush_r + 6, cv2.LINE_AA)
                        
                    cv2.line(canvas, (prev_draw_x, prev_draw_y), (cx, cy), color, brush_r, cv2.LINE_AA)
                    prev_draw_x, prev_draw_y = cx, cy
                    cv2.circle(cam_display, (cx, cy), brush_r + 4, color, 2)
                else: prev_draw_x = prev_draw_y = 0

        mask = canvas.astype(bool).any(axis=2)
        if mask.any(): cam_display[mask] = cv2.addWeighted(cam_display, 0.2, canvas, 0.8, 0)[mask]

        ui = np.full((UI_H, UI_W, 3), C["bg"], dtype=np.uint8)
        ui[CAM_Y1:CAM_Y2, CAM_X1:CAM_X2] = cam_display
        draw_header(ui, mode); draw_left_panel(ui, mode)
        if mode == "MOUSE": draw_right_panel_mouse(ui)
        else: draw_right_panel_draw(ui, state, state["palette_rects"], state["brush_rects"], state["high_rect"])

        t_now = time.time(); fps = 1.0 / max(t_now - t_prev, 1e-6); t_prev = t_now
        fps_list.append(fps); avg_fps = sum(fps_list[-15:]) / len(fps_list[-15:])
        draw_status_bar(ui, avg_fps, hand_ok)

        cv2.imshow(WIN_NAME, ui)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('m'): state["mode"] = "MOUSE"; smoother.reset()
        elif key == ord('d'): state["mode"] = "DRAW"; prev_draw_x = prev_draw_y = 0
        elif key == ord('c') and state["mode"] == "DRAW": canvas[:] = 0; prev_draw_x = prev_draw_y = 0

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()