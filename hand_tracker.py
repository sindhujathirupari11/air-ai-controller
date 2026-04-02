"""
hand_tracker.py
---------------
Handles all MediaPipe hand detection and landmark extraction.
MediaPipe is used because it provides real-time 21-point hand landmark
detection running on CPU, making it lightweight and deployable without GPU.
"""

import cv2
import mediapipe as mp
import numpy as np


class HandTracker:
    """
    Wraps MediaPipe Hands solution to detect and track a single hand.
    
    MediaPipe returns 21 landmarks per hand in normalized [0,1] coordinates.
    We convert them to pixel coordinates for use with OpenCV and PyAutoGUI.
    
    Landmark indices (key ones):
        0  = WRIST
        4  = THUMB_TIP
        8  = INDEX_FINGER_TIP
        12 = MIDDLE_FINGER_TIP
        16 = RING_FINGER_TIP
        20 = PINKY_TIP
        
        MCP joints: 5, 9, 13, 17  (knuckle base)
        PIP joints: 6, 10, 14, 18
    """

    # Tip landmark IDs for each finger
    FINGER_TIPS = [4, 8, 12, 16, 20]
    # PIP (second joint) landmark IDs — used to check if finger is up
    FINGER_PIPS = [3, 6, 10, 14, 18]

    def __init__(self, max_hands: int = 1, detection_confidence: float = 0.8,
                 tracking_confidence: float = 0.7):
        """
        Args:
            max_hands: Maximum number of hands to detect (1 keeps CPU usage low).
            detection_confidence: Minimum confidence to detect a hand.
            tracking_confidence: Minimum confidence to continue tracking.
        """
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_style = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,        # video mode → runs tracking, not detection every frame
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

        self.results      = None            # latest MediaPipe inference results
        self.landmarks    = []              # list of (x_px, y_px) for all 21 landmarks
        self.hand_visible = False           # True when a hand is detected this frame

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Run hand detection on `frame`.
        
        MediaPipe requires RGB input; OpenCV delivers BGR, so we convert.
        We set writeable=False before processing for a small performance gain.
        
        Returns the original BGR frame (for further drawing).
        """
        h, w = frame.shape[:2]

        # BGR → RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        self.results = self.hands.process(rgb)
        rgb.flags.writeable = True

        self.landmarks    = []
        self.hand_visible = False

        if self.results.multi_hand_landmarks:
            # Use the first detected hand only
            hand_lms = self.results.multi_hand_landmarks[0]
            self.hand_visible = True

            for lm in hand_lms.landmark:
                # Convert normalised [0,1] → pixel coords
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                self.landmarks.append((cx, cy))

        return frame

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def draw_landmarks(self, frame: np.ndarray,
                       draw_connections: bool = True) -> np.ndarray:
        """
        Overlay MediaPipe's built-in hand skeleton on the frame.
        Uses the default MediaPipe drawing utility for quick, clean results.
        """
        if self.results and self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw_connections:
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_lms,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_style.get_default_hand_landmarks_style(),
                        self.mp_style.get_default_hand_connections_style(),
                    )
                else:
                    self.mp_draw.draw_landmarks(frame, hand_lms)
        return frame

    # ------------------------------------------------------------------
    # Gesture helpers
    # ------------------------------------------------------------------

    def get_landmark(self, index: int):
        """Return (x, y) pixel position of a specific landmark, or None."""
        if 0 <= index < len(self.landmarks):
            return self.landmarks[index]
        return None

    def fingers_up(self) -> list[int]:
        """
        Return a list [thumb, index, middle, ring, pinky] where 1 = finger extended.
        
        Thumb: compared horizontally (x-axis) since it moves sideways.
        Other fingers: tip y < pip y means finger is pointing up (image coords).
        
        This works for a right hand facing the camera; mirrored for left hand.
        """
        if len(self.landmarks) < 21:
            return [0, 0, 0, 0, 0]

        fingers = []

        # --- Thumb (landmark 4 vs 3, check x axis) ---
        # We check if thumb tip is to the left of its lower joint.
        # Works because frame is mirrored (selfie view).
        if self.landmarks[4][0] < self.landmarks[3][0]:
            fingers.append(1)
        else:
            fingers.append(0)

        # --- Four fingers (tip y < pip y) ---
        for tip, pip in zip(self.FINGER_TIPS[1:], self.FINGER_PIPS[1:]):
            if self.landmarks[tip][1] < self.landmarks[pip][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def bounding_box(self, padding: int = 20):
        """
        Return (x, y, w, h) bounding box around all detected landmarks.
        Useful for drawing a hand outline.
        """
        if not self.landmarks:
            return None
        xs = [lm[0] for lm in self.landmarks]
        ys = [lm[1] for lm in self.landmarks]
        x1 = max(0, min(xs) - padding)
        y1 = max(0, min(ys) - padding)
        x2 = min(9999, max(xs) + padding)
        y2 = min(9999, max(ys) + padding)
        return x1, y1, x2 - x1, y2 - y1
