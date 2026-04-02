"""
hand_tracker.py
---------------
Handles MediaPipe hand detection and landmark extraction.
Provides finger state detection and landmark coordinates.
"""

import cv2
import mediapipe as mp
import math


class HandTracker:
    """
    Wraps MediaPipe Hands for easy landmark access and finger state detection.
    """

    # MediaPipe landmark indices for fingertips and pip joints
    TIP_IDS  = [4, 8, 12, 16, 20]   # Thumb, Index, Middle, Ring, Pinky tips
    PIP_IDS  = [3, 6, 10, 14, 18]   # Corresponding second joints

    def __init__(self, max_hands=1, detection_conf=0.75, tracking_conf=0.75):
        self.mp_hands   = mp.solutions.hands
        self.mp_draw    = mp.solutions.drawing_utils
        self.mp_styles  = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            max_num_hands         = max_hands,
            min_detection_confidence = detection_conf,
            min_tracking_confidence  = tracking_conf,
        )

        self.landmarks   = []   # Raw landmark objects (normalized 0-1)
        self.lm_list     = []   # Pixel-coordinate list [(id, x, y), ...]
        self.h = self.w  = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_hands(self, frame, draw=True):
        """
        Process a BGR frame, optionally draw landmarks, return annotated frame.
        Must be called before using any landmark accessors.
        """
        self.h, self.w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        self.landmarks = []
        self.lm_list   = []

        if results.multi_hand_landmarks:
            # Use the first detected hand only
            hand_lms = results.multi_hand_landmarks[0]
            self.landmarks = hand_lms.landmark

            for idx, lm in enumerate(hand_lms.landmark):
                px = int(lm.x * self.w)
                py = int(lm.y * self.h)
                self.lm_list.append((idx, px, py))

            if draw:
                self.mp_draw.draw_landmarks(
                    frame, hand_lms,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style(),
                )

        return frame

    def get_landmark(self, idx):
        """Return (x, y) pixel coords for landmark idx, or None if not found."""
        if self.lm_list and idx < len(self.lm_list):
            _, x, y = self.lm_list[idx]
            return x, y
        return None

    def fingers_up(self):
        """
        Return a list [thumb, index, middle, ring, pinky] where 1 = extended.
        Works for right hand (mirrors for left automatically via x-axis check).
        """
        if not self.lm_list:
            return [0, 0, 0, 0, 0]

        up = []

        # Thumb: compare tip x to ip joint x (horizontal logic)
        thumb_tip = self.lm_list[4][1]
        thumb_ip  = self.lm_list[3][1]
        wrist_x   = self.lm_list[0][1]
        # Hand faces camera → thumb extends away from wrist
        if thumb_tip < wrist_x:          # Right hand, thumb extends left
            up.append(1 if thumb_tip < thumb_ip else 0)
        else:                            # Left hand, thumb extends right
            up.append(1 if thumb_tip > thumb_ip else 0)

        # Other four fingers: tip y < pip y means extended (y increases downward)
        for tip, pip in zip(self.TIP_IDS[1:], self.PIP_IDS[1:]):
            tip_y = self.lm_list[tip][2]
            pip_y = self.lm_list[pip][2]
            up.append(1 if tip_y < pip_y else 0)

        return up

    def distance_between(self, idx1, idx2):
        """
        Return Euclidean pixel distance between two landmarks.
        Returns (distance, midpoint_x, midpoint_y).
        """
        p1 = self.get_landmark(idx1)
        p2 = self.get_landmark(idx2)
        if p1 is None or p2 is None:
            return 0, 0, 0

        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        mid  = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        return dist, mid[0], mid[1]

    def hand_detected(self):
        """True if at least one hand is currently tracked."""
        return len(self.lm_list) > 0