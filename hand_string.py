#!/usr/bin/env python3
"""
hand_string.py — real-time flowing letter strings between matching fingertips,
with face mesh tracking and cyberpunk CRT aesthetic.

Letters flow along sinusoidal paths distorted by noise, with a Purple→Pink→Red
gradient. Inspired by p5.js flowing-poster technique.

One string per finger pair (thumb↔thumb … pinky↔pinky).
Controls: Q or ESC to quit, R to randomise pattern.
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time
import random
import sys

# ── Palette (RGB) — the ONLY three allowed colours ───────────────────────────
PALETTE = [
    (153,  65, 255),   # Purple
    (255,  94, 243),   # Pink
    (235,   0,  40),   # Red
]

# Fingertip landmark indices (MediaPipe 21-point hand model)
FINGER_TIPS = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky

# ── Tunables ───────────────────────────────────────────────────────────────────
BASE_WORD = "handstring"

# Flowing letters — band layout
N_BANDS          = 7        # parallel sinusoidal bands per string
BAND_SPACING     = 14       # px between adjacent bands
LETTERS_PER_BAND = 30       # letters on each band

# Flow speed (fraction of band traversed per second)
FLOW_SPEED_MIN = 0.03
FLOW_SPEED_MAX = 0.09

# Wave distortion (scaled by hand distance at draw time)
WAVE_AMP_FACTOR = 0.15      # amplitude as fraction of distance
WAVE_FREQ       = 4.0       # full sine cycles across band

# Letter rendering
LETTER_FONT_SCALE = 0.45    # cv2.putText font scale

# Position smoothing (exponential moving average)
SMOOTH_ALPHA = 0.45         # 0 = max smooth, 1 = no smoothing

# One-hand trailing
TRAIL_SMOOTH     = 0.035    # very heavy lag for trail follower
MIN_TRAIL_DIST   = 130.0    # minimum px between tip and trail point

# CRT / cyberpunk overlay
SCANLINE_SPACING = 3        # every N-th row gets darkened
SCANLINE_ALPHA   = 0.88     # brightness multiplier on scanlines

# Cyberpunk font — DUPLEX has a clean, technical, stencil-ish look
CYBER_FONT = cv2.FONT_HERSHEY_DUPLEX

# MediaPipe Face Mesh — face oval contour indices
_FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10,
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _smooth_pt(old: tuple[float, float] | None, new: tuple[int, int],
               alpha: float) -> tuple[float, float]:
    """Exponential moving average for a 2-D point."""
    if old is None:
        return (float(new[0]), float(new[1]))
    return (_lerp(old[0], float(new[0]), alpha),
            _lerp(old[1], float(new[1]), alpha))


def _noise(x: float, y: float = 0.0) -> float:
    """Cheap pseudo Perlin noise via layered sinusoids. Returns ~[0, 1]."""
    v = (math.sin(x * 1.27 + y * 0.73 + 3.17) +
         math.sin(x * 0.43 + y * 1.37 + 7.13) * 0.7 +
         math.sin(x * 2.17 + y * 0.31 + 1.71) * 0.4)
    return max(0.0, min(1.0, v / 4.2 + 0.5))


def _gradient_color_bgr(t: float) -> tuple[int, int, int]:
    """Purple (t=0) → Pink (t=0.5) → Red (t=1).  Returns BGR for OpenCV."""
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        f = t * 2.0
        r = int(153 + (255 - 153) * f)
        g = int(65  + (94  - 65)  * f)
        b = int(255 + (243 - 255) * f)
    else:
        f = (t - 0.5) * 2.0
        r = int(255 + (235 - 255) * f)
        g = int(94  - 94  * f)
        b = int(243 + (40  - 243) * f)
    return (b, g, r)


# ── Face wireframe (minimal, hand-skeleton style) ─────────────────────────────

_FACE_WIREFRAME = [
    # Face oval (sparse — 8 key points)
    (10, 338), (338, 284), (284, 323), (323, 152),
    (152, 93),  (93, 132),  (132, 162), (162, 10),
    # Right eye
    (33, 160), (160, 158), (158, 133), (133, 153), (153, 144), (144, 33),
    # Left eye
    (362, 385), (385, 387), (387, 263), (263, 373), (373, 380), (380, 362),
    # Right eyebrow
    (70, 63), (63, 105), (105, 66), (66, 107),
    # Left eyebrow
    (336, 296), (296, 334), (334, 293), (293, 300),
    # Nose bridge + wings
    (168, 6), (6, 4), (48, 4), (4, 278),
    # Lips (outer, simplified)
    (61, 0), (0, 291), (291, 17), (17, 61),
]

_FACE_KEY_INDICES = sorted({i for pair in _FACE_WIREFRAME for i in pair})


def _draw_face_mesh(frame: np.ndarray, face_landmarks, w: int, h: int,
                    color_bgr: tuple = (255, 255, 255)) -> None:
    """Draw a clean, minimal face wireframe (dots + lines) like the hand skeleton."""
    for face_lm in face_landmarks:
        n_lm = len(face_lm)
        pts = {i: (int(face_lm[i].x * w), int(face_lm[i].y * h))
               for i in _FACE_KEY_INDICES if i < n_lm}
        for a, b in _FACE_WIREFRAME:
            if a in pts and b in pts:
                cv2.line(frame, pts[a], pts[b], color_bgr, 1, cv2.LINE_AA)
        for p in pts.values():
            cv2.circle(frame, p, 3, color_bgr, -1, cv2.LINE_AA)


# ── Cyberpunk visual effects ───────────────────────────────────────────────────

def _draw_scanlines(frame: np.ndarray) -> None:
    """Overlay subtle CRT-style horizontal scanlines."""
    h = frame.shape[0]
    for y in range(0, h, SCANLINE_SPACING):
        frame[y] = (frame[y].astype(np.float32) * SCANLINE_ALPHA).astype(np.uint8)


def _draw_hud_label(
    frame: np.ndarray, text: str, center: tuple[int, int],
    scale: float, color_bgr: tuple,
) -> None:
    """Draw a HUD-style label like [LEFT HAND], centred on the given point."""
    hud_text = f"[{text.upper()}]"
    (tw, _th), _baseline = cv2.getTextSize(hud_text, CYBER_FONT, scale, 1)
    ox = center[0] - tw // 2
    oy = center[1]
    cv2.putText(frame, hud_text, (ox, oy), CYBER_FONT, scale,
                color_bgr, 1, cv2.LINE_AA)


# ── Flowing letter string (p5.js-inspired) ─────────────────────────────────────

class ElasticString:
    """
    A pool of letters flowing along parallel sinusoidal bands between two
    endpoints, distorted by pseudo-noise.  Colour gradients from Purple →
    Pink → Red based on position along the band.
    """

    def __init__(self):
        self.clock   = 0.0
        self.prev_dist: float | None = None
        self._sp1: tuple[float, float] | None = None
        self._sp2: tuple[float, float] | None = None
        self._noise_offset = random.random() * 100.0
        self._trail: tuple[float, float] | None = None

        # Build letter pool
        self.letters: list[dict] = []
        for band_idx in range(N_BANDS):
            offset = (band_idx - (N_BANDS - 1) / 2.0) * BAND_SPACING
            for j in range(LETTERS_PER_BAND):
                self.letters.append({
                    "t":     j / LETTERS_PER_BAND + random.uniform(-0.02, 0.02),
                    "band":  offset,
                    "char":  BASE_WORD[j % len(BASE_WORD)],
                    "speed": random.uniform(FLOW_SPEED_MIN, FLOW_SPEED_MAX),
                    "seed":  random.random() * 100.0,
                })

    # ── Advance ───────────────────────────────────────────────────────────────

    def step(self, dist: float, dt: float) -> None:
        self.prev_dist = dist
        self.clock += dt
        for ltr in self.letters:
            ltr["t"] = (ltr["t"] + ltr["speed"] * dt) % 1.0

    # ── Draw ──────────────────────────────────────────────────────────────────

    def draw(self, frame: np.ndarray, p1_raw, p2_raw) -> np.ndarray:
        # Smooth endpoints
        self._sp1 = _smooth_pt(self._sp1, p1_raw, SMOOTH_ALPHA)
        self._sp2 = _smooth_pt(self._sp2, p2_raw, SMOOTH_ALPHA)
        p1 = np.array(self._sp1, dtype=float)
        p2 = np.array(self._sp2, dtype=float)

        direction = p2 - p1
        dist = np.linalg.norm(direction)
        if dist < 5:
            return frame

        tang = direction / dist
        perp = np.array([-tang[1], tang[0]])

        # Wave amplitude and frequency scale with distance
        wave_amp  = dist * WAVE_AMP_FACTOR
        wave_freq = WAVE_FREQ + dist * 0.003   # slightly more cycles when far

        fh, fw = frame.shape[:2]

        for ltr in self.letters:
            t = ltr["t"]

            # Position along main axis
            pos = p1 + t * direction

            # ── Noise + sine distortion (adapted from p5.js) ──────────────
            nx = t * 3.0 + ltr["seed"]
            ny = ltr["band"] * 0.005 + self._noise_offset
            n  = _noise(nx, ny)
            wave = math.sin(
                t * math.pi * wave_freq
                + self.clock * 2.0
                + n * 10.0
            ) * wave_amp

            # Total perpendicular offset: band position + wave × noise
            total_offset = ltr["band"] + wave * n
            final = pos + total_offset * perp

            fx, fy = int(final[0]), int(final[1])

            # Cull if off-screen
            if fx < -20 or fx > fw + 20 or fy < -20 or fy > fh + 20:
                continue

            # 3-colour gradient based on position along band
            color = _gradient_color_bgr(t)

            cv2.putText(frame, ltr["char"], (fx, fy),
                        CYBER_FONT, LETTER_FONT_SCALE, color,
                        1, cv2.LINE_AA)

        return frame

    # ── Interaction ───────────────────────────────────────────────────────────

    def new_color(self) -> None:
        """R key — shift the noise pattern for a new ripple shape."""
        self._noise_offset += random.uniform(10, 50)

    # ── One-hand trailing ─────────────────────────────────────────────────────

    def update_and_get_trail(self, tip: tuple[int, int],
                             which_missing: str = "right"
                             ) -> tuple[int, int]:
        """
        Return a point that trails behind *tip* with heavy lag.

        On first call after two-hand → one-hand transition the trail
        initialises at the last smoothed position of the *missing* hand
        so the stream doesn't jump.
        """
        if self._trail is None:
            # Seed from the disappearing hand's last smoothed position
            if which_missing == "right" and self._sp2 is not None:
                self._trail = self._sp2
            elif which_missing == "left" and self._sp1 is not None:
                self._trail = self._sp1
            else:
                self._trail = (float(tip[0]), float(tip[1]) + MIN_TRAIL_DIST)

        # Follow the visible tip with heavy lag
        self._trail = _smooth_pt(self._trail, tip, TRAIL_SMOOTH)

        # Enforce minimum distance (return value only — don't modify _trail)
        tx, ty = self._trail
        dx = float(tip[0]) - tx
        dy = float(tip[1]) - ty
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < MIN_TRAIL_DIST:
            if dist < 1:
                return (tip[0], int(tip[1] + MIN_TRAIL_DIST))
            scale = MIN_TRAIL_DIST / dist
            return (int(float(tip[0]) - dx * scale),
                    int(float(tip[1]) - dy * scale))

        return (int(tx), int(ty))


# ── Model paths ────────────────────────────────────────────────────────────────

_HAND_MODEL_PATH = "hand_landmarker.task"
_HAND_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

_FACE_MODEL_PATH = "face_landmarker.task"
_FACE_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# MediaPipe hand-connection pairs for manual skeleton drawing (21 landmarks)
_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]


def _ensure_file(path: str, url: str) -> None:
    """Download a model file if it doesn't already exist on disk."""
    import os, ssl, shutil, urllib.request
    if not os.path.exists(path):
        print(f"Downloading {path} …", flush=True)
        try:
            resp = urllib.request.urlopen(url)
        except urllib.error.URLError:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            resp = urllib.request.urlopen(url, context=ctx)
        with open(path, "wb") as f:
            shutil.copyfileobj(resp, f)
        resp.close()
        print("Download complete.")


def _draw_skeleton(frame: np.ndarray, landmarks, w: int, h: int,
                   label: str = "",
                   color_bgr: tuple = (255, 255, 255)) -> None:
    """Draw hand skeleton in white and optionally a HUD-style label above."""
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in _HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color_bgr, 1, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 3, color_bgr, -1, cv2.LINE_AA)

    if label and pts:
        wrist = pts[0]
        _draw_hud_label(frame, label, (wrist[0], wrist[1] - 30), 0.55, color_bgr)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Detector setup ────────────────────────────────────────────────────────
    USE_LEGACY = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")

    if USE_LEGACY:
        mp_hands  = mp.solutions.hands
        hands_mdl = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.55,
        )
        draw_lm = mp.solutions.drawing_utils.draw_landmarks
        try:
            lm_style = mp.solutions.drawing_styles.get_default_hand_landmarks_style()
            cn_style = mp.solutions.drawing_styles.get_default_hand_connections_style()
        except AttributeError:
            lm_style = cn_style = None
    else:
        from mediapipe.tasks import python as _mp_py
        from mediapipe.tasks.python import vision as _mp_vis

        _ensure_file(_HAND_MODEL_PATH, _HAND_MODEL_URL)
        _opts = _mp_vis.HandLandmarkerOptions(
            base_options=_mp_py.BaseOptions(model_asset_path=_HAND_MODEL_PATH),
            running_mode=_mp_vis.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.65,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        hands_mdl = _mp_vis.HandLandmarker.create_from_options(_opts)

    # ── Face Mesh setup ───────────────────────────────────────────────────────
    if USE_LEGACY:
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    else:
        _ensure_file(_FACE_MODEL_PATH, _FACE_MODEL_URL)
        _face_opts = _mp_vis.FaceLandmarkerOptions(
            base_options=_mp_py.BaseOptions(model_asset_path=_FACE_MODEL_PATH),
            running_mode=_mp_vis.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        face_mesh = _mp_vis.FaceLandmarker.create_from_options(_face_opts)

    # ── Camera ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Error: could not open webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    # One ElasticString per finger pair
    sims = [ElasticString() for _ in range(len(FINGER_TIPS))]
    prev_t  = time.perf_counter()
    hud_col = (255, 255, 255)

    print("Hand String ready.  Q / ESC = quit   R = new pattern")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        now    = time.perf_counter()
        dt     = min(now - prev_t, 0.05)
        prev_t = now

        frame  = cv2.flip(frame, 1)
        h, w   = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # (face detection disabled — no dots/lines on face)

        # ── Hand landmarks ────────────────────────────────────────────────────
        left_tips:  list[tuple[int, int] | None] = [None] * len(FINGER_TIPS)
        right_tips: list[tuple[int, int] | None] = [None] * len(FINGER_TIPS)

        if USE_LEGACY:
            result = hands_mdl.process(rgb)
            if result.multi_hand_landmarks and result.multi_handedness:
                for lm, hd in zip(result.multi_hand_landmarks,
                                  result.multi_handedness):
                    label = hd.classification[0].label
                    for fi, lm_idx in enumerate(FINGER_TIPS):
                        tip = lm.landmark[lm_idx]
                        px, py = int(tip.x * w), int(tip.y * h)
                        if label == "Left":
                            left_tips[fi] = (px, py)
                        else:
                            right_tips[fi] = (px, py)
                    hand_label = ("Right Hand" if label == "Left"
                                  else "Left Hand")
                    _draw_skeleton(frame, lm.landmark, w, h, label=hand_label)
        else:
            ts_ms  = int(now * 1000)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = hands_mdl.detect_for_video(mp_img, ts_ms)
            if result.hand_landmarks and result.handedness:
                for lm_list, hd_list in zip(result.hand_landmarks,
                                            result.handedness):
                    label = hd_list[0].category_name
                    for fi, lm_idx in enumerate(FINGER_TIPS):
                        tip = lm_list[lm_idx]
                        px, py = int(tip.x * w), int(tip.y * h)
                        if label == "Left":
                            left_tips[fi] = (px, py)
                        else:
                            right_tips[fi] = (px, py)
                    hand_label = ("Right Hand" if label == "Left"
                                  else "Left Hand")
                    _draw_skeleton(frame, lm_list, w, h, label=hand_label)

        # ── Flowing letter strings (one per finger pair) ──────────────────────
        any_active = False
        for fi in range(len(FINGER_TIPS)):
            lt, rt = left_tips[fi], right_tips[fi]
            sim = sims[fi]

            if lt and rt:
                # Both hands — connect fingertips
                sim._trail = None          # reset trail for clean re-entry
                dist = math.dist(lt, rt)
                sim.step(dist, dt)
                frame = sim.draw(frame, lt, rt)
                any_active = True
            elif lt:
                # Only left hand — trail where right was
                trail = sim.update_and_get_trail(lt, "right")
                dist = math.dist(lt, trail)
                sim.step(dist, dt)
                frame = sim.draw(frame, lt, trail)
                any_active = True
            elif rt:
                # Only right hand — trail where left was
                trail = sim.update_and_get_trail(rt, "left")
                dist = math.dist(rt, trail)
                sim.step(dist, dt)
                frame = sim.draw(frame, trail, rt)
                any_active = True
            else:
                # No hands — advance letters but don't draw
                sim.step(sim.prev_dist or 100, dt)
                sim._trail = None

        if not any_active:
            msg = "SHOW BOTH HANDS"
            tw  = cv2.getTextSize(msg, CYBER_FONT, 0.9, 2)[0][0]
            cv2.putText(frame, msg, ((w - tw) // 2, h // 2),
                        CYBER_FONT, 0.9, hud_col, 2, cv2.LINE_AA)

        cv2.putText(frame, "HAND STRING  //  Q/ESC QUIT  //  R PATTERN",
                    (12, 30), CYBER_FONT, 0.5, hud_col, 1, cv2.LINE_AA)

        # ── CRT scanline overlay (applied last) ───────────────────────────
        _draw_scanlines(frame)

        cv2.imshow("Hand String", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord("r"), ord("R")):
            for sim in sims:
                sim.new_color()

    cap.release()
    hands_mdl.close()
    face_mesh.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
