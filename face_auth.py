"""face_auth.py — VitalNav face biometric authentication
Uses OpenCV Haar cascade (face detection) + multi-metric comparison:
  1. LBPH histogram distance  (Local Binary Patterns — primary identity signal)
  2. Structural Similarity     (SSIM on greyscale patch)
  3. Colour histogram          (Bhattacharyya distance, supplemental)

Match threshold : 0.70  (70 %)
Center crop     : live frame is cropped to the centre square that matches
                  the blue guide box shown in the UI (52 % of frame width)
                  BEFORE face detection — faces outside the box are ignored.

All processing is local / CPU-only — no external model downloads needed.
"""
from __future__ import annotations

import base64
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Cascade models ────────────────────────────────────────────────────────────
_CASCADE     = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_CASCADE_ALT = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

FACE_SIZE       = 128   # every detected face patch is normalised to this square (px)
MATCH_THRESHOLD = 0.70  # minimum weighted score to accept a match


# ── Image helpers ─────────────────────────────────────────────────────────────

def _bytes_to_bgr(raw: bytes) -> np.ndarray | None:
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)   # None on failure


def _b64_to_bgr(b64: str) -> np.ndarray | None:
    try:
        return _bytes_to_bgr(base64.b64decode(b64))
    except Exception:
        return None


def _crop_center_square(img: np.ndarray, fraction: float = 0.52) -> np.ndarray:
    """
    Crop the centre square from *img*.

    fraction  — side length as a fraction of the image's shorter dimension.
                0.52 matches the CSS ::after overlay in auth.py (52 % of widget width).
    The Y centre is shifted slightly upward (×0.48) to match the CSS
    translate(-50%, -54%) offset used in the guide box.
    """
    h, w  = img.shape[:2]
    side  = int(min(h, w) * fraction)
    cx    = w // 2
    cy    = int(h * 0.48)           # ~48 % from top  ≈  CSS -54 % shift
    x1    = max(0, cx - side // 2)
    y1    = max(0, cy - side // 2)
    x2    = min(w, x1 + side)
    y2    = min(h, y1 + side)
    return img[y1:y2, x1:x2]


def _detect_face(img_bgr: np.ndarray) -> np.ndarray | None:
    """
    Detect the largest frontal face in *img_bgr*.
    Returns a normalised FACE_SIZE × FACE_SIZE BGR patch, or None if not found.
    """
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    faces = np.array([])
    for cascade in (_CASCADE, _CASCADE_ALT):
        faces = cascade.detectMultiScale(
            gray_eq, scaleFactor=1.1, minNeighbors=5,
            minSize=(48, 48), flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(faces) > 0:
            break

    # Relaxed fallback
    if len(faces) == 0:
        faces = _CASCADE.detectMultiScale(
            gray_eq, scaleFactor=1.05, minNeighbors=3, minSize=(32, 32),
        )

    if len(faces) == 0:
        return None

    # Use the largest bounding box
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad = int(0.15 * min(w, h))
    x1  = max(0, x - pad)
    y1  = max(0, y - pad)
    x2  = min(img_bgr.shape[1], x + w + pad)
    y2  = min(img_bgr.shape[0], y + h + pad)
    return cv2.resize(img_bgr[y1:y2, x1:x2], (FACE_SIZE, FACE_SIZE))


# ── Similarity metrics ────────────────────────────────────────────────────────

def _lbph_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """LBP histogram chi-squared distance → 0-1 similarity (1 = identical)."""
    def lbp_hist(gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape
        lbp  = np.zeros_like(gray, dtype=np.uint8)
        for bit, (dy, dx) in enumerate([(-1,-1),(-1,0),(-1,1),(0,1),
                                         (1,1),(1,0),(1,-1),(0,-1)]):
            shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
            lbp    |= ((gray >= shifted).astype(np.uint8) << bit)
        cell_h, cell_w = h // 4, w // 4
        hist = []
        for r in range(4):
            for c in range(4):
                cell = lbp[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
                hc, _ = np.histogram(cell.flatten(), bins=256, range=(0, 256))
                hist.append(hc.astype(np.float32))
        hist = np.concatenate(hist)
        hist /= hist.sum() + 1e-7
        return hist

    ha   = lbp_hist(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY))
    hb   = lbp_hist(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY))
    chi2 = float(cv2.compareHist(ha, hb, cv2.HISTCMP_CHISQR))
    return float(np.clip(1.0 - chi2 / 8.0, 0.0, 1.0))


def _colour_hist_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Per-channel Bhattacharyya distance → 0-1 similarity."""
    scores = []
    for ch in range(3):
        ha = cv2.calcHist([a], [ch], None, [64], [0, 256])
        hb = cv2.calcHist([b], [ch], None, [64], [0, 256])
        cv2.normalize(ha, ha)
        cv2.normalize(hb, hb)
        scores.append(1.0 - float(cv2.compareHist(ha, hb, cv2.HISTCMP_BHATTACHARYYA)))
    return float(np.mean(scores))


def _ssim_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Simplified SSIM on greyscale → 0-1 similarity."""
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    C1, C2 = 6.5025, 58.5225
    k      = (11, 11)
    mu_a   = cv2.GaussianBlur(ga,     k, 1.5)
    mu_b   = cv2.GaussianBlur(gb,     k, 1.5)
    mu_a2, mu_b2, mu_ab = mu_a**2, mu_b**2, mu_a * mu_b
    sig_a2 = cv2.GaussianBlur(ga**2,  k, 1.5) - mu_a2
    sig_b2 = cv2.GaussianBlur(gb**2,  k, 1.5) - mu_b2
    sig_ab = cv2.GaussianBlur(ga*gb,  k, 1.5) - mu_ab
    ssim   = ((2*mu_ab + C1) * (2*sig_ab + C2)) / \
             ((mu_a2 + mu_b2 + C1) * (sig_a2 + sig_b2 + C2))
    return float(np.clip(ssim.mean(), 0.0, 1.0))


# ── Public API ────────────────────────────────────────────────────────────────

def compare_faces(
    stored_b64: str,
    live_bytes: bytes,
    crop_center: bool = True,
) -> tuple[bool, float, str]:
    """
    Compare the stored signup photo (base64) with a live camera frame (raw bytes).

    Parameters
    ----------
    stored_b64  : base64-encoded image saved at signup.
    live_bytes  : raw JPEG/PNG bytes from st.camera_input().
    crop_center : if True (default), crops the live frame to the centre square
                  before detection — only the face inside the blue UI guide box
                  is matched; anything outside is ignored.

    Returns
    -------
    (matched, score, message)
      matched – True when score >= MATCH_THRESHOLD (0.70)
      score   – float 0-1
      message – human-readable result string
    """
    stored_img = _b64_to_bgr(stored_b64)
    if stored_img is None:
        return False, 0.0, "Could not decode stored profile photo."

    live_img = _bytes_to_bgr(live_bytes)
    if live_img is None:
        return False, 0.0, "Could not decode camera frame."

    # Crop live frame to the centre square shown by the UI guide box
    if crop_center:
        live_img = _crop_center_square(live_img)

    face_stored = _detect_face(stored_img)
    if face_stored is None:
        return False, 0.0, "No face found in your stored signup photo. Please update it."

    face_live = _detect_face(live_img)
    if face_live is None:
        return False, 0.0, "No face detected inside the square. Look straight at the camera."

    # Weighted ensemble
    lbph_s   = _lbph_similarity(face_stored, face_live)
    ssim_s   = _ssim_similarity(face_stored, face_live)
    colour_s = _colour_hist_similarity(face_stored, face_live)
    score    = 0.55 * lbph_s + 0.25 * ssim_s + 0.20 * colour_s

    logger.debug(
        "FaceAuth  LBPH=%.3f  SSIM=%.3f  Colour=%.3f  Final=%.3f  threshold=%.2f",
        lbph_s, ssim_s, colour_s, score, MATCH_THRESHOLD,
    )

    if score >= MATCH_THRESHOLD:
        return True, score, f"Face matched ({score * 100:.0f}% similarity)."
    return (
        False, score,
        f"Face did not match ({score * 100:.0f}% — need >= {MATCH_THRESHOLD * 100:.0f}%).",
    )


def has_face(image_bytes: bytes) -> bool:
    """Quick check — does this image contain at least one detectable face?"""
    img = _bytes_to_bgr(image_bytes)
    return img is not None and _detect_face(img) is not None