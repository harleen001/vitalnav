"""face_auth.py — VitalNav face biometric authentication
Uses OpenCV Haar cascade (face detection) + multi-metric comparison:
  1. LBPH histogram distance (Local Binary Patterns Histogram)
  2. Colour histogram correlation
  3. Structural similarity on normalised greyscale patch

Match threshold: similarity score >= 0.70  (70%)

All processing is local / CPU-only — no external model downloads.
"""
from __future__ import annotations

import base64
import io
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Cascade ──────────────────────────────────────────────────────────────────
_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
_CASCADE_ALT = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
)

FACE_SIZE   = 128          # pixels — face patch normalised to this square
MATCH_THRESHOLD = 0.70     # 70 % similarity required


# ── Helpers ──────────────────────────────────────────────────────────────────

def _bytes_to_bgr(raw: bytes) -> np.ndarray | None:
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _b64_to_bgr(b64: str) -> np.ndarray | None:
    try:
        raw = base64.b64decode(b64)
        return _bytes_to_bgr(raw)
    except Exception:
        return None


def _detect_face(img_bgr: np.ndarray) -> np.ndarray | None:
    """Return normalised FACE_SIZE×FACE_SIZE BGR patch, or None if no face found."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    for cascade in (_CASCADE, _CASCADE_ALT):
        faces = cascade.detectMultiScale(
            gray_eq, scaleFactor=1.1, minNeighbors=5,
            minSize=(48, 48), flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) > 0:
            break

    if len(faces) == 0:
        # Relax parameters as fallback
        faces = _CASCADE.detectMultiScale(
            gray_eq, scaleFactor=1.05, minNeighbors=3, minSize=(32, 32)
        )

    if len(faces) == 0:
        return None

    # Largest detected face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad = int(0.15 * min(w, h))
    x1 = max(0, x - pad);  y1 = max(0, y - pad)
    x2 = min(img_bgr.shape[1], x + w + pad)
    y2 = min(img_bgr.shape[0], y + h + pad)

    patch = img_bgr[y1:y2, x1:x2]
    return cv2.resize(patch, (FACE_SIZE, FACE_SIZE))


# ── Comparison metrics ────────────────────────────────────────────────────────

def _lbph_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compare LBPH histograms; returns 0-1 (1 = identical)."""
    gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

    # Build LBP histograms manually (radius=1, points=8, grid 4×4)
    def lbp_hist(gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape
        lbp = np.zeros_like(gray, dtype=np.uint8)
        offsets = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
        for bit, (dy, dx) in enumerate(offsets):
            shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
            lbp |= ((gray >= shifted).astype(np.uint8) << bit)
        # 4×4 grid histogram
        cell_h, cell_w = h // 4, w // 4
        hist = []
        for r in range(4):
            for c in range(4):
                cell = lbp[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
                h_, _ = np.histogram(cell.flatten(), bins=256, range=(0, 256))
                hist.append(h_.astype(np.float32))
        hist = np.concatenate(hist)
        hist /= (hist.sum() + 1e-7)
        return hist

    ha = lbp_hist(gray_a)
    hb = lbp_hist(gray_b)

    # Chi-squared distance → convert to similarity
    chi2 = float(cv2.compareHist(ha, hb, cv2.HISTCMP_CHISQR))
    # Normalise: chi2 ~ 0 means identical; cap at 10 for normalisation
    similarity = max(0.0, 1.0 - chi2 / 8.0)
    return float(np.clip(similarity, 0.0, 1.0))


def _colour_hist_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Bhattacharyya-based colour histogram similarity (0-1)."""
    scores = []
    for ch in range(3):
        ha = cv2.calcHist([a], [ch], None, [64], [0, 256])
        hb = cv2.calcHist([b], [ch], None, [64], [0, 256])
        cv2.normalize(ha, ha)
        cv2.normalize(hb, hb)
        d = cv2.compareHist(ha, hb, cv2.HISTCMP_BHATTACHARYYA)
        scores.append(1.0 - float(d))
    return float(np.mean(scores))


def _ssim_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Structural Similarity Index on greyscale (simplified, 0-1)."""
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)

    C1, C2 = 6.5025, 58.5225  # (0.01*255)^2, (0.03*255)^2
    mu_a = cv2.GaussianBlur(ga, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(gb, (11, 11), 1.5)
    mu_a2, mu_b2, mu_ab = mu_a**2, mu_b**2, mu_a * mu_b
    sig_a2 = cv2.GaussianBlur(ga**2, (11, 11), 1.5) - mu_a2
    sig_b2 = cv2.GaussianBlur(gb**2, (11, 11), 1.5) - mu_b2
    sig_ab = cv2.GaussianBlur(ga * gb, (11, 11), 1.5) - mu_ab
    ssim_map = ((2*mu_ab + C1) * (2*sig_ab + C2)) / \
               ((mu_a2 + mu_b2 + C1) * (sig_a2 + sig_b2 + C2))
    return float(np.clip(ssim_map.mean(), 0.0, 1.0))


# ── Public API ────────────────────────────────────────────────────────────────

def compare_faces(
    stored_b64: str,
    live_bytes: bytes,
) -> tuple[bool, float, str]:
    """
    Compare stored signup face (base64) against live camera frame (raw bytes).

    Returns
    -------
    (matched: bool, score: float 0-1, message: str)
    """
    stored_img = _b64_to_bgr(stored_b64)
    if stored_img is None:
        return False, 0.0, "Could not decode stored profile photo."

    live_img = _bytes_to_bgr(live_bytes)
    if live_img is None:
        return False, 0.0, "Could not decode camera frame."

    face_stored = _detect_face(stored_img)
    if face_stored is None:
        return False, 0.0, "No face detected in your stored profile photo. Please update your photo."

    face_live = _detect_face(live_img)
    if face_live is None:
        return False, 0.0, "No face detected in camera frame. Please look directly at the camera."

    # Weighted ensemble of three metrics
    lbph_score   = _lbph_similarity(face_stored, face_live)
    colour_score = _colour_hist_similarity(face_stored, face_live)
    ssim_score   = _ssim_similarity(face_stored, face_live)

    # Weights: LBPH captures identity best, SSIM catches structure, colour is supplemental
    score = 0.55 * lbph_score + 0.25 * ssim_score + 0.20 * colour_score

    logger.debug(
        "Face scores — LBPH: %.3f  SSIM: %.3f  Colour: %.3f  →  Final: %.3f",
        lbph_score, ssim_score, colour_score, score,
    )

    if score >= MATCH_THRESHOLD:
        return True, score, f"Face matched ({score*100:.0f}% similarity)."
    else:
        pct = score * 100
        needed = MATCH_THRESHOLD * 100
        return (
            False, score,
            f"Face did not match ({pct:.0f}% similarity, need ≥{needed:.0f}%). "
            "Try better lighting or look directly at the camera."
        )


def has_face(image_bytes: bytes) -> bool:
    """Quick check — does this image contain a detectable face?"""
    img = _bytes_to_bgr(image_bytes)
    if img is None:
        return False
    return _detect_face(img) is not None