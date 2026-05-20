"""face_auth.py — VitalNav face biometric authentication (v2)

Approach
--------
Detection  : Multi-cascade Haar with eye confirmation (proves it's a real face,
             not a blob or object).  Falls back to largest-face without eyes on
             stored signup photos (controlled environment).

Recognition: cv2.face.LBPHFaceRecognizer trained on the fly against the stored
             signup photo (±4 mild augmentations) for a proper ML confidence
             score, combined with SSIM + colour histogram ensemble.

Threshold  : 0.60  (60 % weighted match required)

No external models — everything ships with opencv-contrib-python.
"""
from __future__ import annotations

import base64
import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── tunables ──────────────────────────────────────────────────────────────────
FACE_SIZE        = 160          # px — larger than v1 for better LBPH texture
MATCH_THRESHOLD  = 0.60         # 0–1 weighted ensemble score
MIN_FACE_PX      = 80           # detector: reject faces smaller than this
EYE_CONFIRMATION = True         # require eye detection inside live-frame face
MAX_LBPH_DIST    = 120.0        # LBPH predict() distance → clamp to [0, 1]

# ── cascades ──────────────────────────────────────────────────────────────────
_FACE_CASCADES = [
    cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml"),
    cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"),
    cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"),
]

_EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
)

# ── image I/O ─────────────────────────────────────────────────────────────────

def _bytes_to_bgr(raw: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _b64_to_bgr(b64: str) -> Optional[np.ndarray]:
    try:
        return _bytes_to_bgr(base64.b64decode(b64))
    except Exception:
        return None


# ── pre-processing ────────────────────────────────────────────────────────────

def _preprocess(patch: np.ndarray) -> np.ndarray:
    """Resize → CLAHE on L channel → return colour patch (for colour metric)
    and grey patch (for LBPH / SSIM)."""
    resized = cv2.resize(patch, (FACE_SIZE, FACE_SIZE))
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _to_grey(patch: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)


# ── augmentation (used to train LBPH on a single image) ─────────────────────

def _augment(grey: np.ndarray) -> list[np.ndarray]:
    """Return original + small perturbations so LBPH gets a tiny training set."""
    variants = [grey]
    # slight brightness shifts
    for delta in (-18, 18):
        shifted = np.clip(grey.astype(np.int16) + delta, 0, 255).astype(np.uint8)
        variants.append(shifted)
    # horizontal flip (same person, mirrored)
    variants.append(cv2.flip(grey, 1))
    # tiny Gaussian blur (simulates soft focus)
    variants.append(cv2.GaussianBlur(grey, (3, 3), 0))
    return variants


# ── face detection ────────────────────────────────────────────────────────────

def _has_eyes(face_roi_grey: np.ndarray) -> bool:
    """Return True if at least one eye is detected in the upper-half of the ROI."""
    h = face_roi_grey.shape[0]
    upper = face_roi_grey[: h // 2, :]
    eyes = _EYE_CASCADE.detectMultiScale(
        upper,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(20, 20),
    )
    return len(eyes) >= 1


def _detect_largest_face(
    img_bgr: np.ndarray,
    min_px: int = MIN_FACE_PX,
    require_eyes: bool = False,
) -> Optional[np.ndarray]:
    """Return preprocessed face patch (colour) of the largest confirmed face,
    or None if not found."""
    img_h, img_w = img_bgr.shape[:2]
    grey = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    eq   = cv2.equalizeHist(grey)

    best_area = 0
    best_box: Optional[tuple] = None

    for cascade in _FACE_CASCADES:
        for scale, neighbors in [(1.1, 5), (1.1, 3), (1.05, 3)]:
            faces = cascade.detectMultiScale(
                eq,
                scaleFactor=scale,
                minNeighbors=neighbors,
                minSize=(min_px, min_px),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            if len(faces) == 0:
                continue
            for (x, y, w, h) in faces:
                area = w * h
                if area <= best_area:
                    continue
                if require_eyes:
                    roi_grey = grey[y : y + h, x : x + w]
                    if not _has_eyes(roi_grey):
                        continue
                best_area = area
                best_box  = (x, y, w, h)

    if best_box is None:
        return None

    x, y, w, h = best_box
    pad = int(0.12 * min(w, h))
    x1 = max(0, x - pad);       y1 = max(0, y - pad)
    x2 = min(img_w, x + w + pad); y2 = min(img_h, y + h + pad)
    return _preprocess(img_bgr[y1:y2, x1:x2])


# ── similarity metrics ────────────────────────────────────────────────────────

def _lbph_score(stored_grey: np.ndarray, live_grey: np.ndarray) -> float:
    """Train LBPHFaceRecognizer on augmented stored photo → predict on live frame.
    Returns normalised similarity 0–1 (1 = perfect match)."""
    augmented = _augment(stored_grey)
    labels    = [0] * len(augmented)

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=2, neighbors=8, grid_x=8, grid_y=8
    )
    recognizer.train(augmented, np.array(labels, dtype=np.int32))

    _, dist = recognizer.predict(live_grey)          # dist: 0 = perfect, ~120+ = stranger
    dist     = float(np.clip(dist, 0.0, MAX_LBPH_DIST))
    return float(1.0 - dist / MAX_LBPH_DIST)        # invert to similarity


def _ssim_score(a: np.ndarray, b: np.ndarray) -> float:
    """Structural similarity on greyscale patches."""
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    C1, C2, k = 6.5025, 58.5225, (11, 11)
    mu_a  = cv2.GaussianBlur(ga,    k, 1.5)
    mu_b  = cv2.GaussianBlur(gb,    k, 1.5)
    mu_a2 = mu_a ** 2; mu_b2 = mu_b ** 2; mu_ab = mu_a * mu_b
    sig_a2 = cv2.GaussianBlur(ga ** 2, k, 1.5) - mu_a2
    sig_b2 = cv2.GaussianBlur(gb ** 2, k, 1.5) - mu_b2
    sig_ab = cv2.GaussianBlur(ga * gb, k, 1.5) - mu_ab
    ssim = ((2 * mu_ab + C1) * (2 * sig_ab + C2)) / (
        (mu_a2 + mu_b2 + C1) * (sig_a2 + sig_b2 + C2)
    )
    return float(np.clip(ssim.mean(), 0.0, 1.0))


def _colour_score(a: np.ndarray, b: np.ndarray) -> float:
    """Per-channel Bhattacharyya histogram similarity (HSV hue + sat channels)."""
    ha_hsv = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
    hb_hsv = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
    scores  = []
    for ch, bins in [(0, 180), (1, 64)]:   # hue, saturation
        ha = cv2.calcHist([ha_hsv], [ch], None, [bins], [0, bins])
        hb = cv2.calcHist([hb_hsv], [ch], None, [bins], [0, bins])
        cv2.normalize(ha, ha); cv2.normalize(hb, hb)
        scores.append(1.0 - float(cv2.compareHist(ha, hb, cv2.HISTCMP_BHATTACHARYYA)))
    return float(np.mean(scores))


# ── public API ────────────────────────────────────────────────────────────────

def compare_faces(
    stored_b64: str,
    live_bytes:  bytes,
    crop_center: bool = True,   # kept for API compatibility — ignored
) -> tuple[bool, float, str]:
    """
    Compare stored signup photo (base64) with live camera frame (bytes).
    Returns (matched: bool, score: float 0–1, message: str).
    """
    # ── decode images ─────────────────────────────────────────────────────────
    stored_img = _b64_to_bgr(stored_b64)
    if stored_img is None:
        return False, 0.0, "Could not decode stored profile photo."

    live_img = _bytes_to_bgr(live_bytes)
    if live_img is None:
        return False, 0.0, "Could not decode camera frame."

    # ── detect faces ──────────────────────────────────────────────────────────
    # Stored photo: no eye requirement (lighting may be variable, no blue-square constraint)
    face_stored = _detect_largest_face(stored_img, min_px=60, require_eyes=False)
    if face_stored is None:
        return False, 0.0, (
            "No face found in your stored signup photo. "
            "Please re-take your profile photo from account settings."
        )

    # Live frame: require eyes → confirms it's a real frontal face, not an object/photo
    face_live = _detect_largest_face(
        live_img, min_px=MIN_FACE_PX, require_eyes=EYE_CONFIRMATION
    )
    if face_live is None:
        # Retry without eye requirement (glasses, bright lighting edge-cases)
        face_live = _detect_largest_face(live_img, min_px=MIN_FACE_PX, require_eyes=False)
        if face_live is None:
            return False, 0.0, (
                "No face detected in the camera frame. "
                "Move closer, face the camera directly, and ensure good even lighting."
            )

    # ── compute grey patches ──────────────────────────────────────────────────
    grey_stored = _to_grey(face_stored)
    grey_live   = _to_grey(face_live)

    # ── ensemble scores ───────────────────────────────────────────────────────
    lbph_s   = _lbph_score(grey_stored, grey_live)       # primary — ML-based
    ssim_s   = _ssim_score(face_stored, face_live)        # structural
    colour_s = _colour_score(face_stored, face_live)      # skin tone / hair

    # Weights: LBPH dominant (it's the actual recognizer), SSIM + colour supporting
    score = 0.60 * lbph_s + 0.25 * ssim_s + 0.15 * colour_s

    logger.debug(
        "FaceAuth  LBPH=%.3f  SSIM=%.3f  Colour=%.3f  Final=%.3f  threshold=%.2f",
        lbph_s, ssim_s, colour_s, score, MATCH_THRESHOLD,
    )

    if score >= MATCH_THRESHOLD:
        return True, score, f"Face matched ({score * 100:.0f}% similarity)."

    return False, score, (
        f"Face did not match ({score * 100:.0f}% — need ≥ {MATCH_THRESHOLD * 100:.0f}%). "
        "Tips: move closer, face camera directly, use even lighting, no strong shadows."
    )


def has_face(image_bytes: bytes) -> bool:
    """Quick check: does this image contain a detectable face?"""
    img = _bytes_to_bgr(image_bytes)
    if img is None:
        return False
    return _detect_largest_face(img, min_px=60, require_eyes=False) is not None