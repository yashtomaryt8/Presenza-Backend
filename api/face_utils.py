"""
face_utils.py — Presenza Face Pipeline v4

Key fixes vs v3:
  - Confidence display calibrated: cosine sim mapped to 0-100% properly
    (buffalo_l ArcFace range ~0.28–0.55 → display as 0–100%)
  - Centroid-based matching: all stored embeddings averaged → one stable centroid
    per user. More accurate than max-over-all, faster to compute.
  - Pre-computed embedding cache: centroids stored as numpy arrays,
    no JSON parsing on every scan frame.
  - Pose filtering: reject extreme side-profile faces (yaw > 35°)
  - Quality filtering: reject blurry face crops (quality < 0.15)
  - Anti-spoof (is_live) actually wired into process_frame()
  - HF Space keep-alive ping every 8 minutes (prevents cold starts)
  - Image sent at 480px max_dim (was 320px) for better recognition
"""

import os
import cv2
import numpy as np
import requests
import time
import threading
import logging

from django.utils import timezone
from api.models import UserProfile, AttendanceLog, AttendanceSession
from .anti_spoof import is_live

logger = logging.getLogger(__name__)

HF_SPACE_URL          = os.environ.get('HF_SPACE_URL', '').rstrip('/')
HF_TIMEOUT            = int(os.environ.get('HF_TIMEOUT', '10'))
ATTENDANCE_COOLDOWN_S = int(os.environ.get('ATTENDANCE_COOLDOWN_S', '10'))

# Recognition threshold — raw cosine similarity (NOT the display percentage)
# buffalo_l: same-person similarity typically 0.28–0.55 on CPU live frames
# Threshold at 0.30 → anything above is a match (calibrated display handles %)
RECOGNITION_THRESHOLD = float(os.environ.get('RECOGNITION_THRESHOLD', '0.30'))
DUPLICATE_THRESHOLD   = float(os.environ.get('DUPLICATE_THRESHOLD', '0.55'))

# Confidence display calibration
# Maps cosine similarity range [CONF_LO, CONF_HI] → [0%, 100%]
# buffalo_l ArcFace: verified-same-person range is approx 0.28–0.58
CONF_LO = float(os.environ.get('CONF_DISPLAY_LO', '0.28'))
CONF_HI = float(os.environ.get('CONF_DISPLAY_HI', '0.56'))

# Quality gates
MIN_FACE_QUALITY = float(os.environ.get('MIN_FACE_QUALITY', '0.15'))  # sharpness 0–1
MAX_YAW_DEGREES  = float(os.environ.get('MAX_YAW_DEGREES', '35.0'))   # reject extreme profiles
LIVENESS_CHECK   = os.environ.get('LIVENESS_CHECK', 'true').lower() == 'true'


def sim_to_display_pct(sim: float) -> float:
    """
    Calibrate raw cosine similarity to a user-friendly confidence percentage.

    buffalo_l ArcFace raw cosine between same person in live conditions: ~0.28–0.56
    This maps that range to 0–100%. Values above CONF_HI cap at 100%.
    """
    pct = (sim - CONF_LO) / (CONF_HI - CONF_LO) * 100.0
    return round(min(100.0, max(0.0, pct)), 1)


# ── User + embedding cache ────────────────────────────────────────
# Stores pre-computed L2-normalised centroids per user
# Refreshed every 45s or on invalidate()

_user_list:    list  = []          # list of UserProfile ORM objects
_emb_cache:    dict  = {}          # user.id → normalised centroid np.ndarray
_cache_ts:     float = 0.0
_CACHE_TTL_S          = 45


def _rebuild_cache():
    global _user_list, _emb_cache, _cache_ts
    users  = list(UserProfile.objects.all())
    embs   = {}
    for user in users:
        raw = user.get_embeddings()
        if not raw:
            continue
        vecs = [np.array(e, dtype=np.float32) for e in raw]
        centroid = np.mean(vecs, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 1e-9:
            centroid = centroid / norm
        embs[user.id] = centroid
    _user_list = users
    _emb_cache = embs
    _cache_ts  = time.time()
    logger.debug(f"Embedding cache rebuilt: {len(users)} users")


def _get_cache(force=False):
    if force or time.time() - _cache_ts > _CACHE_TTL_S:
        _rebuild_cache()
    return _user_list, _emb_cache


def invalidate_cache():
    global _cache_ts
    _cache_ts = 0.0


# ── HF Space keep-alive ───────────────────────────────────────────
def _keepalive_loop():
    """Ping HF Space every 8 minutes to prevent cold starts (15min timeout)."""
    time.sleep(30)  # initial delay — don't ping on startup
    while True:
        if HF_SPACE_URL:
            try:
                requests.get(f"{HF_SPACE_URL}/health", timeout=5)
                logger.debug("HF Space keep-alive ping sent")
            except Exception:
                pass
        time.sleep(480)  # 8 minutes


_keepalive_thread = threading.Thread(target=_keepalive_loop, daemon=True)
_keepalive_thread.start()


# ── Image helpers ─────────────────────────────────────────────────
def _img_to_bytes(img: np.ndarray, quality: int = 80, max_dim: int = 480) -> bytes:
    """Resize and JPEG-encode for HF upload. 480px is optimal — not too large, not too small."""
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


# ── HF Space callers ──────────────────────────────────────────────
def get_embedding(image_array: np.ndarray):
    """Extract embedding for registration. Returns (np.array | None, error_str)."""
    if not HF_SPACE_URL:
        return None, "HF_SPACE_URL not configured"
    try:
        r = requests.post(
            f"{HF_SPACE_URL}/extract",
            files={"image": ("face.jpg", _img_to_bytes(image_array, 90, 800), "image/jpeg")},
            timeout=HF_TIMEOUT,
        )
        if r.status_code != 200:
            msg = r.json().get('error', f'HTTP {r.status_code}') if r.content else f'HTTP {r.status_code}'
            return None, msg
        data = r.json()

        # Log quality warning if blurry
        quality = data.get('quality', 1.0)
        if quality < 0.2:
            logger.warning(f"Registration photo quality low ({quality:.2f}) — consider better lighting")

        return np.array(data["embedding"], dtype=np.float32), None
    except requests.exceptions.Timeout:
        return None, "HF Space timed out — try again"
    except Exception as e:
        return None, str(e)


def detect_faces_remote(image_array: np.ndarray) -> list:
    """
    Detect all faces via HF Space /detect.
    Returns list of {embedding, bbox, det_score, pose?, quality?}.
    Already filtered by det_score >= 0.82 on HF side.
    """
    if not HF_SPACE_URL:
        return []
    try:
        r = requests.post(
            f"{HF_SPACE_URL}/detect",
            files={"image": ("frame.jpg", _img_to_bytes(image_array, 80, 480), "image/jpeg")},
            timeout=HF_TIMEOUT,
        )
        if r.status_code != 200:
            return []
        return r.json().get("faces", [])
    except Exception:
        return []


# ── Face matching ─────────────────────────────────────────────────
def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised vectors."""
    # b is already normalised (centroid from cache)
    # normalise a just in case
    norm_a = np.linalg.norm(a)
    if norm_a < 1e-9:
        return 0.0
    return float(np.dot(a / norm_a, b))


def match_face(embedding, exclude_user_id: int = None):
    """
    Match embedding against all stored user centroids.
    Centroid matching is more stable than max-over-individual-embeddings:
    it averages out registration photo noise and gives better confidence scores.

    Returns (best_user | None, best_raw_sim).
    """
    users, emb_cache = _get_cache()
    best_user, best_sim = None, 0.0
    emb = np.array(embedding, dtype=np.float32)

    for user in users:
        if exclude_user_id and user.id == exclude_user_id:
            continue
        centroid = emb_cache.get(user.id)
        if centroid is None:
            continue
        sim = _cosine_sim(emb, centroid)
        if sim > best_sim:
            best_sim  = sim
            best_user = user

    if best_sim >= RECOGNITION_THRESHOLD:
        return best_user, best_sim
    return None, best_sim


def check_duplicate_face(embedding: np.ndarray, exclude_user_id: int = None):
    """
    Check if a face is already registered (for registration duplicate check).
    Uses higher DUPLICATE_THRESHOLD to avoid false rejects.
    Returns (existing_user | None, similarity_score).
    """
    users, emb_cache = _get_cache(force=True)
    best_user, best_sim = None, 0.0
    emb = np.array(embedding, dtype=np.float32)

    for user in users:
        if exclude_user_id and user.id == exclude_user_id:
            continue
        centroid = emb_cache.get(user.id)
        if centroid is None:
            continue
        sim = _cosine_sim(emb, centroid)
        if sim > best_sim:
            best_sim  = sim
            best_user = user

    if best_sim >= DUPLICATE_THRESHOLD:
        return best_user, best_sim
    return None, best_sim


# ── Attendance logic ──────────────────────────────────────────────
def mark_attendance(user: 'UserProfile', event_type: str, raw_sim: float):
    """
    Log attendance with per-event-type cooldown.

    Cooldown rules:
      entry → entry: blocked for ATTENDANCE_COOLDOWN_S
      exit  → exit:  blocked for ATTENDANCE_COOLDOWN_S
      entry ↔ exit:  always allowed (opposite event types)

    Returns (logged: bool, reason: str).
    """
    now   = timezone.now()
    today = now.date()

    last_same = (
        AttendanceLog.objects
        .filter(user=user, event_type=event_type)
        .order_by('-timestamp')
        .first()
    )
    if last_same:
        elapsed = (now - last_same.timestamp).total_seconds()
        if elapsed < ATTENDANCE_COOLDOWN_S:
            return False, f'cooldown_{int(ATTENDANCE_COOLDOWN_S - elapsed)}s'

    AttendanceLog.objects.create(user=user, event_type=event_type, confidence=round(raw_sim, 4))

    if event_type == 'entry':
        AttendanceSession.objects.create(user=user, entry_time=now, date=today)
        user.is_present = True
    else:
        open_sess = (
            AttendanceSession.objects
            .filter(user=user, exit_time__isnull=True, date=today)
            .order_by('-entry_time')
            .first()
        )
        if open_sess:
            open_sess.exit_time       = now
            open_sess.duration_minutes = round((now - open_sess.entry_time).total_seconds() / 60, 2)
            open_sess.save()
        user.is_present = False

    user.last_seen = now
    user.save(update_fields=['is_present', 'last_seen'])
    invalidate_cache()
    return True, 'marked'


# ── Main scan pipeline ────────────────────────────────────────────
def process_frame(image_array: np.ndarray, event_type: str = 'entry'):
    """
    Full pipeline:
      1. Detect faces via HF Space (already det_score filtered)
      2. Quality gate: pose + sharpness
      3. Anti-spoof check (MiniFASNet or LBP texture)
      4. Face recognition via centroid matching
      5. Attendance marking with cooldown

    Returns (None, detections_list).
    """
    faces      = detect_faces_remote(image_array)
    detections = []

    for face in faces:
        embedding = face.get('embedding')
        bbox      = face.get('bbox')
        quality   = face.get('quality', 1.0)
        pose      = face.get('pose', [0, 0, 0])  # [yaw, pitch, roll]

        if embedding is None or bbox is None:
            continue

        # ── Quality gate: pose ────────────────────────────────────
        yaw = abs(pose[0]) if pose else 0
        if yaw > MAX_YAW_DEGREES:
            # Too side-on — embedding quality will be poor
            detections.append({
                'name':       'Unknown',
                'student_id': '',
                'department': '',
                'confidence': 0,
                'event_type': None,
                'logged':     False,
                'reason':     'poor_angle',
                'bbox':       bbox,
            })
            continue

        # ── Quality gate: sharpness ───────────────────────────────
        if quality < MIN_FACE_QUALITY:
            detections.append({
                'name':       'Unknown',
                'student_id': '',
                'department': '',
                'confidence': 0,
                'event_type': None,
                'logged':     False,
                'reason':     'blurry',
                'bbox':       bbox,
            })
            continue

        # ── Anti-spoof check ──────────────────────────────────────
        if LIVENESS_CHECK:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            # Add 10% padding around face for better liveness context
            pad_x = int((x2 - x1) * 0.10)
            pad_y = int((y2 - y1) * 0.10)
            h_img, w_img = image_array.shape[:2]
            crop = image_array[
                max(0, y1 - pad_y): min(h_img, y2 + pad_y),
                max(0, x1 - pad_x): min(w_img, x2 + pad_x),
            ]
            if crop.size > 0:
                liveness = is_live(crop)
                if not liveness['live']:
                    detections.append({
                        'name':       'SPOOF',
                        'student_id': '',
                        'department': '',
                        'confidence': round(liveness['score'] * 100, 1),
                        'event_type': None,
                        'logged':     False,
                        'reason':     'spoof',
                        'bbox':       bbox,
                        'liveness':   liveness,
                    })
                    continue

        # ── Face recognition ──────────────────────────────────────
        user, raw_sim = match_face(embedding)
        display_conf  = sim_to_display_pct(raw_sim)

        if user:
            logged, reason = mark_attendance(user, event_type, raw_sim)
            detections.append({
                'name':       user.name,
                'student_id': user.student_id,
                'department': user.department,
                'confidence': display_conf,
                'event_type': event_type,
                'logged':     logged,
                'reason':     reason,
                'bbox':       bbox,
            })
        else:
            detections.append({
                'name':       'Unknown',
                'student_id': '',
                'department': '',
                'confidence': display_conf,
                'event_type': None,
                'logged':     False,
                'reason':     'no_match',
                'bbox':       bbox,
            })

    return None, detections
