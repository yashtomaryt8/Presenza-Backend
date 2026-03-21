"""
face_utils.py — Presenza Face Pipeline (v3)

Key fixes:
  - Duplicate face detection at registration (prevents same face registered twice)
  - Per-event-type cooldown: entry→entry blocked 10s, but entry→exit allowed immediately
  - Exit logic: finds open session, closes it; is_present toggled correctly
  - User cache invalidated on write
  - Faster HF timeout (8s) to fail fast and retry
"""

import os
import cv2
import numpy as np
import requests
import time
from django.utils import timezone
from api.models import UserProfile, AttendanceLog, AttendanceSession

HF_SPACE_URL          = os.environ.get('HF_SPACE_URL', '').rstrip('/')
RECOGNITION_THRESHOLD = float(os.environ.get('RECOGNITION_THRESHOLD', '0.42'))
DUPLICATE_THRESHOLD   = float(os.environ.get('DUPLICATE_THRESHOLD', '0.68'))  # stricter for registration
ATTENDANCE_COOLDOWN_S = int(os.environ.get('ATTENDANCE_COOLDOWN_S', '10'))   # 10s gap per event type
HF_TIMEOUT            = int(os.environ.get('HF_TIMEOUT', '8'))

# User cache — refreshed every 45s
_user_cache    = []
_cache_updated = 0.0

def _get_users(force=False):
    global _user_cache, _cache_updated
    if force or time.time() - _cache_updated > 45:
        _user_cache    = list(UserProfile.objects.all())
        _cache_updated = time.time()
    return _user_cache

def invalidate_cache():
    global _cache_updated
    _cache_updated = 0.0


def _img_to_bytes(img: np.ndarray, quality: int = 70, max_dim: int = 480) -> bytes:
    """Resize and compress — smaller = faster HF upload."""
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


# ── HF Space callers ──────────────────────────────────────────────
def get_embedding(image_array: np.ndarray):
    """Extract embedding from largest face. Returns (np.array | None, error_str)."""
    if not HF_SPACE_URL:
        return None, "HF_SPACE_URL not configured"
    try:
        r = requests.post(
            f"{HF_SPACE_URL}/extract",
            files={"image": ("face.jpg", _img_to_bytes(image_array), "image/jpeg")},
            timeout=HF_TIMEOUT,
        )
        if r.status_code != 200:
            msg = r.json().get('error', f'HTTP {r.status_code}') if r.content else f'HTTP {r.status_code}'
            return None, msg
        return np.array(r.json()["embedding"], dtype=np.float32), None
    except requests.exceptions.Timeout:
        return None, "HF Space timed out — try again"
    except Exception as e:
        return None, str(e)


def detect_faces_remote(image_array: np.ndarray) -> list:
    """Detect all faces. Returns list of {embedding, bbox, det_score}."""
    if not HF_SPACE_URL:
        return []
    try:
        r = requests.post(
            f"{HF_SPACE_URL}/detect",
            files={"image": ("frame.jpg", _img_to_bytes(image_array, 65, 320), "image/jpeg")},
            timeout=HF_TIMEOUT,
        )
        if r.status_code != 200:
            return []
        return r.json().get("faces", [])
    except Exception:
        return []


# ── Face matching ─────────────────────────────────────────────────
def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / (denom + 1e-9))


def match_face(embedding, exclude_user_id: int = None):
    """
    Match embedding against all stored users.
    Returns (best_user | None, best_score).
    """
    best_user, best_score = None, 0.0
    emb = np.array(embedding, dtype=np.float32)
    for user in _get_users():
        if exclude_user_id and user.id == exclude_user_id:
            continue
        for stored in user.get_embeddings():
            sim = _cosine_sim(emb, np.array(stored, dtype=np.float32))
            if sim > best_score:
                best_score, best_user = sim, user
    return (best_user, best_score) if best_score >= RECOGNITION_THRESHOLD else (None, best_score)


def check_duplicate_face(embedding: np.ndarray, exclude_user_id: int = None):
    """
    Check if a face is already registered.
    Returns (existing_user | None, similarity_score).
    Uses a higher threshold (DUPLICATE_THRESHOLD) to avoid false positives.
    """
    best_user, best_score = None, 0.0
    emb = np.array(embedding, dtype=np.float32)
    for user in _get_users(force=True):
        if exclude_user_id and user.id == exclude_user_id:
            continue
        for stored in user.get_embeddings():
            sim = _cosine_sim(emb, np.array(stored, dtype=np.float32))
            if sim > best_score:
                best_score, best_user = sim, user
    if best_score >= DUPLICATE_THRESHOLD:
        return best_user, best_score
    return None, best_score


# ── Attendance logic ──────────────────────────────────────────────
def mark_attendance(user: 'UserProfile', event_type: str, confidence: float):
    """
    Log an attendance event with per-event-type cooldown.

    Rules:
      - entry → entry: blocked for ATTENDANCE_COOLDOWN_S seconds
      - exit  → exit:  blocked for ATTENDANCE_COOLDOWN_S seconds
      - entry → exit:  ALWAYS allowed (different event type)
      - exit  → entry: ALWAYS allowed (different event type)

    Returns (logged: bool, reason: str).
    """
    now   = timezone.now()
    today = now.date()

    # Check cooldown for same event type only
    last_same_type = (
        AttendanceLog.objects
        .filter(user=user, event_type=event_type)
        .order_by('-timestamp')
        .first()
    )
    if last_same_type:
        elapsed = (now - last_same_type.timestamp).total_seconds()
        if elapsed < ATTENDANCE_COOLDOWN_S:
            remaining = int(ATTENDANCE_COOLDOWN_S - elapsed)
            return False, f'cooldown_{remaining}s'

    # Create log
    AttendanceLog.objects.create(
        user=user,
        event_type=event_type,
        confidence=round(confidence, 4)
    )

    if event_type == 'entry':
        # Open new session
        AttendanceSession.objects.create(user=user, entry_time=now, date=today)
        user.is_present = True

    else:  # exit
        # Close the most recent open session for today
        open_sess = (
            AttendanceSession.objects
            .filter(user=user, exit_time__isnull=True, date=today)
            .order_by('-entry_time')
            .first()
        )
        if open_sess:
            open_sess.exit_time = now
            open_sess.duration_minutes = round(
                (now - open_sess.entry_time).total_seconds() / 60, 2
            )
            open_sess.save()
        user.is_present = False

    user.last_seen = now
    user.save(update_fields=['is_present', 'last_seen'])
    invalidate_cache()

    return True, 'marked'


# ── Main pipeline ─────────────────────────────────────────────────
def process_frame(image_array: np.ndarray, event_type: str = 'entry'):
    faces      = detect_faces_remote(image_array)
    detections = []

    for face in faces:
        embedding = face['embedding']
        bbox      = face['bbox']

        user, sim = match_face(embedding)
        conf_pct  = round(sim * 100, 1)

        if user:
            logged, reason = mark_attendance(user, event_type, sim)
            detections.append({
                'name':       user.name,
                'student_id': user.student_id,
                'department': user.department,
                'confidence': conf_pct,
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
                'confidence': conf_pct,
                'event_type': None,
                'logged':     False,
                'reason':     'no_match',
                'bbox':       bbox,
            })

    return None, detections
