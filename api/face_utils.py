"""
face_utils.py — Presenza Face Pipeline
All heavy CV runs on HF Space (InsightFace buffalo_sc).
Django does: matching, attendance logging, duplicate prevention.

Speed optimisations:
  - JPEG quality 75 (smaller upload)
  - 320x240 frame size (faster HF inference)
  - In-memory user cache (avoids DB hit every frame)
  - 30s cooldown prevents duplicate logs
"""

import os
import cv2
import numpy as np
import requests
from django.utils import timezone
from api.models import UserProfile, AttendanceLog, AttendanceSession

HF_SPACE_URL          = os.environ.get('HF_SPACE_URL', '').rstrip('/')
RECOGNITION_THRESHOLD = float(os.environ.get('RECOGNITION_THRESHOLD', '0.42'))
ATTENDANCE_COOLDOWN_S = int(os.environ.get('ATTENDANCE_COOLDOWN_S', '30'))
HF_TIMEOUT            = int(os.environ.get('HF_TIMEOUT', '12'))

# Simple in-memory user cache — refreshed every 60s
_user_cache    = []
_cache_updated = 0

def _get_users():
    import time
    global _user_cache, _cache_updated
    if time.time() - _cache_updated > 60:
        _user_cache    = list(UserProfile.objects.all())
        _cache_updated = time.time()
    return _user_cache


def _img_to_bytes(img: np.ndarray, quality: int = 75) -> bytes:
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def get_embedding(image_array: np.ndarray):
    """Call HF Space /extract. Used during registration."""
    if not HF_SPACE_URL:
        return None
    try:
        r = requests.post(
            f"{HF_SPACE_URL}/extract",
            files={"image": ("face.jpg", _img_to_bytes(image_array), "image/jpeg")},
            timeout=HF_TIMEOUT,
        )
        if r.status_code != 200:
            return None
        return np.array(r.json()["embedding"], dtype=np.float32)
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"HF extract: {e}")
        return None


def detect_faces_remote(image_array: np.ndarray) -> list:
    """Call HF Space /detect. Used during scanning."""
    if not HF_SPACE_URL:
        return []
    try:
        r = requests.post(
            f"{HF_SPACE_URL}/detect",
            files={"image": ("frame.jpg", _img_to_bytes(image_array, 75), "image/jpeg")},
            timeout=HF_TIMEOUT,
        )
        if r.status_code != 200:
            return []
        return r.json().get("faces", [])
    except Exception:
        return []


def _cosine_sim(a, b) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / (denom + 1e-9))


def match_face(embedding):
    best_user, best_score = None, 0.0
    emb = np.array(embedding, dtype=np.float32)
    for user in _get_users():
        for stored in user.get_embeddings():
            sim = _cosine_sim(emb, np.array(stored, dtype=np.float32))
            if sim > best_score:
                best_score, best_user = sim, user
    return (best_user, best_score) if best_score >= RECOGNITION_THRESHOLD else (None, best_score)


def mark_attendance(user, event_type: str, confidence: float):
    now  = timezone.now()
    last = AttendanceLog.objects.filter(user=user).order_by('-timestamp').first()
    if last and (now - last.timestamp).total_seconds() < ATTENDANCE_COOLDOWN_S:
        return False, 'cooldown'

    AttendanceLog.objects.create(user=user, event_type=event_type, confidence=round(confidence, 4))
    today = now.date()

    if event_type == 'entry':
        AttendanceSession.objects.create(user=user, entry_time=now, date=today)
        user.is_present = True
    else:
        sess = AttendanceSession.objects.filter(user=user, exit_time=None, date=today).order_by('-entry_time').first()
        if sess:
            sess.exit_time = now
            sess.duration_minutes = round((now - sess.entry_time).total_seconds() / 60, 2)
            sess.save()
        user.is_present = False

    user.last_seen = now
    user.save(update_fields=['is_present', 'last_seen'])

    # Invalidate user cache
    global _cache_updated
    _cache_updated = 0

    return True, 'marked'


def process_frame(image_array: np.ndarray, event_type: str = 'entry'):
    faces      = detect_faces_remote(image_array)
    detections = []

    for face in faces:
        embedding  = face['embedding']
        bbox       = face['bbox']
        det_score  = face.get('det_score', 1.0)
        liveness   = face.get('liveness', None)

        user, sim = match_face(embedding)
        conf_pct  = round(sim * 100, 1)

        if user:
            logged, reason = mark_attendance(user, event_type, sim)
            detections.append({
                'name':            user.name,
                'student_id':      user.student_id,
                'department':      user.department,
                'confidence':      conf_pct,
                'event_type':      event_type,
                'logged':          logged,
                'reason':          reason,
                'liveness_score':  liveness,
                'bbox':            bbox,
            })
        else:
            detections.append({
                'name':           'Unknown',
                'student_id':     '',
                'department':     '',
                'confidence':     conf_pct,
                'event_type':     None,
                'logged':         False,
                'reason':         'no_match',
                'liveness_score': liveness,
                'bbox':           bbox,
            })

    return None, detections
