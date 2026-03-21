"""
anti_spoof.py
─────────────────────────────────────────────────────────────────────────────
Liveness Detection — prevents photo / screen / mask attacks.

TWO-LAYER APPROACH (both run on CPU, no GPU needed):
  Layer 1 – MiniFASNetV2   (Silent-Face-Anti-Spoofing, ~2 MB model)
             Deep neural net trained specifically to distinguish real faces
             from printed photos, phone screens, and replay attacks.
  Layer 2 – LBP Texture Score  (fallback, zero-download)
             Local Binary Pattern analysis. Real skin has rich micro-texture;
             printed photos and screens are flat / overly smooth.

RESULT:
  is_live(face_crop) → (live: bool, score: float, method: str)

SETUP (one-time):
  Download the two MiniFASNet weight files (~4 MB total) and place them in:
  backend/api/anti_spoof_models/

  Download script:
    python backend/api/anti_spoof.py --download

  Or manually from:
    https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/tree/master/resources/anti_spoof_models
    Files needed:
      2.7_80x80_MiniFASNetV2.pth
      4_0_0_80x80_MiniFASNetV1SE.pth
─────────────────────────────────────────────────────────────────────────────
"""

import os
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR   = os.path.join(_HERE, 'anti_spoof_models')
_MODEL_FILES = [
    '2.7_80x80_MiniFASNetV2.pth',
    '4_0_0_80x80_MiniFASNetV1SE.pth',
]

# ── Thresholds ────────────────────────────────────────────────────────────────
SPOOF_THRESHOLD  = 0.55   # below → spoof (stricter = safer, more false rejects)
TEXTURE_THRESHOLD = 0.35  # LBP fallback threshold

# ── Try loading PyTorch + MiniFASNet ──────────────────────────────────────────
_TORCH_OK = False
_models   = []

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_OK = True
except ImportError:
    logger.warning("PyTorch not installed — using LBP texture fallback only.")


# ── MiniFASNet architecture (self-contained, no external repo needed) ─────────
if _TORCH_OK:
    class _Conv_block(nn.Module):
        def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
            super().__init__()
            self.linear = linear
            if dw:
                self.conv = nn.Sequential(
                    nn.Conv2d(inp, inp, k, s, p, groups=inp, bias=False),
                    nn.BatchNorm2d(inp),
                    nn.PReLU(inp) if not linear else nn.Identity(),
                    nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    nn.Conv2d(inp, oup, k, s, p, bias=False),
                    nn.BatchNorm2d(oup),
                )
            self.act = nn.PReLU(oup) if not linear else nn.Identity()

        def forward(self, x):
            return self.act(self.conv(x))

    class _Residual(nn.Module):
        def __init__(self, c, k=3, s=1, p=1):
            super().__init__()
            self.residual = nn.Sequential(
                _Conv_block(c, c, 1, 1, 0),
                _Conv_block(c, c, k, s, p, dw=True, linear=True),
            )
            self.shortcut = _Conv_block(c, c, 1, 1, 0, linear=True)

        def forward(self, x):
            return self.residual(x) + self.shortcut(x)

    class MiniFASNetV2(nn.Module):
        """~2 MB lightweight anti-spoofing net."""
        def __init__(self, embedding_size=128, conv6_kernel=(5, 5), num_classes=2):
            super().__init__()
            self.conv1   = _Conv_block(3, 64,  3, 2, 1)
            self.conv2_1 = _Conv_block(64, 64, 3, 1, 1, dw=True)
            self.conv2_2 = _Conv_block(64, 128, 3, 2, 1, dw=True)
            self.conv3_1 = _Conv_block(128, 128, 3, 1, 1, dw=True)
            self.conv3_2 = _Conv_block(128, 256, 3, 2, 1, dw=True)
            self.conv4_1 = _Conv_block(256, 256, 3, 1, 1, dw=True)
            self.conv4_2 = _Conv_block(256, 512, 3, 2, 1, dw=True)
            self.conv5_1 = _Conv_block(512, 512, 3, 1, 1, dw=True)
            self.conv5_2 = _Conv_block(512, 512, 3, 1, 1, dw=True)
            self.conv5_3 = _Conv_block(512, 512, 3, 1, 1, dw=True)
            self.conv5_4 = _Conv_block(512, 512, 3, 1, 1, dw=True)
            self.conv6   = _Conv_block(512, 512, conv6_kernel, 1, 0, dw=True)
            self.conv7   = _Conv_block(512, embedding_size, 1, 1, 0)
            self.bn      = nn.BatchNorm2d(embedding_size)
            self.drop    = nn.Dropout()
            self.fc      = nn.Linear(embedding_size, num_classes)

        def forward(self, x):
            x = self.conv1(x);   x = self.conv2_1(x); x = self.conv2_2(x)
            x = self.conv3_1(x); x = self.conv3_2(x)
            x = self.conv4_1(x); x = self.conv4_2(x)
            x = self.conv5_1(x); x = self.conv5_2(x)
            x = self.conv5_3(x); x = self.conv5_4(x)
            x = self.conv6(x);   x = self.conv7(x)
            x = self.drop(self.bn(x))
            x = x.view(x.size(0), -1)
            return self.fc(x)

    def _load_models():
        """Load all available .pth models from _MODEL_DIR."""
        loaded = []
        if not os.path.isdir(_MODEL_DIR):
            return loaded
        for fname in _MODEL_FILES:
            path = os.path.join(_MODEL_DIR, fname)
            if not os.path.exists(path):
                continue
            try:
                # Parse kernel size from filename, e.g. "80x80" → (5,5) or (3,3)
                kernel = (5, 5) if '80x80' in fname else (3, 3)
                model  = MiniFASNetV2(conv6_kernel=kernel)
                state  = torch.load(path, map_location='cpu')
                # Handle wrapped state dicts
                if 'state_dict' in state:
                    state = state['state_dict']
                model.load_state_dict(state, strict=False)
                model.eval()
                loaded.append(model)
                logger.info(f"Loaded anti-spoof model: {fname}")
            except Exception as e:
                logger.warning(f"Could not load {fname}: {e}")
        return loaded

    _models = _load_models()


# ── LBP texture analysis (always available, zero dependencies) ────────────────
def _lbp_score(gray: np.ndarray) -> float:
    """
    Compute normalised LBP variance.
    Real skin: high variance (rich texture).
    Printed photo / screen: low variance (smooth, uniform).
    Returns 0.0 (definitely fake) → 1.0 (likely real).
    """
    h, w = gray.shape
    if h < 16 or w < 16:
        return 0.5   # too small to judge

    # Resize to fixed window
    patch = cv2.resize(gray, (64, 64)).astype(np.float32)

    # LBP-like: compare each pixel to 8 neighbours, build histogram
    neighbours = [
        patch[:-2, :-2], patch[:-2, 1:-1], patch[:-2, 2:],
        patch[1:-1, :-2],                   patch[1:-1, 2:],
        patch[2:,  :-2], patch[2:,  1:-1], patch[2:,  2:],
    ]
    center = patch[1:-1, 1:-1]
    lbp    = np.zeros_like(center, dtype=np.uint8)
    for i, nb in enumerate(neighbours):
        lbp += ((nb >= center).astype(np.uint8) << i)

    # Variance of LBP pattern is a strong liveness cue
    var = float(np.var(lbp))

    # Empirically calibrated: real faces ~700–2000, photos ~50–300
    score = min(1.0, max(0.0, (var - 30) / 1200))
    return score


# ── Frequency domain texture (catches screen moiré patterns) ─────────────────
def _freq_score(gray: np.ndarray) -> float:
    """
    Real faces have broad frequency content.
    Printed / screen fakes show dominant low-freq + moiré artifacts.
    """
    patch = cv2.resize(gray, (64, 64)).astype(np.float32)
    fft   = np.fft.fft2(patch)
    fft_s = np.fft.fftshift(fft)
    mag   = np.log1p(np.abs(fft_s))

    # High-frequency energy ratio
    cx, cy = 32, 32
    r_lo, r_hi = 8, 28
    y, x = np.ogrid[:64, :64]
    mask_lo = ((x - cx)**2 + (y - cy)**2) <= r_lo**2
    mask_hi = ((x - cx)**2 + (y - cy)**2) <= r_hi**2

    lo = float(mag[mask_lo].mean())
    hi = float(mag[~mask_lo & mask_hi].mean())
    total = lo + hi + 1e-9

    # Real faces: more balanced; fake: dominated by low freqs
    hf_ratio = hi / total
    score = min(1.0, max(0.0, (hf_ratio - 0.2) / 0.4))
    return score


# ── Colour saturation check (screens are over-saturated) ─────────────────────
def _color_score(bgr: np.ndarray) -> float:
    """Screens and glossy prints often have extreme colour saturation."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat = hsv[:, :, 1].mean() / 255.0
    # Real skin: moderate saturation (~0.2–0.5)
    # Screens: often > 0.6
    score = 1.0 - max(0.0, (sat - 0.45) / 0.3)
    return min(1.0, max(0.0, score))


# ── Public API ────────────────────────────────────────────────────────────────
def is_live(face_bgr: np.ndarray) -> dict:
    """
    Main entry point.

    Args:
        face_bgr: Face crop (BGR numpy array, any size ≥ 16×16).

    Returns:
        {
          'live':   bool,    True if real person
          'score':  float,   0.0 = definite spoof, 1.0 = definite live
          'method': str,     'deep' | 'texture'
          'detail': dict,    sub-scores for debugging
        }
    """
    if face_bgr is None or face_bgr.size == 0:
        return {'live': False, 'score': 0.0, 'method': 'error', 'detail': {}}

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

    # ── Layer 1: Deep model ──────────────────────────────────────────────────
    if _TORCH_OK and _models:
        try:
            resized = cv2.resize(face_bgr, (80, 80))
            tensor  = _preprocess(resized)
            scores  = []
            with torch.no_grad():
                for model in _models:
                    out   = model(tensor)
                    prob  = F.softmax(out, dim=1)
                    # class 1 = real, class 0 = spoof
                    scores.append(float(prob[0, 1]))
            deep_score = float(np.mean(scores))

            # Combine with texture for robustness
            lbp  = _lbp_score(gray)
            freq = _freq_score(gray)
            col  = _color_score(face_bgr)
            combined = deep_score * 0.6 + lbp * 0.2 + freq * 0.1 + col * 0.1

            return {
                'live':   combined >= SPOOF_THRESHOLD,
                'score':  round(combined, 3),
                'method': 'deep',
                'detail': {
                    'deep':    round(deep_score, 3),
                    'texture': round(lbp, 3),
                    'freq':    round(freq, 3),
                    'color':   round(col, 3),
                },
            }
        except Exception as e:
            logger.warning(f"Deep anti-spoof failed, falling back: {e}")

    # ── Layer 2: Texture-only fallback ───────────────────────────────────────
    lbp  = _lbp_score(gray)
    freq = _freq_score(gray)
    col  = _color_score(face_bgr)
    score = lbp * 0.5 + freq * 0.3 + col * 0.2

    return {
        'live':   score >= TEXTURE_THRESHOLD,
        'score':  round(score, 3),
        'method': 'texture',
        'detail': {'texture': round(lbp, 3), 'freq': round(freq, 3), 'color': round(col, 3)},
    }


def _preprocess(bgr: np.ndarray):
    """Convert 80×80 BGR crop to normalised float tensor."""
    rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean  = np.array([0.485, 0.456, 0.406])
    std   = np.array([0.229, 0.224, 0.225])
    rgb   = (rgb - mean) / std
    t     = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float()
    return t


# ── Download helper ───────────────────────────────────────────────────────────
def download_models():
    """Run with: python anti_spoof.py --download"""
    import urllib.request
    BASE = (
        "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing"
        "/raw/master/resources/anti_spoof_models/"
    )
    os.makedirs(_MODEL_DIR, exist_ok=True)
    for fname in _MODEL_FILES:
        dest = os.path.join(_MODEL_DIR, fname)
        if os.path.exists(dest):
            print(f"  Already exists: {fname}")
            continue
        url = BASE + fname
        print(f"  Downloading {fname} …", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, dest)
            print("done")
        except Exception as e:
            print(f"FAILED: {e}")
            print(f"  Manual URL: {url}")
    print("\nAll done. Re-start the Django server.")


if __name__ == "__main__":
    import sys
    if "--download" in sys.argv:
        download_models()
    else:
        print("Usage:  python anti_spoof.py --download")
        print(f"Models go in: {_MODEL_DIR}")
