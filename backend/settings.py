"""
Django settings — Presenza Backend v4

Changes vs v3:
  - PostgreSQL support via dj_database_url (set DATABASE_URL env var on Railway)
  - Falls back to SQLite for local dev if DATABASE_URL not set
  - RECOGNITION_THRESHOLD / DUPLICATE_THRESHOLD / CONF_DISPLAY vars exposed
  - LIVENESS_CHECK env var to toggle anti-spoof on/off
  - Added CONN_MAX_AGE for persistent DB connections (faster repeat queries)
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'change-me-in-production-use-env-var')

DEBUG = os.environ.get('DEBUG', 'false').lower() == 'true'

ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'api',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'backend.urls'

TEMPLATES = [{
    'BACKEND': 'django.template.backends.django.DjangoTemplates',
    'DIRS': [],
    'APP_DIRS': True,
    'OPTIONS': {
        'context_processors': [
            'django.template.context_processors.debug',
            'django.template.context_processors.request',
            'django.contrib.auth.context_processors.auth',
            'django.contrib.messages.context_processors.messages',
        ],
    },
}]

WSGI_APPLICATION = 'backend.wsgi.application'

# ── Database ──────────────────────────────────────────────────────
# Set DATABASE_URL on Railway to use PostgreSQL and persist data across deploys.
# Format: postgres://user:password@host:port/dbname
# Railway provides this automatically when you add a PostgreSQL plugin.
# Without DATABASE_URL, falls back to SQLite (fine for local dev, NOT for Railway).

_DATABASE_URL = os.environ.get('DATABASE_URL', '')

if _DATABASE_URL:
    import dj_database_url
    DATABASES = {
        'default': dj_database_url.parse(
            _DATABASE_URL,
            conn_max_age=600,      # keep connections alive 10 min
            conn_health_checks=True,
        )
    }
else:
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME':   BASE_DIR / 'db.sqlite3',
        }
    }

# ── CORS ──────────────────────────────────────────────────────────
CORS_ALLOW_ALL_ORIGINS  = True
CORS_ALLOW_CREDENTIALS  = True
CORS_ALLOW_METHODS      = ['DELETE', 'GET', 'OPTIONS', 'PATCH', 'POST', 'PUT']
CORS_ALLOW_HEADERS      = ['*']

# ── Static files ──────────────────────────────────────────────────
STATIC_URL  = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

USE_TZ       = True
TIME_ZONE    = 'UTC'
LANGUAGE_CODE = 'en-us'

# ── Presenza config (all overridable via env vars) ────────────────
HF_SPACE_URL  = os.environ.get('HF_SPACE_URL',  '').rstrip('/')
GROQ_API_KEY  = os.environ.get('GROQ_API_KEY',  '')
GROQ_MODEL    = os.environ.get('GROQ_MODEL',    'llama-3.1-8b-instant')

# Face recognition thresholds (raw cosine similarity)
RECOGNITION_THRESHOLD = float(os.environ.get('RECOGNITION_THRESHOLD', '0.30'))
DUPLICATE_THRESHOLD   = float(os.environ.get('DUPLICATE_THRESHOLD',   '0.55'))

# Confidence display calibration (maps raw sim → display %)
CONF_DISPLAY_LO = float(os.environ.get('CONF_DISPLAY_LO', '0.28'))
CONF_DISPLAY_HI = float(os.environ.get('CONF_DISPLAY_HI', '0.56'))

# Cooldown between same-type events (seconds)
ATTENDANCE_COOLDOWN_S = int(os.environ.get('ATTENDANCE_COOLDOWN_S', '10'))

# HF Space request timeout
HF_TIMEOUT = int(os.environ.get('HF_TIMEOUT', '10'))

# Anti-spoof / quality gates
LIVENESS_CHECK   = os.environ.get('LIVENESS_CHECK',   'true')
MIN_FACE_QUALITY = float(os.environ.get('MIN_FACE_QUALITY', '0.15'))
MAX_YAW_DEGREES  = float(os.environ.get('MAX_YAW_DEGREES',  '35.0'))
