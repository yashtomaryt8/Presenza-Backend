"""
ai_utils.py — Presenza AI Engine
Semantic / RAG attendance queries + Groq / Ollama support.

SEMANTIC QUERY FLOW:
  1. Parse natural language query
  2. Pull relevant attendance data from Django ORM
  3. Build a rich data context
  4. Send context + query to LLM
  5. Return structured answer

Supports queries like:
  - "When did Yash arrive today?"
  - "Who was absent last 5 days?"
  - "Who entered between 4pm and 6pm?"
  - "On which days was Sneha absent this week?"
  - "What is the attendance rate today?"
"""

import re
import requests
from datetime import datetime, date, timedelta
from django.utils import timezone

# ── Strict system prompts ─────────────────────────────────────────────────────
_ANALYTICS_SYSTEM = (
    "You are an attendance analytics assistant. "
    "Analyse the provided attendance data and give 3-5 SHORT, SPECIFIC bullet points. "
    "Each bullet must start with '• '. "
    "Do NOT add greetings, disclaimers, or text outside bullet points. "
    "Be direct and factual."
)

_SEMANTIC_SYSTEM = (
    "You are an intelligent attendance system assistant. "
    "You are given real attendance data from a database. "
    "Answer the user's question DIRECTLY and SPECIFICALLY using only the data provided. "
    "Be concise. If data shows specific times, list them. "
    "If asked about a person, mention them by name. "
    "If no data matches the query, say exactly what was found (or not found). "
    "Do NOT make up data. Do NOT add disclaimers."
)


# ── Groq ──────────────────────────────────────────────────────────────────────
def query_groq(prompt: str, api_key: str, model: str = "llama-3.1-8b-instant",
               system: str = None) -> str:
    if not api_key:
        return "Error: GROQ_API_KEY not set — add it to Railway environment variables."
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system or _ANALYTICS_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                "max_tokens": 600,
                "temperature": 0.2,
            },
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.Timeout:
        return "Error: Groq request timed out."
    except Exception as e:
        return f"Error: {e}"


# ── Ollama ────────────────────────────────────────────────────────────────────
def query_ollama(prompt: str, host: str = "http://localhost:11434",
                 model: str = "llama3.2:1b", system: str = None) -> str:
    """
    Best models for old CPUs (i7-3770):
      llama3.2:1b  (~700MB) — fastest, good quality
      qwen2.5:1.5b (~900MB) — slightly better for structured queries
      phi3.5:mini  (~2GB)   — needs 8GB+ RAM
    """
    full = f"{system or _ANALYTICS_SYSTEM}\n\nDATA:\n{prompt}\n\nANSWER:"
    try:
        r = requests.post(
            f"{host}/api/generate",
            json={
                "model":  model,
                "prompt": full,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 500},
            },
            timeout=120,
        )
        r.raise_for_status()
        text = r.json().get("response", "").strip()
        return text or "No response generated."
    except requests.exceptions.ConnectionError:
        return "Error: Ollama not running. Run: ollama serve"
    except requests.exceptions.Timeout:
        return "Error: Ollama timed out. Try a smaller model."
    except Exception as e:
        return f"Error: {e}"


# ── Semantic query engine ─────────────────────────────────────────────────────
def _extract_name(query: str) -> str | None:
    """Try to extract a person's name from the query."""
    # Skip common words, capitalize proper nouns
    skip = {'did', 'when', 'who', 'was', 'is', 'the', 'a', 'an', 'to', 'in', 'on',
            'at', 'of', 'for', 'with', 'by', 'from', 'today', 'last', 'this', 'week',
            'days', 'time', 'exit', 'enter', 'arrive', 'absence', 'absent', 'present',
            'attendance', 'show', 'me', 'list', 'give', 'data', 'students', 'student',
            'what', 'how', 'many', 'between', 'and', 'pm', 'am'}
    words = re.findall(r'\b[A-Z][a-z]+\b', query)  # Capitalized words
    for w in words:
        if w.lower() not in skip:
            return w
    # Also try all words capitalized in context
    words2 = re.findall(r'\b([a-z]{3,})\b', query.lower())
    for w in words2:
        if w not in skip and not w.isdigit():
            pass  # Can't distinguish names from other words without NER
    return None


def _extract_time_range(query: str):
    """Extract time range from query, e.g. '4pm to 6pm' → (16, 18)."""
    pattern = r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)\s*(?:to|-)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)'
    m = re.search(pattern, query, re.IGNORECASE)
    if m:
        h1, m1, p1, h2, m2, p2 = m.groups()
        h1 = int(h1); h2 = int(h2)
        if p1.lower() == 'pm' and h1 != 12: h1 += 12
        if p2.lower() == 'pm' and h2 != 12: h2 += 12
        if p1.lower() == 'am' and h1 == 12: h1 = 0
        if p2.lower() == 'am' and h2 == 12: h2 = 0
        return h1, h2
    return None, None


def _days_back(query: str) -> int:
    """Extract number of days from query, e.g. 'last 5 days' → 5."""
    m = re.search(r'last\s+(\d+)\s+days?', query, re.IGNORECASE)
    if m:
        return int(m.group(1))
    if 'week' in query.lower():
        return 7
    return 7  # default


def build_semantic_context(query: str) -> tuple[str, list]:
    """
    Parse the query, fetch relevant DB data, and build a rich context string.
    Returns (context_string, raw_records_list).
    """
    from api.models import UserProfile, AttendanceLog, AttendanceSession

    q_lower = query.lower()
    today   = timezone.now().date()
    records = []
    lines   = []

    # Extract entities
    name       = _extract_name(query)
    days_back  = _days_back(query)
    h_from, h_to = _extract_time_range(query)
    date_range = [today - timedelta(days=i) for i in range(days_back)]

    lines.append(f"Query: {query}")
    lines.append(f"Today: {today}")
    lines.append(f"Total registered students: {UserProfile.objects.count()}")
    lines.append("")

    # ── Person-specific query ─────────────────────────────────────────────────
    if name:
        users = UserProfile.objects.filter(name__icontains=name)
        if users.exists():
            user = users.first()
            lines.append(f"Student: {user.name} (ID: {user.student_id or 'N/A'}, Dept: {user.department or 'N/A'})")
            lines.append(f"Currently present: {'Yes' if user.is_present else 'No'}")
            lines.append(f"Last seen: {user.last_seen.strftime('%Y-%m-%d %H:%M') if user.last_seen else 'Never'}")
            lines.append("")

            # Logs for this person
            qs = AttendanceLog.objects.filter(user=user, timestamp__date__in=date_range).order_by('timestamp')
            if h_from is not None and h_to is not None:
                qs = qs.filter(timestamp__hour__gte=h_from, timestamp__hour__lt=h_to)

            if qs.exists():
                lines.append(f"Attendance logs for {user.name} (last {days_back} days):")
                for log in qs:
                    lines.append(f"  {log.timestamp.strftime('%Y-%m-%d %a')} {log.timestamp.strftime('%H:%M')} — {log.event_type.upper()} ({log.confidence*100:.0f}% conf)")
                    records.append({
                        'name': user.name, 'user_name': user.name,
                        'event_type': log.event_type,
                        'timestamp': log.timestamp.isoformat(),
                    })
            else:
                lines.append(f"No attendance records found for {user.name} in the last {days_back} days.")

            # Absence detection
            present_dates = set(
                AttendanceLog.objects.filter(user=user, event_type='entry')
                .values_list('timestamp__date', flat=True)
            )
            absent_dates = [d for d in date_range if d not in present_dates]
            if absent_dates:
                lines.append(f"\nAbsent on: {', '.join(d.strftime('%a %d %b') for d in sorted(absent_dates))}")
        else:
            lines.append(f"No student named '{name}' found in the database.")

    # ── Time-range query ──────────────────────────────────────────────────────
    elif h_from is not None and h_to is not None:
        qs = AttendanceLog.objects.filter(
            timestamp__date=today,
            event_type='entry',
            timestamp__hour__gte=h_from,
            timestamp__hour__lt=h_to,
        ).select_related('user').order_by('timestamp')

        lines.append(f"Students who entered between {h_from:02d}:00 and {h_to:02d}:00 today:")
        if qs.exists():
            for log in qs:
                lines.append(f"  {log.user.name} at {log.timestamp.strftime('%H:%M')} ({log.confidence*100:.0f}% conf)")
                records.append({'name': log.user.name, 'user_name': log.user.name, 'event_type': 'entry', 'timestamp': log.timestamp.isoformat()})
        else:
            lines.append("  No entries found in this time range.")

    # ── Absence / attendance rate query ───────────────────────────────────────
    elif any(w in q_lower for w in ['absent', 'absence', 'missing', 'not present']):
        users = UserProfile.objects.all()
        lines.append(f"Absence analysis (last {days_back} days):")
        for user in users:
            present_count = AttendanceLog.objects.filter(
                user=user, event_type='entry', timestamp__date__in=date_range
            ).values('timestamp__date').distinct().count()
            absent_count = days_back - present_count
            if absent_count > 0:
                lines.append(f"  {user.name}: absent {absent_count}/{days_back} days")
                records.append({'name': user.name, 'user_name': user.name, 'absent_days': absent_count, 'timestamp': ''})

    # ── Attendance rate ───────────────────────────────────────────────────────
    elif any(w in q_lower for w in ['rate', 'percentage', 'how many', 'count', 'total']):
        total = UserProfile.objects.count()
        present_today = AttendanceLog.objects.filter(
            timestamp__date=today, event_type='entry'
        ).values('user').distinct().count()
        late = AttendanceLog.objects.filter(
            timestamp__date=today, event_type='entry', timestamp__hour__gte=9
        ).values('user').distinct().count()
        rate = round(present_today / total * 100, 1) if total else 0
        lines.append(f"Today's attendance ({today}):")
        lines.append(f"  Present: {present_today} / {total} students ({rate}%)")
        lines.append(f"  Late arrivals (after 9 AM): {late}")

    # ── General / present now ─────────────────────────────────────────────────
    else:
        present = UserProfile.objects.filter(is_present=True)
        lines.append("Students currently present:")
        if present.exists():
            for u in present:
                lines.append(f"  {u.name} (last seen: {u.last_seen.strftime('%H:%M') if u.last_seen else 'N/A'})")
                records.append({'name': u.name, 'user_name': u.name, 'timestamp': u.last_seen.isoformat() if u.last_seen else ''})
        else:
            lines.append("  Nobody currently marked as present.")

        # Recent logs
        recent = AttendanceLog.objects.filter(timestamp__date=today).order_by('-timestamp')[:10]
        if recent:
            lines.append("\nMost recent events today:")
            for log in recent:
                lines.append(f"  {log.user.name}: {log.event_type.upper()} at {log.timestamp.strftime('%H:%M')}")

    return "\n".join(lines), records


def build_analytics_prompt(stats: dict) -> str:
    return (
        f"Total registered students: {stats.get('total_users', 0)}\n"
        f"Present today: {stats.get('present_today', 0)}\n"
        f"Attendance rate today: {stats.get('attendance_rate_today', 0)}%\n"
        f"Late arrivals (after 9 AM): {stats.get('late_today', 0)}\n"
        f"Weekly attendance total: {stats.get('week_total', 0)}\n"
        f"Average daily attendance this week: {stats.get('week_avg', 0)}\n"
        f"Most frequent attendee: {stats.get('top_attendee', 'N/A')}\n"
        f"Peak arrival hour: {stats.get('peak_hour', 'N/A')}\n\n"
        f"Provide 4 specific observations and 1 recommendation."
    )
