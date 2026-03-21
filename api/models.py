from django.db import models
import json


class UserProfile(models.Model):
    name = models.CharField(max_length=100)
    student_id = models.CharField(max_length=50, blank=True, default='')
    department = models.CharField(max_length=100, blank=True, default='')
    # Stores a JSON list of embedding arrays for multi-photo accuracy
    embeddings_json = models.TextField(default='[]')
    photo_count = models.IntegerField(default=0)
    is_present = models.BooleanField(default=False)
    last_seen = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def get_embeddings(self):
        try:
            return json.loads(self.embeddings_json)
        except Exception:
            return []

    def add_embedding(self, embedding_array):
        embs = self.get_embeddings()
        embs.append(embedding_array.tolist())
        self.embeddings_json = json.dumps(embs)
        self.photo_count = len(embs)

    def __str__(self):
        return f"{self.name} ({self.student_id or 'No ID'})"


class AttendanceLog(models.Model):
    EVENT_CHOICES = [('entry', 'Entry'), ('exit', 'Exit')]
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='logs')
    timestamp = models.DateTimeField(auto_now_add=True)
    event_type = models.CharField(max_length=10, choices=EVENT_CHOICES, default='entry')
    confidence = models.FloatField(default=0.0)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.user.name} [{self.event_type}] @ {self.timestamp:%Y-%m-%d %H:%M}"


class AttendanceSession(models.Model):
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='sessions')
    entry_time = models.DateTimeField()
    exit_time = models.DateTimeField(null=True, blank=True)
    duration_minutes = models.FloatField(null=True, blank=True)
    date = models.DateField()

    class Meta:
        ordering = ['-entry_time']

    def __str__(self):
        return f"{self.user.name} session on {self.date}"
