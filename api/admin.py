from django.contrib import admin
from .models import UserProfile, AttendanceLog, AttendanceSession

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display  = ['name', 'student_id', 'department', 'photo_count', 'is_present', 'last_seen', 'created_at']
    list_filter   = ['department', 'is_present']
    search_fields = ['name', 'student_id']

@admin.register(AttendanceLog)
class AttendanceLogAdmin(admin.ModelAdmin):
    list_display  = ['user', 'event_type', 'confidence', 'timestamp']
    list_filter   = ['event_type', 'timestamp']
    search_fields = ['user__name', 'user__student_id']

@admin.register(AttendanceSession)
class AttendanceSessionAdmin(admin.ModelAdmin):
    list_display  = ['user', 'date', 'entry_time', 'exit_time', 'duration_minutes']
    list_filter   = ['date']
    search_fields = ['user__name']
