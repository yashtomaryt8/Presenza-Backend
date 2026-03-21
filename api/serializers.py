from rest_framework import serializers
from .models import UserProfile, AttendanceLog, AttendanceSession


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model  = UserProfile
        fields = ['id', 'name', 'student_id', 'department',
                  'photo_count', 'is_present', 'last_seen', 'created_at']


class AttendanceLogSerializer(serializers.ModelSerializer):
    user_name       = serializers.CharField(source='user.name', read_only=True)
    user_student_id = serializers.CharField(source='user.student_id', read_only=True)
    department      = serializers.CharField(source='user.department', read_only=True)

    class Meta:
        model  = AttendanceLog
        fields = ['id', 'user', 'user_name', 'user_student_id', 'department',
                  'timestamp', 'event_type', 'confidence']


class AttendanceSessionSerializer(serializers.ModelSerializer):
    user_name = serializers.CharField(source='user.name', read_only=True)

    class Meta:
        model  = AttendanceSession
        fields = ['id', 'user', 'user_name', 'entry_time', 'exit_time',
                  'duration_minutes', 'date']
