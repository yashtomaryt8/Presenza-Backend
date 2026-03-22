from django.urls import path
from .views import (
    RegisterUser, AddPhotos, ScanFrame,
    UserListView, UserDetailView,
    UserEmbeddingsView, LogAttendanceView,
    AttendanceLogView, AttendanceSessionView,
    AnalyticsView, ExportCSV,
    AIInsightView, SemanticQueryView,
    ResetPresence, HealthCheck, ExtractMultiFaces,
)

urlpatterns = [
    path('health/',                   HealthCheck.as_view()),
    path('register/',                 RegisterUser.as_view()),

    # Users
    path('users/',                    UserListView.as_view()),
    path('users/embeddings/',         UserEmbeddingsView.as_view()),   # NEW — client-side matching
    path('users/<int:pk>/delete/',    UserDetailView.as_view()),
    path('users/<int:pk>/photos/',    AddPhotos.as_view()),

    # Attendance
    path('scan/',                     ScanFrame.as_view()),            # kept for compat
    path('log/',                      LogAttendanceView.as_view()),    # NEW — fire-and-forget log
    path('logs/',                     AttendanceLogView.as_view()),
    path('sessions/',                 AttendanceSessionView.as_view()),

    # Analytics + AI
    path('analytics/',                AnalyticsView.as_view()),
    path('export/',                   ExportCSV.as_view()),
    path('ai-insight/',               AIInsightView.as_view()),
    path('semantic-query/',           SemanticQueryView.as_view()),

    # Admin
    path('reset-presence/',           ResetPresence.as_view()),
    path('extract-multi/',            ExtractMultiFaces.as_view()),
]
