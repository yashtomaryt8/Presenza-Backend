from django.urls import path
from .views import (
    RegisterUser, AddPhotos, ScanFrame,
    UserListView, UserDetailView,
    AttendanceLogView, AttendanceSessionView,
    AnalyticsView, ExportCSV,
    AIInsightView, SemanticQueryView,
    ResetPresence, HealthCheck, ExtractMultiFaces
)

urlpatterns = [
    path('health/',                   HealthCheck.as_view()),
    path('register/',                 RegisterUser.as_view()),
    path('users/',                    UserListView.as_view()),
    path('users/<int:pk>/delete/',    UserDetailView.as_view()),
    path('users/<int:pk>/photos/',    AddPhotos.as_view()),
    path('scan/',                     ScanFrame.as_view()),
    path('logs/',                     AttendanceLogView.as_view()),
    path('sessions/',                 AttendanceSessionView.as_view()),
    path('analytics/',                AnalyticsView.as_view()),
    path('export/',                   ExportCSV.as_view()),
    path('ai-insight/',               AIInsightView.as_view()),
    path('semantic-query/',           SemanticQueryView.as_view()),
    path('reset-presence/',           ResetPresence.as_view()),
    path('extract-multi/',            ExtractMultiFaces.as_view()),
]
