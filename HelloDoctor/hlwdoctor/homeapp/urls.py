from django.urls import path
from. import views
from api.views import download_pdf

urlpatterns = [
    path('h/',views.machinelearning),
    path('media/reports/<str:filename>/', download_pdf, name='download_pdf'),
]
