from django.urls import path
from. import views

urlpatterns = [
    path('mri_input/',views.mri_input_form),
    path('predict/', views.predict, name='predict'),
    path('download/<str:filename>/', views.download_pdf, name='download_file'),
    ]
