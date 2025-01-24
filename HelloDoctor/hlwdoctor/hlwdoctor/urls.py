from django.contrib import admin
from django.urls import path, include
from session import views
from django.conf import settings
from django.conf.urls.static import static
#from . import views
urlpatterns = [
    path('admin/', admin.site.urls),
    path('home/',include('homeapp.urls')),
    path('sl/',include('service.urls')),
    path('r/', views.register, name='register'),
    path('l/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('profile/', views.profile, name='profile'),
    path('mri/',include('api.urls')),
    #path('download/<str:filename>/', views.download_pdf, name='download_file'),
    #path('logout/', views.logout_view, name='logout'),
]

# Serve media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
