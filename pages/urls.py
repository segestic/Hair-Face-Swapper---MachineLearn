from django.urls import path, include
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    
    path('', views.upload_cloud, name='pages-index'),
    path('upload/', views.upload, name='pic-upload'),#local
    path('uploadcloud/', views.upload_cloud, name='cloud-upload'),
    path('check/', views.check, name='pic-check'),
    path('up/', views.upload_fl, name='pic-upload-fl'),
]

if settings.DEBUG:
     urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
