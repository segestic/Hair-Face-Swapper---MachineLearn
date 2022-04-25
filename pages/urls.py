from django.urls import path, include
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    
    path('', views.upload_new, name='pages-index'),
    path('upload/', views.upload, name='pic-upload'),#local
    path('uploadcloud/', views.upload_cloud, name='cloud-upload'),#working
    path('uploadcloudnew/', views.upload_cloud_new, name='cloud-upload-new'),
    path('check/', views.check, name='pic-check'),
    path('up/', views.upload_fl, name='pic-upload-fl'),

    #nwend     
    path("list/", views.StyleListView.as_view(), name="style_list"),
    path("create/", views.StyleCreateView, name="style_create"),
    path("update/<uuid:uuid>/", views.StyleUpdateView.as_view(), name="style_update"),
    path("detail/<uuid:uuid>/", views.StyleDetailView.as_view(), name="style_detail"),
    # path("update/<int:pk>/", views.StyleUpdateView.as_view(), name="style_update"),
    # path("detail/<int:pk>/", views.StyleDetailView.as_view(), name="style_detail"),
]

if settings.DEBUG:
     urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
