from django.urls import path
from .views import home, SwapUploadView, SwapUploadSerView, SwapUploadViewLimit
#never use from .views import * - cause of 2 days error, never import *

urlpatterns = [
    path('', home, name = 'home'),
    path('upload/swap', SwapUploadView.as_view(), name = 'hair-swap'),
    path('upload/ser', SwapUploadSerView.as_view(), name = 'hair-ser-swap'),
    path('upload/limit', SwapUploadViewLimit.as_view(), name = 'hair-swap-limit'),
]
