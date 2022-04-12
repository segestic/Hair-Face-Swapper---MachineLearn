from django.urls import path, include
from . import views

urlpatterns = [
    
    path('', views.index, name='pages-index'),
    # path('create/', views.create, name='posts.create',
    # path('<int:id>/', views.show, name='posts.show'),
    # path('<int:id>/edit/', views.edit, name='posts.edit'),
    # path('<int:id>/delete/', views.delete, name='posts.delete')
]

