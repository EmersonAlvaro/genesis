from django.urls import path
from . import views

urlpatterns = [
    path('', views.Home, name='Utano_Home'),
    path('output', views.output,name="script"),
    path('falar/', views.Falar, name='Utano_Falar'),
]