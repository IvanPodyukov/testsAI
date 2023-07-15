from django.urls import path
from webTests import views

urlpatterns = [
    path('', views.index, name='home'),
    path('form', views.form, name='form'),
    path('download', views.download, name='download')
]
