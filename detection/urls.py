from django.urls import path
from . import views
app_name = "detection"
urlpatterns = [
 path('', views.upload_video, name='upload_video'),
 ]