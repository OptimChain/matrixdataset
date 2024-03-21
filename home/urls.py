from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('get-dataset_data', views.get_dataset_data, name='get_dataset_data'),
    path('get-all-filenames', views.get_filenames, name='get_filenames'),
    path('get-all-buckets', views.get_all_buckets, name='get_all_buckets'),
    
    ]
