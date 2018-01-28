from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^download.html$', views.download, name='download'),
    url(r'^read.html$', views.read, name='read'),
    url(r'^info.html$', views.show_info, name='info')
]
