from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^series/$', views.series, name='series'),
    url(r'^init/$', views.init, name='init'),
    url(r'^points/$', views.points, name='points'),
    url(r'^addPoint/$', views.add_point, name='addPoint'),
    url(r'^removePoint/$', views.remove_point, name='removePoint'),
    url(r'^train/$', views.train, name='train'),
]
