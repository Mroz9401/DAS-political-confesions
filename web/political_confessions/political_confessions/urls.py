"""
URL configuration for political_confessions project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from frontend.views import homepage_view, chat_view, llm_response_view, user_login_view

app_name = 'frontend'

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', user_login_view, name='login'),
    path('homepage/', homepage_view, name='homepage'),
    path('chat_view/<int:politician_id>/', chat_view, name='chat_view'),
    path('get_llm_response/', llm_response_view, name='get_llm_response'),
]

if settings.DEBUG:
        urlpatterns += static(settings.MEDIA_URL,
                              document_root=settings.MEDIA_ROOT)

