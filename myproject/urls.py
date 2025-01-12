"""myproject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
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
from django.urls import path, include
from myapp.views import reverse_text, add_is_cool, test, UserRegisterView, UserLoginView, UserLogoutView



urlpatterns = [
    path('admin/', admin.site.urls),
    path('reverse_text/', reverse_text, name='reverse-text'),
    path('test/', test, name='test'),
    path('accounts/', include('allauth.urls')),
    path('add_is_cool/', add_is_cool, name='add-is-cool'),
    path('register/', UserRegisterView.as_view(), name='api_register'),
    path('login/', UserLoginView.as_view(), name='api_login'),
    path('logout/', UserLogoutView.as_view(), name='logout'),
]
