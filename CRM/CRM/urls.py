"""
URL configuration for CRM project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
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
from django.urls import path
from customer import views

urlpatterns = [ 
    path('admin/', admin.site.urls),
    path('',views.home),
    path('retail_data',views.customer_retail_data_showing, name="retail_data"),
    path('segmentation',views.cutomize_segmentation, name="segmentation"),
    path('analysis',views.RFM_analyzer, name="analysis"),
    path('analytics_dashboard',views.analytics_dashboard, name="analytics_dashboard"),
    path('error/', views.error_page, name='error'),


]
