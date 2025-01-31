from django.contrib import admin
from .models import *
# Register your models here.



@admin.register(Customer)
class CustomerAdmin(admin.ModelAdmin):
    list_display = ['customer_id', 'recency', 'frequency', 'monetary', 'clusters', 'group']

