# Create your models here.
from django.db import models




class Customer(models.Model):

    customer_id = models.IntegerField()  
    recency = models.IntegerField()  
    frequency = models.IntegerField()  
    monetary = models.DecimalField(max_digits=10, decimal_places=2) 
    clusters = models.IntegerField()  
    group = models.CharField(max_length=30)  



class UploadedFileName(models.Model):
    file_name = models.CharField(max_length=255, default=None) 
    file_identifier = models.IntegerField(default=1) 
    
    def __str__(self):
        return self.file_name