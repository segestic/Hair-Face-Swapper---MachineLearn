from django.db import models

# Create your models here.
class Customer(models.Model):
    name = models.CharField(max_length=120, null=True)
    email = models.EmailField(max_length=255, blank=True, null=True)
    style = models.ImageField(max_length=255, null=True)
    pic = models.ImageField(max_length=255, null=True)
    merge = models.ImageField(max_length=255, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
   
    def __str__(self):
        return self.name