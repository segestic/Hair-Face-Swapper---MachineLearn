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

from django.urls import reverse
import uuid

class Style(models.Model):
    uuid = models.UUIDField(unique=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=120, null=True)
    style = models.ImageField(max_length=255, null=True)
    date_created = models.DateField(auto_now_add=True, auto_now=False)
    last_modified = models.DateField(auto_now=True)


    def __str__(self):
        return self.name

    def get_update_url(self):
        return reverse("style_update", args=(self.uuid,))
        # return reverse("style_update", args=(self.pk,))
    
    def get_absolute_url(self):
        return reverse("style_detail", args=(self.uuid,))#args=(self.pk,))
        #return reverse("style_detail", args=(self.pk,))#args=(self.pk,))