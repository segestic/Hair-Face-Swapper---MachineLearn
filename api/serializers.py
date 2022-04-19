from rest_framework import serializers
# from rest_framework.serializers import ModelSerializer
from pages.models import Customer


class ImageSerializer(serializers.Serializer):
    style = serializers.ImageField(max_length=None, use_url=True, allow_null=True, required=False)
    pic = serializers.ImageField(max_length=None, use_url=True, allow_null=True, required=False)
  
    def create(self, validated_data):
        return Customer.objects.create(**validated_data)

    #not useddddddddd
    def get_image(self, obj):
        return self.context['request'].build_absolute_url(obj.picture.url)