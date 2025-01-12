from rest_framework import serializers
from .models import UserInput, CustomUser

class UserInputSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserInput
        fields = '__all__'

class CustomUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomUser
        fields = ['email', 'password']
        extra_kwargs = {'password': {'write_only': True}}

    def create(self, validated_data):
        user = CustomUser.objects.create_user(**validated_data)
        return user