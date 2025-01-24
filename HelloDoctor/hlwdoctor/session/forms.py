from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Profile

class CustomUserCreationForm(UserCreationForm):
    first_name = forms.CharField(max_length=30, required=True, help_text='Required. Enter your first name.')
    last_name = forms.CharField(max_length=30, required=True, help_text='Required. Enter your last name.')
    email = forms.EmailField(required=True)
    age = forms.IntegerField(required=True, label="Age")
    phone_number = forms.CharField(max_length=15, required=True, label="Phone Number")

    class Meta:
        model = User
        fields = ['username', 'first_name', 'last_name', 'email', 'age', 'phone_number', 'password1', 'password2']

