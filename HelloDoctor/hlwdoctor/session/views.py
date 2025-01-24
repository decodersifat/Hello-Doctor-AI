from django.shortcuts import render
from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout, update_session_auth_hash
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm, PasswordChangeForm
from django.contrib import messages
from django.core.mail import send_mail
from django.contrib.auth.decorators import login_required
from django.urls import reverse



from django.shortcuts import render, redirect
from django.contrib.auth import login
from django.contrib import messages
from .forms import CustomUserCreationForm
from .models import Profile
def register(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            # Capture age and phone number
            age = form.cleaned_data['age']
            phone_number = form.cleaned_data['phone_number']
            
            # Create Profile instance with the captured data
            Profile.objects.create(user=user, age=age, phone_number=phone_number)
            
            messages.success(request, "Registration successful!")
            login(request, user)
            return redirect('profile')  # Redirect to the profile page after successful registration
        else:
            messages.error(request, "Invalid registration. Please try again.")
    else:
        form = CustomUserCreationForm()
    
    return render(request, 'home/registration.html', {'form': form})


def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request=request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            messages.success(request, f"Welcome {user.first_name}!")
            return redirect('/home/h/')
        else:
            messages.error(request, "Invalid credentials. Please try again.")
    else:
        form = AuthenticationForm()
    return render(request, 'home/login.html', {'form': form})


def logout_view(request):
    logout(request)
    messages.success(request, "You have been logged out.")
    return redirect('/home/h/')

@login_required
def profile(request):
    # Get the Profile for the currently logged-in user
    profile = Profile.objects.get(user=request.user)
    return render(request, 'home/profile.html', {'profile': profile})

'''
@login_required
def logout_view(request):
    logout(request)
    messages.success(request, "Logged out successfully.")
    return redirect('login')
'''



