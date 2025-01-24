from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.
def machinelearning(request):
    return render(request,'home/index.html')

