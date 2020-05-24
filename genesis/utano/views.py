from django.shortcuts import render
from django.http import HttpResponse
import speech_recognition as sr

# Create your views here.

def Home(request):
    return render(request, 'utano/home.html')
    # return HttpResponse ('<h1> Pizza Home </h1>')

def Falar(request):
    return render(request, 'utano/home2.html')
