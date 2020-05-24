from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.

def Home(request):
    return render(request, 'utano/home.html')
    # return HttpResponse ('<h1> Pizza Home </h1>')
