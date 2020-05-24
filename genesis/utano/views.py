from django.shortcuts import render
from django.http import HttpResponse
import requests
import speech_recognition as sr

# Create your views here.

def Home(request):
    return render(request, 'utano/home.html')

def button(request):

    return render(request,'utano/home.html')

 
def output(request):

#Habilita o microfone para ouvir o usuario
	microfone = sr.Recognizer()
	with sr.Microphone() as source:
		#Chama a funcao de reducao de ruido disponivel na speech_recognition
		microfone.adjust_for_ambient_noise(source)
		#Avisa ao usuario que esta pronto para ouvir
		print("Diga alguma coisa: ")
		#Armazena a informacao de audio na variavel
		audio = microfone.listen(source)


	try:
		#Passa o audio para o reconhecedor de padroes do speech_recognition
		data = microfone.recognize_google(audio,language='pt-BR')
		#Após alguns segundos, retorna a frase falada
		print("Você disse: " + data)

		#Caso nao tenha reconhecido o padrao de fala, exibe esta mensagem
	except sr.UnkownValueError:
		print("Não entendi")

	return render(request,'utano/home.html',{'data':data})
	
#data = output()
