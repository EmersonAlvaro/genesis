from django.shortcuts import render
from django.http import HttpResponse
import requests
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound


# Create your views here.

def Home(request):
    return render(request, 'utano/home.html')

def button(request):

    return render(request,'utano/home2.html')

#Funcao responsavel por falar 
def cria_audio(audio):
	tts = gTTS(audio,lang='en-us')
	#Salva o arquivo de audio
	tts.save('utano/hello.mp3')
	print("Estou aprendendo o que você disse...")
	#Da play ao audio
	playsound('utano/hello.mp3')

 
def output(request):

#Habilita o microfone para ouvir o usuario
	microfone = sr.Recognizer()
	with sr.Microphone() as source:

		datatemp = "Hi, my name is Utano, I’m Chatbot, tell me how you feel and I’ll try to tell if you have cancer or not"

		cria_audio(datatemp)

		#Chama a funcao de reducao de ruido disponivel na speech_recognition
		microfone.adjust_for_ambient_noise(source)
		#Avisa ao usuario que esta pronto para ouvir
		print("Diga alguma coisa: ")
		#Armazena a informacao de audio na variavel
		audio = microfone.listen(source)


	try:
		#Passa o audio para o reconhecedor de padroes do speech_recognition
		data = microfone.recognize_google(audio,language='en-us')
		#Após alguns segundos, retorna a frase falada
		print("Você disse: " + data)

		#Caso nao tenha reconhecido o padrao de fala, exibe esta mensagem
	except sr.UnkownValueError:
		print("Não entendi")

	return render(request,'utano/home2.html',{'data':data})
	
#data = output()

def Falar(request):
    return render(request, 'utano/home2.html')
