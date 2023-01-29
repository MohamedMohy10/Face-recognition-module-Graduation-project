import playsound
from gtts import gTTS
import os


def speak(text):
    text = 'الشخص الذي امامك هو' + text
    tts = gTTS(text=text, lang="ar")  #transform this text into an audio file
    filename = "voice.mp3"
    tts.save(filename)
    audio_file = os.path.dirname(__file__) + '\\voice.mp3'
    playsound.playsound(audio_file)