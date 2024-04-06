# speech_recognition_module.py
import speech_recognition as sr

class SpeechRecognitionModule:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def transcribe_audio(self, audio_data):
        with sr.AudioFile(audio_data) as source:
            audio = self.recognizer.record(source)
        try:
            text = self.recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            print("Speech recognition could not understand audio")
            return ""
