import speech_recognition as sr


class SpeechToText:

    def __init__(self):
        self.rec = sr.Recognizer()
        self.rec.dynamic_energy_threshold = True
        self.rec.dynamic_energy_adjustment_damping = 0.15
        # minimum length of silence (in seconds) that will register as the end of a phrase
        self.rec.pause_threshold = 0.8
        print('Silence!')
        with sr.Microphone() as source:
            self.rec.adjust_for_ambient_noise(source, 3)

    def speechRecognition(self):
        # obtain audio from the microphone
        with sr.Microphone() as source:
            print("Say something!")
            audio = self.r.listen(source)

        # recognize speech using Google Speech Recognition
        try:
            print("Google Speech Recognition thinks you said: " + self.rec.recognize_google(audio))
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio.")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return self.rec.recognize_google(audio)
