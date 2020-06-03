import subprocess
import logging
import os
import time
from mutagen.mp3 import MP3
from gtts import gTTS
from datetime import datetime
import speech_recognition as sr
import winsound
import requests
import pyttsx3


class VoiceAssistant:
    def __init__(self):
        self.__logPath = None
        self.__curHash = None
        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', 170)  # setting up new voice rate
        self.tts.setProperty('volume', 1.0)  # setting up volume level  between 0 and 1
        voices = self.tts.getProperty('voices')  # getting details of current voice
        self.tts.setProperty('voice', voices[1].id)  # changing index, changes voices. 1 for female

    def voiceText(self, text, lang):
        self.__curHash = self.__genrateHash()
        self.__logData("{ \"text\": \"%s\", \"lang\": \"%s\"}" % (text, lang))
        self.__logData("%s\t%s\tGet" % (datetime.now().isoformat(), self.__curHash))
        # file = self.__genVoice(text, lang)
        self.__logData("%s\t%s\tCreated" % (datetime.now().isoformat(), self.__curHash))
        # openSubprocess = subprocess.Popen(self.__playerPath)
        # os.startfile(file.filename)
        # self.__logData("%s\t%s\tPlayed" % (datetime.now().isoformat(), self.__curHash))
        # time.sleep(file.info.length)
        # openSubprocess.kill()

        self.tts.say(text)
        # self.tts.save_to_file(text, 'text.mp3')
        self.tts.runAndWait()

        self.__logData("%s\t%s\tDeleted" % (datetime.now().isoformat(), self.__curHash))
        # os.remove(file.filename)
        # self.__deleteLog()

    def __genrateHash(self):
        intHash = abs(hash(time.ctime()))
        hexHash = '{:X}'.format(intHash)
        return hexHash

    def __genVoice(self, text, lang):
        tts = gTTS(text, lang=lang)
        fileName = "%s.mp3" % self.__curHash
        tts.save(fileName)
        return MP3(fileName)

    def __logData(self, log):
        logging.basicConfig(format="", filename="sample.log", level=logging.INFO)
        logging.info(log)

    def keyWordActivate(self):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Say <<Hey Assistant!>>")
            audio = r.listen(source)
        try:
            rec = r.recognize_google(audio)
            print(rec)
        except:
            return self.keyWordActivate()
        if rec == "hey assistant" or rec == "assistant":
            with sr.Microphone() as source:
                winsound.Beep(500, 700)
                audio = r.listen(source)
            try:
                text = r.recognize_google(audio)
                print(text)

                # data = {"text": text}
                # r = requests.post(url='http://127.0.0.1:5000/get-intent', json=data)
                # reply_text = r.json()['response']

                # self.voiceText(reply_text, "en")
                return text
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
                print("Try one more time please")
                return self.keyWordActivate()
            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))
                print("Try one more time please")
                return self.keyWordActivate()
        else:
            print("Try one more time please")
            return self.keyWordActivate()


if __name__ == "__main__":
    VA = VoiceAssistant()
    VA.keyWordActivate()
