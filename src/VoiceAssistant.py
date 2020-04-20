import subprocess
import logging
import os
import time
import winsound
import speech_recognition as sr
from mutagen.mp3 import MP3
from gtts import gTTS
from datetime import datetime


class VoiceAssistant:
    def __init__(self, playerPath):
        self.__playerPath = playerPath
        self.__logPath = None
        self.__curHash = None

    def voiceText(self, text, lang):
        self.__curHash = self.__generateHash()
        self.__logData("{ \"text\": \"%s\", \"lang\": \"%s\"}" % (text, lang))
        self.__logData("%s\t%s\tGet" % (datetime.now().isoformat(), self.__curHash))
        file = self.__genVoice(text, lang)
        self.__logData("%s\t%s\tCreated" % (datetime.now().isoformat(), self.__curHash))
        openSubprocess = subprocess.Popen(self.__playerPath)
        os.startfile(file.filename)
        self.__logData("%s\t%s\tPlayed" % (datetime.now().isoformat(), self.__curHash))
        time.sleep(file.info.length)
        # play mp3
        openSubprocess.kill()
        # delete mp3
        self.__logData("%s\t%s\tDeleted" % (datetime.now().isoformat(), self.__curHash))
        os.remove(file.filename)
        # self.__deleteLog()

    def __generateHash(self):
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
            print("Say <<Hey assistant!>>")
            audio = r.listen(source)

        if r.recognize_google(audio) == "hey assistant":
            with sr.Microphone() as source:
                winsound.Beep(500, 700)
                #print("Program activated! Talk!")
                audio = r.listen(source)
            try:
                text = r.recognize_google(audio)
                self.voiceText(text, "en")
                return text
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
                print("Try one more time please")
                return self.keyWordActivate()
            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))
                print("Try one more time please")
                return self.keyWordActivate()
        if r.recognize_google(audio) == "stop":
            print("Program stopped")
        else:
            print("Try one more time please")
            return self.keyWordActivate()


if __name__ == "__main__":
    VA = VoiceAssistant("C:/Program Files (x86)/Windows Media Player/wmplayer.exe")
    VA.keyWordActivate()
