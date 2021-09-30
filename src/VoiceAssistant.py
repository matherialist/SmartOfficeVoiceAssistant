import subprocess
import logging
import os
import time
# from mutagen.mp3 import MP3
# from gtts import gTTS
from datetime import datetime
# import speech_recognition as sr
# import winsound
# import pyttsx3


class VoiceAssistant:
    def __init__(self):
        self.__logPath = None
        self.__curHash = None
        # self.tts = pyttsx3.init()
        # self.tts.setProperty('rate', 170)  # setting up new voice rate
        # self.tts.setProperty('volume', 1.0)  # setting up volume level  between 0 and 1
        # self.voices = self.tts.getProperty('voices')  # getting details of current voice
        # self.tts.setProperty('voice', self.voices[1].id)  # changing index, changes voices. 1 for female

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

        # self.tts.say(text)
        # self.tts.runAndWait()
        # self.tts.stop()
        self.tts.save_to_file(text, 'audio.mp3')

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

    def recognise_audio(self, AUDIO_FILE):
        r = sr.Recognizer()
        # Convert Audio to Audio Source Format
        #audio_source = sr.AudioData(AUDIO_FILE, 48000, 1)
        with sr.AudioFile('audio.flac') as source:
            audio_source = r.record(source)
        # recognize speech using Google Speech Recognition
        text_en = "I don't understand you"
        text_ru = "I don't understand you"
        try:
            text_en = r.recognize_google(audio_source)
            text_ru = r.recognize_google(audio_source, language='ru')
            print("You said (en): " + text_en)
            print("You said (ru): " + text_ru)
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return text_en, text_ru

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
