import subprocess
import logging
import os
import time
from mutagen.mp3 import MP3
from gtts import gTTS
from datetime import datetime
import json


class TextToSpeech:
    def __init__(self, load_folder_path):
        self.__playerPath = self.readConfig(load_folder_path)
        self.__logPath = None
        self.__curHash = None

    def readConfig(self, load_folder_path):
        with open(load_folder_path +'/params.json') as f:
            data = json.load(f)
        return data['player path']

    def voiceText(self, text, lang):
        self.__curHash = self.__genrateHash()
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
