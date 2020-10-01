import logging
import time
from datetime import datetime
import speech_recognition as sr
# import winsound
# import pyttsx3


class VoiceAssistant:
    def __init__(self):
        self._log_path = None
        self._cur_hash = None
        # self.tts = pyttsx3.init()
        # self.tts.setProperty('rate', 170)  # setting up new voice rate
        # self.tts.setProperty('volume', 1.0)  # setting up volume level  between 0 and 1
        # self.voices = self.tts.getProperty('voices')  # getting details of current voice
        # self.tts.setProperty('voice', self.voices[1].id)  # changing index, changes voices. 1 for female

    def voice_text(self, text, lang):
        self._cur_hash = self._genrate_hash()
        self._log_data("{ \"text\": \"%s\", \"lang\": \"%s\"}" % (text, lang))
        self._log_data("%s\t%s\tGet" % (datetime.now().isoformat(), self._cur_hash))
        self.tts.save_to_file(text, 'audio.mp3')
        self._log_data("%s\t%s\tDeleted" % (datetime.now().isoformat(), self._cur_hash))

    def _genrate_hash(self):
        int_hash = abs(hash(time.ctime()))
        hex_hash = '{:X}'.format(int_hash)
        return hex_hash

    def _log_data(self, log):
        logging.basicConfig(format="", filename="sample.log", level=logging.INFO)
        logging.info(log)

    def recognise_audio(self):
        r = sr.Recognizer()
        with sr.AudioFile('audio.flac') as source:
            audio_source = r.record(source)
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

    def keyword_activate(self):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Say <<Hey Assistant!>>")
            audio = r.listen(source)
        try:
            rec = r.recognize_google(audio)
            print(rec)
        except:
            return self.keyword_activate()
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
                return self.keyword_activate()
            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))
                print("Try one more time please")
                return self.keyword_activate()
        else:
            print("Try one more time please")
            return self.keyword_activate()
