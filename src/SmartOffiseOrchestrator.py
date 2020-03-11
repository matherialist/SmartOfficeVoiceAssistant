import json
from src.ActionClassifer import ActionClassifier
from src.SpeechToText import SpeechToText
from src.TextToSpeech import TextToSpeech


class SmartOfficeOrchestrator:
    def __init__(self, load_folder_path):
        self.action_classifier = ActionClassifier(load_folder_path)
        self.stt = SpeechToText()
        self.tts = TextToSpeech()

