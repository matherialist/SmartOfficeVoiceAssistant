from src.ActionClassifer import ActionClassifier
from src.VoiceAssistant import VoiceAssistant


class SmartOfficeOrchestrator:
    def __init__(self, load_folder_path, sess):
        self.action_classifier = ActionClassifier(load_folder_path, sess)
        self.voice_assistant = VoiceAssistant("C:/Program Files (x86)/Windows Media Player/wmplayer.exe")

    def run(self):
        text = None
        while not text:
            text = self.voice_assistant.keyWordActivate()
        intent_slots = self.action_classifier.make_prediction(text)
        response = intent_slots['response']
        self.voice_assistant.voiceText(response, 'en')
