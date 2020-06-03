from src.ActionClassifer import ActionClassifier
from src.VoiceAssistant import VoiceAssistant


class SmartOfficeOrchestrator:
    def __init__(self, load_folder_path, sess):
        self.action_classifier = ActionClassifier(load_folder_path, sess)
        self.voice_assistant = VoiceAssistant()

    def run(self):
        text = self.voice_assistant.keyWordActivate()
        intent_slots = self.action_classifier.make_prediction(text)
        response = intent_slots['response']
        self.voice_assistant.voiceText(response, 'en')
