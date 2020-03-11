from src.ActionClassifer import ActionClassifier
from src.SpeechToText import SpeechToText
from src.TextToSpeech import TextToSpeech


class SmartOfficeOrchestrator:
    def __init__(self, load_folder_path):
        self.action_classifier = ActionClassifier(load_folder_path)
        self.stt = SpeechToText()
        self.tts = TextToSpeech(load_folder_path)

    def run(self):
        text = self.stt.speechRecognition()
        intent_slots = self.action_classifier.get_response(text)
        response = 'roger that, i will '
        for i in intent_slots:
            response += ' ' + i
        self.tts.voiceText(response, 'en')
