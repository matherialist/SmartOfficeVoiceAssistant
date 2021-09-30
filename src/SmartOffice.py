from src.ActionClassifer import ActionClassifier
from src.VoiceAssistant import VoiceAssistant


class SmartOffice:
    def __init__(self, load_folder_path='files'):
        self.action_classifier = ActionClassifier(
            load_folder_path,
            model_hub_path="https://tfhub.dev/tensorflow/albert_en_base/3",
            is_bert=False)
        self.voice_assistant = VoiceAssistant()

    def get_command(self, text):
        command, response = self.action_classifier.make_prediction(text)
        return command, response
