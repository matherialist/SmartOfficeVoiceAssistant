from src.ActionClassifer import ActionClassifier
from src.VoiceAssistant import VoiceAssistant


class SmartOffice:
    def __init__(self, load_folder_path):
        self.action_classifier = ActionClassifier(
            load_folder_path,
            model_hub_path="https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/2",
            is_bert=True)
        self.voice_assistant = VoiceAssistant()

    def run(self, audio_file):
        text = self.voice_assistant.recognise_audio(audio_file)
        command = self.action_classifier.make_prediction(text)
        response = command['response']
        self.voice_assistant.voiceText(response, 'en')
        return command['command']
