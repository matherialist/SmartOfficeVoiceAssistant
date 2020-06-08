from src.ActionClassifer import ActionClassifier
from src.VoiceAssistant import VoiceAssistant
import tensorflow as tf
from tensorflow.python.keras.backend import set_session


class SmartOffice:
    def __init__(self, load_folder_path, sess):
        self.action_classifier = ActionClassifier(load_folder_path, sess)
        self.voice_assistant = VoiceAssistant()

    def run(self, audio_file):
        text = self.voice_assistant.recognise_audio(audio_file)
        command = self.action_classifier.make_prediction(text)
        response = command['response']
        self.voice_assistant.voiceText(response, 'en')
        return command['command']


graph = tf.compat.v1.get_default_graph()
sess = tf.compat.v1.Session()
tf.compat.v1.keras.backend.set_session(sess)

if __name__ == '__main__':
    so = SmartOffice('C:/Users/Lagrange/Desktop/SmartOfficeVoiceAssistant/files', sess)
    so.run('example.mp3')
