import base64
import os
import tensorflow as tf
from flask import Flask, request
from os import path
from tensorflow.python.keras.backend import set_session
from src.SmartOffise import SmartOffice

app = Flask(__name__)

config_relative_path = 'files'
config_path = path.join(path.dirname(__file__), config_relative_path)

graph = tf.compat.v1.get_default_graph()
sess = tf.compat.v1.Session()
tf.compat.v1.keras.backend.set_session(sess)

so = SmartOffice("files", sess)


@app.route('/get-intent', methods=['POST'])
def get_intent():
    with graph.as_default():
        set_session(sess)
        data = request.get_json()
        #audio = base64.decodebytes(data["audio"].encode('UTF-8'))
        audio = base64.b64decode(data["audio"])
        command = so.run(audio)
        with open('audio.wav', 'rb') as f:
            audio_data = f.read()
        audio = base64.b64encode(audio_data)
        os.remove('audio.wav')
    return {"command": command, "audio": audio.decode("UTF-8")}


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
