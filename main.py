import base64
import os
from flask import Flask, request
from os import path
from flask_cors import CORS
from pydub import AudioSegment
from src.SmartOffice import SmartOffice


app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

config_relative_path = 'files'
config_path = path.join(path.dirname(__file__), config_relative_path)

so = SmartOffice("files")
AudioSegment.converter = r'ffmpeg\bin\ffmpeg.exe'
AudioSegment.ffprobe = r'ffmpeg\bin\ffprobe.exe'


@app.route('/get-intent', methods=['POST'])
def get_intent():
    data = request.get_json()
    audio_str = data["audio"].replace('data:audio/mpeg-3;base64,', '')
    audio = base64.b64decode(audio_str)
    with open('audio.mp3', 'wb') as f:
        f.write(audio)
    os.system("ffmpeg\\bin\\ffmpeg.exe -y -loglevel warning -i audio.mp3 -c:a flac audio.flac")
    prediction = so.run()
    os.remove('audio.flac')
    os.remove('audio.mp3')
    print(prediction)
    return prediction


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
