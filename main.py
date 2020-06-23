import base64
import os
from flask import Flask, request
from os import path
from src.SmartOffice import SmartOffice
from flask_cors import CORS
from pydub import AudioSegment


app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

config_relative_path = 'files'
config_path = path.join(path.dirname(__file__), config_relative_path)

so = SmartOffice("files")

AudioSegment.converter = r"C:\Users\Lagrange\Desktop\SmartOfficeVoiceAssistant\ffmpeg\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\Users\Lagrange\Desktop\SmartOfficeVoiceAssistant\ffmpeg\bin\ffprobe.exe"


@app.route('/get-intent', methods=['POST'])
def get_intent():
    data = request.get_json()
    #audio = base64.decodebytes(data["audio"].encode('UTF-8'))
    audio_str = data["audio"].replace('data:audio/mpeg-3;base64,', '')
    audio = base64.b64decode(audio_str)
    with open('audio.mp3', 'wb') as f:
        f.write(audio)
    os.system("ffmpeg\\bin\\ffmpeg.exe -y -loglevel warning -i audio.mp3 -c:a flac audio.flac")
    with open('audio.flac', 'rb') as f:
        audio = f.read()
    # flac_audio = AudioSegment.from_file("audio.mp3", format="mp3")
    # flac_audio.export("audio.flac", format="flac")
    command = so.run(audio)
    with open('audio.mp3', 'rb') as f:
        audio_data = f.read()
    audio = base64.b64encode(audio_data)
    os.remove('audio.flac')
    #os.remove('audio.mp3')
    print(command)
    return {"command": command, "response": audio.decode("UTF-8")}


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
