import json
import requests
import speech_recognition as sr
import winsound


def keyword_activate():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say <<Hey Assistant!>>")
        audio = r.listen(source)
    try:
        rec = r.recognize_google(audio)
        print(rec)
    except:
        return keyword_activate()
    if rec == "hey assistant" or "assistant" in rec.split():
        with sr.Microphone() as source:
            winsound.Beep(500, 700)
            audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            print(text)
            return text
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
            print("Try one more time please")
            return keyword_activate()
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
            print("Try one more time please")
            return keyword_activate()
    else:
        print("Try one more time please")
        return keyword_activate()


if __name__ == '__main__':
    while True:
        text = keyword_activate()
        if isinstance(text, str):
            r = requests.get('http://localhost:5000/recognise-command',
                             json={'text': text})
            ans = json.loads(r.text)
            print(ans)
