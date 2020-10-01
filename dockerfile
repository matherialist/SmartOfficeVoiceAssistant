FROM python:3.7

COPY . /home/app

RUN apt-get -y update
RUN pip install --upgrade pip
RUN apt-get -y install ffmpeg

RUN pip install scikit-learn==0.22.2
RUN pip install tensorflow==2.1.0
RUN pip install tensorflow-hub==0.8.0
RUN pip install sentencepiece==0.1.85
RUN pip install speechrecognition==3.8.1
RUN pip install Flask==1.1.1
RUN pip install flask_cors==3.0.8

EXPOSE 5000

WORKDIR /home/app

CMD python main.py