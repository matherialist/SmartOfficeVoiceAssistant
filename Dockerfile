FROM python:3.8

COPY ./requirements.txt /home/app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /home/app/requirements.txt
COPY . /home/app
WORKDIR /home/app

EXPOSE 5000
CMD python main.py