FROM tensorflow/tensorflow:2.2.0-gpu

RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y ffmpeg x264 libx264-dev
RUN apt-get install -y python3-dev
RUN apt-get install -y libsm6 libxext6 libxrender-dev

WORKDIR /usr/src/app

COPY requirements.txt .
RUN pip3 install -r requirements.txt
ENTRYPOINT [ "python3" ]
CMD ["app.py"]

COPY . .


