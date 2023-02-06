FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel

RUN apt-get -y update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get -y install git


RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

RUN pip install opencv-python kserve
WORKDIR /app

RUN git clone https://github.com/facebookresearch/detectron2.git
RUN pip install -e detectron2

COPY ./app.py .

