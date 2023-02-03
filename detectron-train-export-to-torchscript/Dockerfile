FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel

RUN apt-get -y update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get -y install git
RUN apt-get install -y wget

RUN pip install opencv-python tensorboard cmake onnx tqdm  tritonclient[all] jupyterlab
WORKDIR /app

RUN git clone https://github.com/facebookresearch/detectron2.git
RUN cd detectron2 && git checkout 4a5e6d79e626837a0317195131afaca64b3f4e2d && cd /app
RUN rm /app/detectron2/tools/deploy/export_model.py
COPY ./src/export_model.py /app/detectron2/tools/deploy/export_model.py
# RUN cp /app/src/export_model.py /app/detectron2/tools/deploy/export_model.py
RUN pip install -e detectron2

EXPOSE 8888
ENTRYPOINT jupyter lab --ip=0.0.0.0 --allow-root --port=8888 --NotebookApp.token='' --NotebookApp.password=''