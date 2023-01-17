FROM python:3.10-slim

COPY main.py /usr/src

RUN apt update && apt install git && \
    git clone https://github.com/openvinotoolkit/open_model_zoo.git && \
    pip install open_model_zoo/demos/object_detection_demo/python && \
    pip install open_model_zoo/tools/model_tools && \
    pip install openvino-dev && \
    cp open_model_zoo/demos/common/python/openvino/model_zoo/model_api /usr/src/ && \
    mkdir -p models/alexeyab && cd models/alexeyab && \
    curl -JOLk https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights && \
    curl -JOLk https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg && \
    cd - && \
    curl -JOLk https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names && \
    omz_downloader --name person-detection-0200 --output_dir models

WORKDIR /usr/src

ENTRYPOINT /usr/bin/python3

CMD main.py